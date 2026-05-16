"""
Extract equilibrium Green's functions in time domain.

This script computes:
- g^R(τ): Retarded Green's function as function of relative time τ
- g^A(τ): Advanced Green's function
- f(τ): Thermal distribution function
- g^K(τ): Keldysh Green's function

All functions are extracted from equilibrium states computed using
usadel_keldysh_evolution.py methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from usadel_keldysh_evolution import UsadelKeldyshEvolution


def main():
    print("="*70)
    print("Extract Equilibrium Green's Functions in Time Domain")
    print("="*70)
    print()

    # ========== Define Parameters ==========
    # Single consistent grid
    grid_parameters = {
        'time_sampling': 1500,       # Number of time points
        'time_duration': 2 * np.pi * 5,  # Time duration T_max
        'eta': 0.2                   # Broadening parameter
    }

    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': 0.3,
        'eta': 0.2
    }

    print("Grid parameters:")
    print(f"  Time points: {grid_parameters['time_sampling']}")
    print(f"  Time duration: {grid_parameters['time_duration']:.4f}")
    print(f"  Broadening η: {grid_parameters['eta']}")
    print()

    print("System parameters:")
    print(f"  Critical temperature: {system_parameters['critical_temperature']}")
    print(f"  Temperature: {system_parameters['temperature']}")
    print()

    # ========== Create Evolution Object ==========
    print("Creating evolution object...")
    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)

    print(f"  Time grid size: {evolution.ntpoints}")
    print(f"  Omega grid size: {len(evolution.omega_grid)}")
    print(f"  Delta t: {evolution.delta_t:.6f}")
    print()

    # ========== Generate Equilibrium State ==========
    print("Computing equilibrium state...")
    initial_state, gr_tau, gk_tau = evolution.generate_initial_state()

    print(f"  g^R(τ) shape: {gr_tau.data.shape}")
    print(f"  g^K(τ) shape: {gk_tau.data.shape}")
    print()

    # ========== Generate Thermal Distribution ==========
    print("Generating thermal distribution...")
    evolution.get_thermal_occupation(system_parameters['temperature'])
    f_two_time = evolution.thermal_dist  # f(t,t')

    print(f"  f(t,t') shape: {f_two_time.data.shape}")
    print()

    # ========== Compute f(τ) directly on FFT tau grid ==========
    print("Computing f(τ) = -iT/sinh(πτT) directly on FFT tau grid...")
    # Use analytical formula on the same tau grid as Green's functions

    # Get omega grid and compute corresponding tau grid from FFT
    omega_grid = evolution.omega_grid
    d_omega = omega_grid[1] - omega_grid[0]
    n_omega = len(omega_grid)

    # Tau grid from FFT: symmetric around 0
    tau_grid_fft = np.linspace(-np.pi/d_omega, np.pi/d_omega, n_omega)

    # Compute f(τ) = -i T / sinh(π τ T) on the FFT tau grid
    T = system_parameters['temperature']
    f_extended = np.zeros(n_omega, dtype=complex)

    # Avoid division by zero at τ = 0
    mask = np.abs(tau_grid_fft) > 1e-10
    f_extended[mask] = -1j * T / np.sinh(np.pi * tau_grid_fft[mask] * T)

    # For τ=0, test both approaches
    # Option A: L'Hôpital's limit = -i/π
    # Option B: Set to 0 (analytical choice)
    USE_LHOPITAL = False  # Set to False to use analytical f(τ=0) = 0

    if USE_LHOPITAL:
        f_extended[n_omega//2] = -1j / np.pi
        f_tau_0_label = "L'Hôpital (-i/π)"
    else:
        f_extended[n_omega//2] = 0.0
        f_tau_0_label = "Analytical (0)"

    # Create NambuKeldyshTensor (identity in Nambu space)
    from nambu_keldysh_class import NambuKeldyshTensor
    f_tau = NambuKeldyshTensor(f_extended, pauli_channel=0)

    print(f"  f(τ) shape: {f_tau.data.shape}")
    print(f"  Tau grid from FFT: [{tau_grid_fft[0]:.4f}, {tau_grid_fft[-1]:.4f}]")
    print(f"  f(τ=0) = {f_extended[n_omega//2]:.6e} ({f_tau_0_label})")

    # Investigate where the ±15i values come from after ifftshift
    print(f"\n  Investigating ifftshift corruption...")
    f_before = f_extended.copy()
    f_after = np.fft.ifftshift(f_extended)

    # Find indices with large values in f_after
    large_threshold = 10.0  # Find values > 10
    large_indices_after = np.where(np.abs(f_after) > large_threshold)[0]

    print(f"  Found {len(large_indices_after)} positions with |f| > {large_threshold} after ifftshift:")
    for idx_after in large_indices_after[:5]:  # Show first 5
        # Find where this came from before ifftshift
        # ifftshift shifts by N//2, so: after[i] = before[(i + N//2) % N]
        # Therefore: before[j] = after[(j - N//2) % N]
        idx_before = (idx_after + n_omega//2) % n_omega
        tau_before = tau_grid_fft[idx_before]
        tau_after_label = f"position {idx_after}"

        print(f"    After ifftshift[{idx_after}] = {f_after[idx_after]:.3e}")
        print(f"      ← came from before[{idx_before}] (τ={tau_before:.3f}) = {f_before[idx_before]:.3e}")

    print()

    # ========== Compute g^A(τ) from g^R(τ) ==========
    print("Computing g^A(τ) from g^R(τ)...")
    # For equilibrium in time domain: g^A(τ) = -[g^R(-τ)]^†
    # With Nambu structure: g^A = -τ₃ [g^R(-τ)]^† τ₃

    from nambu_keldysh_class import NambuKeldyshTensor

    n_tau = gr_tau.data.shape[-1]
    ga_tau_data = np.zeros_like(gr_tau.data)

    # Time reversal and complex conjugation
    for i in range(2):
        for j in range(2):
            ga_tau_data[i, j, :] = np.conj(gr_tau.data[i, j, ::-1])

    # Apply tau_3 involution: -tau_3 ... tau_3
    # This swaps (0,1) <-> (1,0) and changes signs
    ga_tau_data_new = np.zeros_like(ga_tau_data)
    ga_tau_data_new[0, 0, :] = -ga_tau_data[0, 0, :]
    ga_tau_data_new[0, 1, :] = ga_tau_data[1, 0, :]
    ga_tau_data_new[1, 0, :] = ga_tau_data[0, 1, :]
    ga_tau_data_new[1, 1, :] = -ga_tau_data[1, 1, :]

    ga_tau = NambuKeldyshTensor(ga_tau_data_new)

    print(f"  g^A(τ) shape: {ga_tau.data.shape}")
    print()

    # ========== Summary ==========
    print("="*70)
    print("Extracted Functions")
    print("="*70)
    print(f"g^R(τ): Retarded GF, shape {gr_tau.data.shape}")
    print(f"g^A(τ): Advanced GF, shape {ga_tau.data.shape}")
    print(f"f(τ): Thermal distribution, shape {f_tau.data.shape}")
    print(f"g^K(τ): Keldysh GF, shape {gk_tau.data.shape}")
    print()

    # ========== Plotting ==========
    print("Generating plots...")

    # Use the FFT tau grid for plotting (already computed above)
    tau_axis = tau_grid_fft

    # Extract components for plotting (τ_2 component)
    gr_tau_real = np.real(gr_tau.trace(2)) / 2
    ga_tau_real = np.real(ga_tau.trace(2)) / 2
    gk_tau_imag = np.imag(gk_tau.trace(2)) / 2
    f_tau_imag = np.imag(f_tau.trace(0)) / 2  # τ_0 component for scalar f

    # Plot individual functions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # g^R
    ax = axes[0, 0]
    ax.plot(tau_axis, gr_tau_real, linewidth=2, color='blue')
    ax.set_xlabel('τ (relative time)', fontsize=12)
    ax.set_ylabel('Re[g^R(τ)]  (τ_2 component)', fontsize=12)
    ax.set_title('Retarded Green\'s Function', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # g^A
    ax = axes[0, 1]
    ax.plot(tau_axis, ga_tau_real, linewidth=2, color='red')
    ax.set_xlabel('τ (relative time)', fontsize=12)
    ax.set_ylabel('Re[g^A(τ)]  (τ_2 component)', fontsize=12)
    ax.set_title('Advanced Green\'s Function', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # g^K
    ax = axes[1, 0]
    ax.plot(tau_axis, gk_tau_imag, linewidth=2, color='green')
    ax.set_xlabel('τ (relative time)', fontsize=12)
    ax.set_ylabel('Im[g^K(τ)]  (τ_2 component)', fontsize=12)
    ax.set_title('Keldysh Green\'s Function', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # f(τ)
    ax = axes[1, 1]
    ax.plot(tau_axis, f_tau_imag, linewidth=2, color='orange')
    ax.set_xlabel('τ (relative time)', fontsize=12)
    ax.set_ylabel('Im[f(τ)]', fontsize=12)
    ax.set_title('Thermal Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== Combined Plot ==========
    print("\nGenerating combined plot of all functions...")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(tau_axis, gr_tau_real, linewidth=2, color='blue',
            label='Re[g^R(τ)]', alpha=0.8)
    ax.plot(tau_axis, ga_tau_real, linewidth=2, color='red',
            label='Re[g^A(τ)]', alpha=0.8)
    ax.plot(tau_axis, gk_tau_imag, linewidth=2, color='green',
            label='Im[g^K(τ)]', alpha=0.8)

    ax2 = ax.twinx()
    ax2.plot(tau_axis, f_tau_imag, linewidth=2, color='orange',
             label='Im[f(τ)]', alpha=0.7, linestyle='--')
    ax2.set_ylabel('Im[f(τ)]', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    ax.set_xlabel('τ (relative time)', fontsize=12)
    ax.set_ylabel('Green\'s Functions (τ_2 component)', fontsize=12)
    ax.set_title('All Equilibrium Functions', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== Discrete Circular Convolution ==========
    print("\nComputing discrete circular convolution: g^R ⊛ f - f ⊛ g^A...")

    n_g = gr_tau.data.shape[-1]
    n_f = f_tau.data.shape[-1]

    print(f"  Input shapes: g^R = {gr_tau.data.shape}, g^A = {ga_tau.data.shape}, f = {f_tau.data.shape}")

    # Extract scalar thermal distribution (diagonal element, tau_0 component)
    f_scalar = f_tau.data[0, 0, :]

    # Check if sizes match
    if n_f != n_g:
        print(f"  WARNING: Size mismatch - g has {n_g} points, f has {n_f} points")
        print(f"  Using the minimum size: {min(n_g, n_f)}")
        N = min(n_g, n_f)
    else:
        N = n_g
        print(f"  Grid sizes match: N = {N} points")

    # Use the FFT tau spacing for convolution normalization
    d_tau = tau_grid_fft[1] - tau_grid_fft[0]
    print(f"  Using d_tau = {d_tau:.6f} for convolution normalization")

    print(f"  Computing circular convolution with N = {N} points...")
    print(f"  NOTE: Computing convolution on fftshifted (centered) arrays DIRECTLY")

    # Compute g^R ⊛ f with detailed diagnostics at τ=0
    gr_conv_f_data = np.zeros((2, 2, N), dtype=complex)

    # For diagnostic: track contributions to center point
    diagnostic_contributions_gr_f = []

    for i in range(2):
        for j in range(2):
            # Use arrays in their current fftshifted (centered) state
            # NO ifftshift/fftshift - compute convolution directly!
            x = gr_tau.data[i, j, :N]
            h = f_scalar[:N]

            result = np.zeros(N, dtype=complex)
            for n in range(N):
                conv_sum = 0.0
                for m in range(N):
                    term = x[m] * h[(n - m) % N]
                    conv_sum += term

                    # Diagnostic: Save contributions to center point for [0,1] component
                    if n == N//2 and i == 0 and j == 1:
                        # Store (m, x[m], h[(n-m)%N], term)
                        diagnostic_contributions_gr_f.append((m, x[m], h[(N//2 - m) % N], term))

                result[n] = conv_sum * d_tau

            # Store result directly (no fftshift needed)
            gr_conv_f_data[i, j, :] = result

    print(f"  Computed g^R ⊛ f")

    # Compute f ⊛ g^A with detailed diagnostics at τ=0
    f_conv_ga_data = np.zeros((2, 2, N), dtype=complex)

    # For diagnostic: track contributions to center point
    diagnostic_contributions_f_ga = []

    for i in range(2):
        for j in range(2):
            # Use arrays in their current fftshifted (centered) state
            # NO ifftshift/fftshift - compute convolution directly!
            x = f_scalar[:N]
            h = ga_tau.data[i, j, :N]

            result = np.zeros(N, dtype=complex)
            for n in range(N):
                conv_sum = 0.0
                for m in range(N):
                    term = x[m] * h[(n - m) % N]
                    conv_sum += term

                    # Diagnostic: Save contributions to center point for [0,1] component
                    if n == N//2 and i == 0 and j == 1:
                        diagnostic_contributions_f_ga.append((m, x[m], h[(N//2 - m) % N], term))

                result[n] = conv_sum * d_tau

            # Store result directly (no fftshift needed)
            f_conv_ga_data[i, j, :] = result

    print(f"  Computed f ⊛ g^A")

    # ========== Analyze Convolution Terms at τ=0 ==========
    print(f"\n  Analyzing convolution sum at τ=0 (component [0,1])...")

    # Find largest contributions to g^R ⊛ f
    contrib_gr_f_sorted = sorted(diagnostic_contributions_gr_f, key=lambda x: abs(x[3]), reverse=True)
    print(f"\n  Top 10 contributions to (g^R ⊛ f)[τ=0]:")
    for idx, (m, x_val, h_val, term) in enumerate(contrib_gr_f_sorted[:10]):
        # Convert m back to shifted index for tau value
        m_shifted = (m + N//2) % N
        tau_val = tau_grid_fft[m_shifted]
        print(f"    #{idx+1}: m={m} (τ≈{tau_val:.3f}): g^R={x_val:.3e} × f={h_val:.3e} = {term:.3e}")

    # Find largest contributions to f ⊛ g^A
    contrib_f_ga_sorted = sorted(diagnostic_contributions_f_ga, key=lambda x: abs(x[3]), reverse=True)
    print(f"\n  Top 10 contributions to (f ⊛ g^A)[τ=0]:")
    for idx, (m, x_val, h_val, term) in enumerate(contrib_f_ga_sorted[:10]):
        m_shifted = (m + N//2) % N
        tau_val = tau_grid_fft[m_shifted]
        print(f"    #{idx+1}: m={m} (τ≈{tau_val:.3f}): f={x_val:.3e} × g^A={h_val:.3e} = {term:.3e}")

    # Compute g^K = g^R ⊛ f - f ⊛ g^A
    gk_conv_data = gr_conv_f_data - f_conv_ga_data

    print(f"\n  Convolution complete.")

    # Create NambuKeldyshTensor
    gk_conv = NambuKeldyshTensor(gk_conv_data)

    print(f"  Convolution result shape: {gk_conv.data.shape}")
    print(f"  Convolution result [0,1] max: {np.max(np.abs(gk_conv.data[0, 1, :])):.6e}")
    print(f"  Convolution result [1,0] max: {np.max(np.abs(gk_conv.data[1, 0, :])):.6e}")

    # Extract traces
    gk_conv_tau2_imag = np.imag(gk_conv.trace(2)) / 2
    gk_conv_tau3_imag = np.imag(gk_conv.trace(3)) / 2

    print(f"  Convolution tau2 trace max: {np.max(np.abs(gk_conv_tau2_imag)):.6e}")
    print(f"  Convolution tau3 trace max: {np.max(np.abs(gk_conv_tau3_imag)):.6e}")

    # For g^K from equilibrium, use traces
    gk_tau2_imag_eq = np.imag(gk_tau.trace(2)) / 2
    gk_tau3_imag_eq = np.imag(gk_tau.trace(3)) / 2

    print(f"  Equilibrium g^K tau2 trace max: {np.max(np.abs(gk_tau2_imag_eq)):.6e}")
    print(f"  Equilibrium g^K tau3 trace max: {np.max(np.abs(gk_tau3_imag_eq)):.6e}")

    # ========== Diagnostic: τ=0 Point Analysis ==========
    print("\n" + "="*70)
    print("DIAGNOSTIC: τ=0 Point Analysis")
    print("="*70)

    # Center index (τ=0 point)
    n_center = N // 2

    # Check f(τ=0) values
    print(f"\nf(τ=0) values:")
    print(f"  From direct computation: {f_tau.data[0, 0, n_center]:.6e}")
    print(f"  L'Hôpital limit (-i/π): {-1j/np.pi:.6e}")
    print(f"  Difference: {f_tau.data[0, 0, n_center] - (-1j/np.pi):.6e}")

    # Check all Nambu components at τ=0 for equilibrium g^K
    print(f"\ng^K(τ=0) Nambu components (equilibrium):")
    for i in range(2):
        for j in range(2):
            val = gk_tau.data[i, j, n_center]
            print(f"  [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    # Check all Nambu components at τ=0 for convolved g^K
    print(f"\ng^K(τ=0) Nambu components (convolution):")
    for i in range(2):
        for j in range(2):
            val = gk_conv.data[i, j, n_center]
            print(f"  [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    # Compute and display the difference
    print(f"\nDifference (convolution - equilibrium) at τ=0:")
    for i in range(2):
        for j in range(2):
            diff = gk_conv.data[i, j, n_center] - gk_tau.data[i, j, n_center]
            print(f"  [{i},{j}]: {diff.real:+.6e} {diff.imag:+.6e}j")

    # Check individual convolution terms at τ=0
    print(f"\nConvolution terms at τ=0:")
    print(f"  g^R ⊛ f:")
    for i in range(2):
        for j in range(2):
            val = gr_conv_f_data[i, j, n_center]
            print(f"    [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    print(f"\n  f ⊛ g^A:")
    for i in range(2):
        for j in range(2):
            val = f_conv_ga_data[i, j, n_center]
            print(f"    [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    # Check g^R and g^A at τ=0
    print(f"\ng^R(τ=0) Nambu components:")
    for i in range(2):
        for j in range(2):
            val = gr_tau.data[i, j, n_center]
            print(f"  [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    print(f"\ng^A(τ=0) Nambu components:")
    for i in range(2):
        for j in range(2):
            val = ga_tau.data[i, j, n_center]
            print(f"  [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    # Extract Pauli components
    print(f"\nPauli decomposition at τ=0:")
    print(f"  Equilibrium g^K:")
    for pauli_idx, pauli_name in enumerate(['τ₀', 'τ₁', 'τ₂', 'τ₃']):
        val = gk_tau.trace(pauli_idx)[n_center] / 2
        print(f"    {pauli_name}: {val.real:+.6e} {val.imag:+.6e}j")

    print(f"  Convolved g^K:")
    for pauli_idx, pauli_name in enumerate(['τ₀', 'τ₁', 'τ₂', 'τ₃']):
        val = gk_conv.trace(pauli_idx)[n_center] / 2
        print(f"    {pauli_name}: {val.real:+.6e} {val.imag:+.6e}j")

    print("="*70)

    # ========== Test: Frequency-Domain Convolution ==========
    print("\n" + "="*70)
    print("TEST: Frequency-Domain Convolution")
    print("="*70)

    # Get the equilibrium solver that was used to create the initial state
    from equilibrium_class import EquilibriumSolver

    # Create equilibrium solver with same parameters
    eq_grid_params = {
        'time_sampling': grid_parameters['time_sampling'],
        'time_duration': grid_parameters['time_duration'],
        'energy_cutoff': evolution.energy_cutoff,
        'omega_sampling': len(evolution.omega_grid),
        'omega_grid': evolution.omega_grid,
        'eta': system_parameters['eta']
    }

    eq_solver = EquilibriumSolver(eq_grid_params, system_parameters)

    # Compute equilibrium in frequency domain
    gr_omega, gk_omega_eq = eq_solver.compute_equilibrium_gr(
        temperature=system_parameters['temperature'],
        compute_gk=True
    )

    print(f"\nComputing g^K in frequency domain using FDT...")

    # Compute g^A from g^R in frequency domain
    from nambu_class import NambuTensor
    import jax.numpy as jnp

    # g^A(ω) = [g^R(ω)]^† using the involution
    ga_omega = eq_solver._compute_advanced(gr_omega)

    # Get f(ω) = tanh(ω/2T)
    omega_grid_eq = eq_solver.usadel_solver.w_arr
    T = system_parameters['temperature']
    f_omega_array = jnp.tanh(0.5 * omega_grid_eq / T)

    # Create NambuTensor for f(ω)
    f_omega = NambuTensor(f_omega_array, pauli_channel=0)

    # Frequency-domain FDT: g^K(ω) = g^R(ω) @ f(ω) - f(ω) @ g^A(ω)
    # This is matrix multiplication in Nambu space at each ω
    gk_omega_conv = gr_omega @ f_omega - f_omega @ ga_omega

    print(f"  g^K(ω) computed via frequency-domain FDT")

    # Transform to time domain
    gk_tau_freq = eq_solver.omega_to_one_time(gk_omega_conv, g_type='k')

    print(f"  Transformed to time domain")
    print(f"  Shape: {gk_tau_freq.data.shape}")

    # Extract τ=0 point
    n_center_freq = len(omega_grid_eq) // 2

    print(f"\ng^K(τ=0) from frequency-domain convolution:")
    for i in range(2):
        for j in range(2):
            val = gk_tau_freq.data[i, j, n_center_freq]
            print(f"  [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    print(f"\nPauli decomposition (frequency-domain):")
    for pauli_idx, pauli_name in enumerate(['τ₀', 'τ₁', 'τ₂', 'τ₃']):
        val = gk_tau_freq.trace(pauli_idx)[n_center_freq] / 2
        print(f"  {pauli_name}: {val.real:+.6e} {val.imag:+.6e}j")

    print(f"\nComparison at τ=0:")
    print(f"  Equilibrium direct:        τ₃ = {gk_tau.trace(3)[n_center]/2:.6e}")
    print(f"  Time-domain convolution:   τ₃ = {gk_conv.trace(3)[n_center]/2:.6e}")
    print(f"  Freq-domain convolution:   τ₃ = {gk_tau_freq.trace(3)[n_center_freq]/2:.6e}")

    print("="*70)

    # ========== Check f(τ) values at boundaries ==========
    print("\n" + "="*70)
    print("Checking f(τ) values at boundaries vs. center")
    print("="*70)

    print(f"\nDirect f_tau values (before ifftshift):")
    print(f"  f[0] (τ={tau_grid_fft[0]:.3f}): {f_scalar[0]:.6e}")
    print(f"  f[1] (τ={tau_grid_fft[1]:.3f}): {f_scalar[1]:.6e}")
    print(f"  f[N//2] (τ={tau_grid_fft[N//2]:.3f}): {f_scalar[N//2]:.6e}")
    print(f"  f[N-2] (τ={tau_grid_fft[N-2]:.3f}): {f_scalar[N-2]:.6e}")
    print(f"  f[N-1] (τ={tau_grid_fft[N-1]:.3f}): {f_scalar[N-1]:.6e}")

    print(f"\nAfter ifftshift for convolution:")
    f_unshifted = np.fft.ifftshift(f_scalar)
    print(f"  f_unshifted[0]: {f_unshifted[0]:.6e}")
    print(f"  f_unshifted[1]: {f_unshifted[1]:.6e}")
    print(f"  f_unshifted[N//2]: {f_unshifted[N//2]:.6e}")
    print(f"  f_unshifted[N-2]: {f_unshifted[N-2]:.6e}")
    print(f"  f_unshifted[N-1]: {f_unshifted[N-1]:.6e}")

    print("="*70)

    # ========== Plot Convolution vs g^K ==========
    if True:
        print("\nGenerating convolution comparison plots...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Use only the first N points for tau_axis
        tau_axis_conv = tau_axis[:N]

        # Plot 1: τ_2 component (using imaginary part)
        ax = axes[0]
        ax.plot(tau_axis_conv, gk_tau2_imag_eq[:N], linewidth=2, color='black',
                label='g^K(τ) from equilibrium', alpha=0.9)
        ax.plot(tau_axis_conv, gk_conv_tau2_imag, linewidth=2, color='blue',
                label='g^R ⊛ f - f ⊛ g^A', alpha=0.7, linestyle='--')
        ax.set_xlabel('τ (relative time)', fontsize=12)
        ax.set_ylabel('Im[g^K(τ)]  (τ_2 component)', fontsize=12)
        ax.set_title('FDT Verification: τ_2 Component', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: τ_3 component (using imaginary part)
        ax = axes[1]
        ax.plot(tau_axis_conv, gk_tau3_imag_eq[:N], linewidth=2, color='black',
                label='g^K(τ) from equilibrium', alpha=0.9)
        ax.plot(tau_axis_conv, gk_conv_tau3_imag, linewidth=2, color='red',
                label='g^R ⊛ f - f ⊛ g^A', alpha=0.7, linestyle='--')
        ax.set_xlabel('τ (relative time)', fontsize=12)
        ax.set_ylabel('Im[g^K(τ)]  (τ_3 component)', fontsize=12)
        ax.set_title('FDT Verification: τ_3 Component', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print()
    print("All functions extracted successfully!")
    print("="*70)

    # Store in dictionary for easy access
    results = {
        'gr_tau': gr_tau,
        'ga_tau': ga_tau,
        'gk_tau': gk_tau,
        'f_tau': f_tau,
        'delta_t': evolution.delta_t,
        'tau_axis': tau_axis
    }

    return results


if __name__ == "__main__":
    results = main()
