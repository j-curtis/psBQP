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
        'time_sampling': 1501,       # Number of time points (odd ensures tau=0 at integer index)
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

    # ========== Compute Equilibrium in Frequency Domain ==========
    print("Computing equilibrium Green's functions in frequency domain...")

    from equilibrium_class import EquilibriumSolver
    from nambu_class import NambuTensor
    import jax.numpy as jnp

    # Create equilibrium solver
    eq_grid_params = {
        'time_sampling': grid_parameters['time_sampling'],
        'time_duration': grid_parameters['time_duration'],
        'energy_cutoff': evolution.energy_cutoff,
        'omega_sampling': len(evolution.omega_grid),
        'omega_grid': evolution.omega_grid,
        'eta': system_parameters['eta']
    }

    eq_solver = EquilibriumSolver(eq_grid_params, system_parameters)

    # Compute equilibrium g^R(ω) and g^K(ω)
    gr_omega, gk_omega = eq_solver.compute_equilibrium_gr(
        temperature=system_parameters['temperature'],
        compute_gk=True
    )

    print(f"  g^R(ω) shape: {gr_omega.data.shape}")
    print(f"  g^K(ω) shape: {gk_omega.data.shape}")

    # ========== COMMENTED OUT: Initial diagnostic plots ==========
    # # ========== Plot g^K(ω) Components ==========
    # print("\nPlotting g^K(ω) Pauli components to examine asymptotics...")

    # omega_axis = eq_solver.usadel_solver.w_arr

    # # Extract all Pauli components of g^K(ω)
    # gk_omega_pauli = []
    # pauli_names = ['τ₀ (Identity)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']

    # for pauli_idx in range(4):
    #     component = np.array(gk_omega._trace(pauli_idx)) / 2
    #     gk_omega_pauli.append(component)

    # # Create plot with 4 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # axes = axes.flatten()

    # for idx, (component, name) in enumerate(zip(gk_omega_pauli, pauli_names)):
    #     ax = axes[idx]

    #     # For τ₂ component (idx=2), multiply by |ω| to check 1/ω decay
    #     if idx == 2:
    #         # Multiply by |ω| to check if g^K ~ 1/ω at large ω
    #         plot_data = component * np.abs(omega_axis)
    #         ax.plot(omega_axis, np.real(plot_data), linewidth=2, color='blue',
    #                 label='Re[ω·g^K(ω)]', alpha=0.8)
    #         ax.plot(omega_axis, np.imag(plot_data), linewidth=2, color='red',
    #                 label='Im[ω·g^K(ω)]', alpha=0.8)
    #         ax.set_ylabel(f'ω · g^K(ω) - {name}', fontsize=12)
    #         ax.set_title(f'Keldysh GF (×|ω|): {name}', fontsize=13, fontweight='bold')
    #     else:
    #         ax.plot(omega_axis, np.real(component), linewidth=2, color='blue',
    #                 label='Re[g^K(ω)]', alpha=0.8)
    #         ax.plot(omega_axis, np.imag(component), linewidth=2, color='red',
    #                 label='Im[g^K(ω)]', alpha=0.8)
    #         ax.set_ylabel(f'g^K(ω) - {name}', fontsize=12)
    #         ax.set_title(f'Keldysh Green\'s Function: {name}', fontsize=13, fontweight='bold')

    #     ax.set_xlabel('ω (energy)', fontsize=12)
    #     ax.legend(fontsize=10)
    #     ax.grid(True, alpha=0.3)

    #     # Add vertical lines at ±gap for reference (optional)
    #     if hasattr(eq_solver, 'gap_0') and eq_solver.gap_0 is not None:
    #         ax.axvline(eq_solver.gap_0, color='gray', linestyle='--', alpha=0.5, label=f'Δ₀={eq_solver.gap_0:.3f}')
    #         ax.axvline(-eq_solver.gap_0, color='gray', linestyle='--', alpha=0.5)

    # plt.suptitle('Equilibrium g^K(ω) - All Pauli Components', fontsize=15, fontweight='bold', y=0.995)
    # plt.tight_layout()
    # plt.show()
    # print()

    # # ========== Plot g^K(ω) with Subtracted Asymptotics ==========
    # print("Plotting g^K(ω) with subtracted asymptotics (regularized)...")

    # # Subtract asymptotics for each component
    # gk_omega_regularized = []
    # T = system_parameters['temperature']
    # gap = eq_solver.gap_0 if hasattr(eq_solver, 'gap_0') and eq_solver.gap_0 is not None else 0.0

    # for pauli_idx in range(4):
    #     component = gk_omega_pauli[pauli_idx].copy()

    #     if pauli_idx == 3:  # τ₃: subtract C·tanh(ω/2T)
    #         # Fit C from the tail to match actual asymptotic behavior
    #         tail_indices = omega_axis > 0.8 * np.max(omega_axis)
    #         tanh_tail = np.tanh(omega_axis[tail_indices] / (2.0 * T))
    #         g_tail = component[tail_indices]
    #         C_tanh = np.mean(np.real(g_tail / tanh_tail))  # Fit coefficient
    #         print(f"  Fitted C_tanh = {C_tanh:.6f} (theoretical = 2.0)")
    #         component = component - C_tanh * np.tanh(omega_axis / (2.0 * T))

    #     elif pauli_idx == 2:  # τ₂: subtract C·ω₀/√(ω² + ω₀²)
    #         # Match the regularization in equilibrium_class.py (corrected version)
    #         if gap > 0:
    #             omega_0 = min(2 * gap, np.abs(omega_axis[int(0.05 * len(omega_axis))]))
    #             C_bessel = -2j * gap / omega_0  # Corrected: C = -2iΔ/ω₀
    #             component = component - (C_bessel * omega_0) / np.sqrt(omega_axis**2 + omega_0**2)
    #         else:
    #             pass  # No regularization if gap not set

    #     gk_omega_regularized.append(component)

    # # Create plot with 4 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # axes = axes.flatten()

    # for idx, (component, name) in enumerate(zip(gk_omega_regularized, pauli_names)):
    #     ax = axes[idx]
    #     ax.plot(omega_axis, np.real(component), linewidth=2, color='blue',
    #             label='Re[g^K_reg(ω)]', alpha=0.8)
    #     ax.plot(omega_axis, np.imag(component), linewidth=2, color='red',
    #             label='Im[g^K_reg(ω)]', alpha=0.8)
    #     ax.set_xlabel('ω (energy)', fontsize=12)
    #     ax.set_ylabel(f'g^K_reg(ω) - {name}', fontsize=12)
    #     ax.set_title(f'Regularized Keldysh GF: {name}', fontsize=13, fontweight='bold')
    #     ax.legend(fontsize=10)
    #     ax.grid(True, alpha=0.3)

    #     # Add vertical lines at ±gap for reference
    #     if gap > 0:
    #         ax.axvline(gap, color='gray', linestyle='--', alpha=0.5)
    #         ax.axvline(-gap, color='gray', linestyle='--', alpha=0.5)

    # plt.suptitle('Regularized g^K(ω) - Asymptotics Subtracted', fontsize=15, fontweight='bold', y=0.995)
    # plt.tight_layout()
    # plt.show()
    # print()

    # # ========== Plot g^R(ω) Components ==========
    # print("Plotting g^R(ω) Pauli components...")

    # # Extract all Pauli components of g^R(ω)
    # gr_omega_pauli = []
    # for pauli_idx in range(4):
    #     component = np.array(gr_omega._trace(pauli_idx)) / 2
    #     gr_omega_pauli.append(component)

    # # Create plot with 4 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # axes = axes.flatten()

    # for idx, (component, name) in enumerate(zip(gr_omega_pauli, pauli_names)):
    #     ax = axes[idx]
    #     ax.plot(omega_axis, np.real(component), linewidth=2, color='blue',
    #             label='Re[g^R(ω)]', alpha=0.8)
    #     ax.plot(omega_axis, np.imag(component), linewidth=2, color='red',
    #             label='Im[g^R(ω)]', alpha=0.8)
    #     ax.set_xlabel('ω (energy)', fontsize=12)
    #     ax.set_ylabel(f'g^R(ω) - {name}', fontsize=12)
    #     ax.set_title(f'Retarded Green\'s Function: {name}', fontsize=13, fontweight='bold')
    #     ax.legend(fontsize=10)
    #     ax.grid(True, alpha=0.3)

    #     # Add vertical lines at ±gap for reference
    #     if gap > 0:
    #         ax.axvline(gap, color='gray', linestyle='--', alpha=0.5)
    #         ax.axvline(-gap, color='gray', linestyle='--', alpha=0.5)

    # plt.suptitle('Equilibrium g^R(ω) - All Pauli Components', fontsize=15, fontweight='bold', y=0.995)
    # plt.tight_layout()
    # plt.show()
    # print()

    # # ========== Plot g^R(ω) with Subtracted Asymptotics ==========
    # print("Plotting g^R(ω) with subtracted asymptotics (regularized)...")

    # # Subtract asymptotics for g^R
    # # Match what equilibrium_class.py actually does
    # gr_omega_regularized = []
    # for pauli_idx in range(4):
    #     component = gr_omega_pauli[pauli_idx].copy()

    #     if pauli_idx == 3:  # τ₃: subtract constant
    #         # g^R_3(ω) → C (constant) at large |ω|
    #         C_constant = component[-1]  # Use last value as constant
    #         component = component - C_constant

    #     elif pauli_idx == 2:  # τ₂: subtract Lorentzian regularization
    #         # g^R_2(ω) ~ C/ω at large |ω|
    #         # Match the regularization in equilibrium_class.py
    #         if gap > 0:
    #             # Use same logic as equilibrium_class.py
    #             omega_10percent = np.max(omega_axis) * 1e-2
    #             omega_0 = np.abs(omega_10percent) / 2.0
    #             C_decay = -1j * gap
    #             C_prime = -1j * gap * (-1j * 0.2)  # eta = 0.2

    #             regularization = (C_decay * omega_axis / (omega_axis**2 + omega_0**2) +
    #                             C_prime / (omega_axis**2 + omega_0**2))
    #             component = component - regularization

    #     gr_omega_regularized.append(component)

    # # Create plot with 4 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # axes = axes.flatten()

    # for idx, (component, name) in enumerate(zip(gr_omega_regularized, pauli_names)):
    #     ax = axes[idx]
    #     ax.plot(omega_axis, np.real(component), linewidth=2, color='blue',
    #             label='Re[g^R_reg(ω)]', alpha=0.8)
    #     ax.plot(omega_axis, np.imag(component), linewidth=2, color='red',
    #             label='Im[g^R_reg(ω)]', alpha=0.8)
    #     ax.set_xlabel('ω (energy)', fontsize=12)
    #     ax.set_ylabel(f'g^R_reg(ω) - {name}', fontsize=12)
    #     ax.set_title(f'Regularized Retarded GF: {name}', fontsize=13, fontweight='bold')
    #     ax.legend(fontsize=10)
    #     ax.grid(True, alpha=0.3)

    #     # Add vertical lines at ±gap for reference
    #     if gap > 0:
    #         ax.axvline(gap, color='gray', linestyle='--', alpha=0.5)
    #         ax.axvline(-gap, color='gray', linestyle='--', alpha=0.5)

    # plt.suptitle('Regularized g^R(ω) - Asymptotics Subtracted', fontsize=15, fontweight='bold', y=0.995)
    # plt.tight_layout()
    # plt.show()
    # print()

    # ========== Generate Equilibrium State ==========
    print("Computing equilibrium state in time domain...")
    initial_state, gr_tau, gk_tau = evolution.generate_initial_state()

    print(f"  g^R(τ) shape: {gr_tau.data.shape}")
    print(f"  g^K(τ) shape: {gk_tau.data.shape}")
    print()

    # # ========== Plot g^K(τ) Components ==========
    # print("Plotting g^K(τ) Pauli components in time domain...")

    # # Get omega grid and compute tau grid
    # omega_grid = evolution.omega_grid
    # d_omega = omega_grid[1] - omega_grid[0]
    # n_omega = len(omega_grid)
    # tau_grid_fft = np.linspace(-np.pi/d_omega, np.pi/d_omega, n_omega)

    # # Extract all Pauli components of g^K(τ)
    # gk_tau_pauli = []
    # pauli_names = ['τ₀ (Identity)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']
    # for pauli_idx in range(4):
    #     component = gk_tau.trace(pauli_idx) / 2
    #     gk_tau_pauli.append(component)

    # # Create plot with 4 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # axes = axes.flatten()

    # for idx, (component, name) in enumerate(zip(gk_tau_pauli, pauli_names)):
    #     ax = axes[idx]
    #     ax.plot(tau_grid_fft, np.real(component), linewidth=2, color='blue',
    #             label='Re[g^K(τ)]', alpha=0.8)
    #     ax.plot(tau_grid_fft, np.imag(component), linewidth=2, color='red',
    #             label='Im[g^K(τ)]', alpha=0.8)
    #     ax.set_xlabel('τ (relative time)', fontsize=12)
    #     ax.set_ylabel(f'g^K(τ) - {name}', fontsize=12)
    #     ax.set_title(f'Keldysh GF in Time Domain: {name}', fontsize=13, fontweight='bold')
    #     ax.legend(fontsize=10)
    #     ax.grid(True, alpha=0.3)
    #     ax.axvline(0, color='gray', linestyle='--', alpha=0.3)  # Mark τ=0

    # plt.suptitle('Equilibrium g^K(τ) - All Pauli Components', fontsize=15, fontweight='bold', y=0.995)
    # plt.tight_layout()
    # plt.show()
    # print()

    # # ========== Plot g^R(τ) Components ==========
    # print("Plotting g^R(τ) Pauli components in time domain...")

    # # Extract all Pauli components of g^R(τ)
    # gr_tau_pauli = []
    # for pauli_idx in range(4):
    #     component = gr_tau.trace(pauli_idx) / 2
    #     gr_tau_pauli.append(component)

    # # Create plot with 4 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # axes = axes.flatten()

    # for idx, (component, name) in enumerate(zip(gr_tau_pauli, pauli_names)):
    #     ax = axes[idx]
    #     ax.plot(tau_grid_fft, np.real(component), linewidth=2, color='blue',
    #             label='Re[g^R(τ)]', alpha=0.8)
    #     ax.plot(tau_grid_fft, np.imag(component), linewidth=2, color='red',
    #             label='Im[g^R(τ)]', alpha=0.8)
    #     ax.set_xlabel('τ (relative time)', fontsize=12)
    #     ax.set_ylabel(f'g^R(τ) - {name}', fontsize=12)
    #     ax.set_title(f'Retarded GF in Time Domain: {name}', fontsize=13, fontweight='bold')
    #     ax.legend(fontsize=10)
    #     ax.grid(True, alpha=0.3)
    #     ax.axvline(0, color='gray', linestyle='--', alpha=0.3)  # Mark τ=0

    # plt.suptitle('Equilibrium g^R(τ) - All Pauli Components', fontsize=15, fontweight='bold', y=0.995)
    # plt.tight_layout()
    # plt.show()
    # print()

    # Define tau_grid_fft for later use
    omega_grid = evolution.omega_grid
    d_omega = omega_grid[1] - omega_grid[0]
    n_omega = len(omega_grid)
    tau_grid_fft = np.linspace(-np.pi/d_omega, np.pi/d_omega, n_omega)

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

    # ========== Plot: Analytical vs Numerical f(τ) Comparison ==========
    print("Generating f(τ) comparison plot...")

    # Use the FFT tau grid for plotting
    tau_axis = tau_grid_fft

    # Get analytical f(τ) = -iT/sinh(πτT) (already computed)
    f_analytical = f_tau.trace(0) / 2  # τ_0 component

    # Get numerical f(τ) from Fourier transform of f(ω) = tanh(ω/2T)
    from equilibrium_class import EquilibriumSolver

    eq_grid_params = {
        'time_sampling': grid_parameters['time_sampling'],
        'time_duration': grid_parameters['time_duration'],
        'energy_cutoff': evolution.energy_cutoff,
        'omega_sampling': len(evolution.omega_grid),
        'omega_grid': evolution.omega_grid,
        'eta': system_parameters['eta']
    }


    # ========== Helper Function for Convolution ==========
    def compute_convolution(gr_data, ga_data, f_scalar, N, d_tau, label=""):
        """Compute g^K = g^R ⊛ f - f ⊛ g^A using circular convolution."""
        # Compute g^R ⊛ f
        gr_conv_f_data = np.zeros((2, 2, N), dtype=complex)
        for i in range(2):
            for j in range(2):
                x = gr_data[i, j, :N]
                h = f_scalar[:N]
                result = np.zeros(N, dtype=complex)
                for n in range(N):
                    conv_sum = 0.0
                    for m in range(N):
                        conv_sum += x[m] * h[(n - m) % N]
                    result[n] = conv_sum * d_tau
                gr_conv_f_data[i, j, :] = result

        # Compute f ⊛ g^A
        f_conv_ga_data = np.zeros((2, 2, N), dtype=complex)
        for i in range(2):
            for j in range(2):
                x = f_scalar[:N]
                h = ga_data[i, j, :N]
                result = np.zeros(N, dtype=complex)
                for n in range(N):
                    conv_sum = 0.0
                    for m in range(N):
                        conv_sum += x[m] * h[(n - m) % N]
                    result[n] = conv_sum * d_tau
                f_conv_ga_data[i, j, :] = result

        # Compute g^K = g^R ⊛ f - f ⊛ g^A
        gk_conv_data = gr_conv_f_data - f_conv_ga_data
        return NambuKeldyshTensor(gk_conv_data)

    # ========== Discrete Circular Convolution ==========
    print("\nComputing discrete circular convolutions...")

    n_g = gr_tau.data.shape[-1]
    n_f_analytical = f_tau.data.shape[-1]

    print(f"  Input shapes:")
    print(f"    g^R, g^A: {gr_tau.data.shape}")
    print(f"    f_analytical: {f_tau.data.shape}")

    # Check sizes and use minimum
    N = min(n_g, n_f_analytical)
    print(f"  Using N = {N} points for convolution")

    # Use the FFT tau spacing for convolution normalization
    d_tau = tau_grid_fft[1] - tau_grid_fft[0]
    print(f"  Using d_tau = {d_tau:.6f} for convolution normalization")

    # Extract scalar thermal distribution
    f_analytical_scalar = f_tau.data[0, 0, :]

    # Compute convolution
    print("  Computing convolution (g^R ⊛ f - f ⊛ g^A)...")
    gk_conv_analytical = compute_convolution(gr_tau.data, ga_tau.data,
                                             f_analytical_scalar, N, d_tau, "analytical")

    print("  Computed convolution")

    # For diagnostics, compute detailed breakdown with analytical f only
    gr_conv_f_data = np.zeros((2, 2, N), dtype=complex)

    # For diagnostic: track contributions to center point
    diagnostic_contributions_gr_f = []

    for i in range(2):
        for j in range(2):
            # Use arrays in their current fftshifted (centered) state
            # NO ifftshift/fftshift - compute convolution directly!
            x = gr_tau.data[i, j, :N]
            h = f_analytical_scalar[:N]

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
            x = f_analytical_scalar[:N]
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
        # Arrays are already fftshifted: tau=0 is at index N//2
        # Index m directly corresponds to tau_grid_fft[m]
        tau_val = tau_grid_fft[m]
        print(f"    #{idx+1}: m={m} (τ≈{tau_val:.3f}): g^R={x_val:.3e} × f={h_val:.3e} = {term:.3e}")

    # Find largest contributions to f ⊛ g^A
    contrib_f_ga_sorted = sorted(diagnostic_contributions_f_ga, key=lambda x: abs(x[3]), reverse=True)
    print(f"\n  Top 10 contributions to (f ⊛ g^A)[τ=0]:")
    for idx, (m, x_val, h_val, term) in enumerate(contrib_f_ga_sorted[:10]):
        # Arrays are already fftshifted: tau=0 is at index N//2
        # Index m directly corresponds to tau_grid_fft[m]
        tau_val = tau_grid_fft[m]
        print(f"    #{idx+1}: m={m} (τ≈{tau_val:.3f}): f={x_val:.3e} × g^A={h_val:.3e} = {term:.3e}")

    # Compute g^K = g^R ⊛ f - f ⊛ g^A (detailed diagnostics)
    gk_conv_data = gr_conv_f_data - f_conv_ga_data

    print(f"\n  Convolution complete.")

    # Extract traces for convolution
    gk_conv_analytical_tau2 = np.imag(gk_conv_analytical.trace(2)) / 2
    gk_conv_analytical_tau3 = np.imag(gk_conv_analytical.trace(3)) / 2

    # Apply time shift: roll by N//2 + 1 to align with tau grid properly
    gk_conv_analytical_tau2 = np.roll(gk_conv_analytical_tau2, N//2 + 1)
    gk_conv_analytical_tau3 = np.roll(gk_conv_analytical_tau3, N//2 + 1)

    print(f"  Convolution tau2 max: {np.max(np.abs(gk_conv_analytical_tau2)):.6e}")
    print(f"  Convolution tau3 max: {np.max(np.abs(gk_conv_analytical_tau3)):.6e}")

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

    # Check all Nambu components at τ=0 for analytical convolved g^K
    print(f"\ng^K(τ=0) Nambu components (analytical convolution):")
    for i in range(2):
        for j in range(2):
            val = gk_conv_analytical.data[i, j, n_center]
            print(f"  [{i},{j}]: {val.real:+.6e} {val.imag:+.6e}j")

    # Compute and display the differences
    print(f"\nDifference (convolution - equilibrium) at τ=0:")
    for i in range(2):
        for j in range(2):
            diff = gk_conv_analytical.data[i, j, n_center] - gk_tau.data[i, j, n_center]
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
        val = gk_conv_analytical.trace(pauli_idx)[n_center] / 2
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
    print(f"  Equilibrium direct:           τ₃ = {gk_tau.trace(3)[n_center]/2:.6e}")
    print(f"  Time-domain convolution:      τ₃ = {gk_conv_analytical.trace(3)[n_center]/2:.6e}")
    print(f"  Freq-domain convolution:      τ₃ = {gk_tau_freq.trace(3)[n_center_freq]/2:.6e}")

    print("="*70)

    # ========== Check f(τ) values at boundaries ==========
    print("\n" + "="*70)
    print("Checking f(τ) values at boundaries vs. center")
    print("="*70)

    print(f"\nDirect f_analytical values (before ifftshift):")
    print(f"  f[0] (τ={tau_grid_fft[0]:.3f}): {f_analytical_scalar[0]:.6e}")
    print(f"  f[1] (τ={tau_grid_fft[1]:.3f}): {f_analytical_scalar[1]:.6e}")
    print(f"  f[N//2] (τ={tau_grid_fft[N//2]:.3f}): {f_analytical_scalar[N//2]:.6e}")
    print(f"  f[N-2] (τ={tau_grid_fft[N-2]:.3f}): {f_analytical_scalar[N-2]:.6e}")
    print(f"  f[N-1] (τ={tau_grid_fft[N-1]:.3f}): {f_analytical_scalar[N-1]:.6e}")

    print(f"\nAfter ifftshift for convolution:")
    f_unshifted = np.fft.ifftshift(f_analytical_scalar)
    print(f"  f_unshifted[0]: {f_unshifted[0]:.6e}")
    print(f"  f_unshifted[1]: {f_unshifted[1]:.6e}")
    print(f"  f_unshifted[N//2]: {f_unshifted[N//2]:.6e}")
    print(f"  f_unshifted[N-2]: {f_unshifted[N-2]:.6e}")
    print(f"  f_unshifted[N-1]: {f_unshifted[N-1]:.6e}")

    print("="*70)

    # ========== Two-Time Formalism Convolution ==========
    print("\n" + "="*70)
    print("Two-Time Formalism: g^K(t, t') vs g^R @ f - f @ g^A")
    print("="*70)

    # Get two-time Green's functions from initial_state
    # These were computed by fourier_transform_to_two_time() in generate_initial_state()
    gr_two_time = initial_state.gr
    gk_two_time = initial_state.gk

    # Compute g^A(t, t') from g^R(t, t') using involution: g^A = -τ₃ (g^R)^† τ₃
    ga_two_time = -gr_two_time.involution()

    # f(t, t') is already available from thermal_dist
    f_two_time = evolution.thermal_dist

    print(f"\nTwo-time Green's functions shapes:")
    print(f"  g^R(t, t'): {gr_two_time.data.shape}")
    print(f"  g^A(t, t'): {ga_two_time.data.shape}")
    print(f"  g^K(t, t'): {gk_two_time.data.shape}")
    print(f"  f(t, t'): {f_two_time.data.shape}")

    # Get actual time grid size from two-time functions
    n_t_two_time = gr_two_time.data.shape[-1]

    # Perform convolution ONLY for last row: g^R[-1,:] @ f - f[-1,:] @ g^A
    # This is much faster than computing the full convolution
    print(f"\nComputing two-time convolution for last row only:")
    print(f"  dt * (g^R[-1,:] @ f - f[-1,:] @ g^A)")

    # Extract last rows as NambuKeldyshTensor objects
    from nambu_keldysh_class import NambuKeldyshTensor

    # Create single-row tensors (keeping time dimension)
    gr_last_row = NambuKeldyshTensor(gr_two_time.data[:, :, -1:, :])  # Shape: (2, 2, 1, n_t)
    ga_last_row = ga_two_time  # Full ga needed for convolution
    f_last_row = NambuKeldyshTensor(f_two_time.data[:, :, -1:, :])  # Shape: (2, 2, 1, n_t)

    # Compute: dt * (gr[-1,:] @ f - f[-1,:] @ ga)
    # The @ operator performs matrix convolution, must multiply by dt
    dt = evolution.delta_t
    print(f"  Using dt = {dt:.6f}")
    gk_conv_last_row_raw = gr_last_row @ f_two_time - f_last_row @ ga_two_time

    # Multiply by dt to get proper convolution normalization
    gk_conv_last_row = NambuKeldyshTensor(gk_conv_last_row_raw.data * dt)

    print(f"  Result shape: {gk_conv_last_row.data.shape}")

    # Compare last row: g^K[-1, :] vs (g^R[-1,:] @ f - f[-1,:] @ g^A)
    print(f"\nComparing last time slice (t = t_max):")

    # Extract last row for each component
    t_last = -1

    print(f"\n  g^K[{t_last}, :] vs convolution:")
    print(f"  Nambu components at a few time points:")

    for i in range(2):
        for j in range(2):
            print(f"\n    Component [{i},{j}]:")
            for tp_idx in [0, n_t_two_time//4, n_t_two_time//2, 3*n_t_two_time//4, -1]:
                gk_val = gk_two_time.data[i, j, t_last, tp_idx]
                conv_val = gk_conv_last_row.data[i, j, 0, tp_idx]  # Shape is (2,2,1,n_t), so use index 0 for the single row
                diff = gk_val - conv_val
                print(f"      t'[{tp_idx}]: g^K = {gk_val:.6e}, conv = {conv_val:.6e}, diff = {diff:.6e}")

    # Extract Pauli components for last row
    print(f"\n  Pauli components at t = t_max:")
    pauli_names = ['τ₀', 'τ₁', 'τ₂', 'τ₃']

    for pauli_idx, pauli_name in enumerate(pauli_names):
        gk_last_row = gk_two_time.trace(pauli_idx)[t_last, :] / 2
        conv_last_row_traced = gk_conv_last_row.trace(pauli_idx)[0, :] / 2  # Shape is (1, n_t), use index 0

        max_diff = np.max(np.abs(gk_last_row - conv_last_row_traced))
        max_gk = np.max(np.abs(gk_last_row))
        rel_error = max_diff / max_gk if max_gk > 0 else 0

        print(f"    {pauli_name}: max|g^K - conv| = {max_diff:.6e}, max|g^K| = {max_gk:.6e}, rel_error = {rel_error:.6e}")

    # Plot comparison for last row
    print(f"\n  Plotting last row comparison...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot τ_2 component
    ax = axes[0]
    gk_tau2_last = np.imag(gk_two_time.trace(2)[t_last, :]) / 2
    conv_tau2_last = np.imag(gk_conv_last_row.trace(2)[0, :]) / 2  # Index 0 for single row

    t_axis_two_time = np.arange(n_t_two_time)
    ax.plot(t_axis_two_time, gk_tau2_last, linewidth=2, color='black',
            label='g^K(t_max, t\')', alpha=0.9)
    ax.plot(t_axis_two_time, conv_tau2_last, linewidth=2, color='blue',
            label='(g^R[-1,:] @ f - f[-1,:] @ g^A)', alpha=0.7, linestyle='--')
    ax.set_xlabel('t\' (time index)', fontsize=12)
    ax.set_ylabel('Im[g^K(t_max, t\')]  (τ_2)', fontsize=12)
    ax.set_title('Two-Time FDT: τ_2 Component (Last Row)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot τ_3 component
    ax = axes[1]
    gk_tau3_last = np.imag(gk_two_time.trace(3)[t_last, :]) / 2
    conv_tau3_last = np.imag(gk_conv_last_row.trace(3)[0, :]) / 2  # Index 0 for single row

    ax.plot(t_axis_two_time, gk_tau3_last, linewidth=2, color='black',
            label='g^K(t_max, t\')', alpha=0.9)
    ax.plot(t_axis_two_time, conv_tau3_last, linewidth=2, color='red',
            label='(g^R[-1,:] @ f - f[-1,:] @ g^A)', alpha=0.7, linestyle='--')
    ax.set_xlabel('t\' (time index)', fontsize=12)
    ax.set_ylabel('Im[g^K(t_max, t\')]  (τ_3)', fontsize=12)
    ax.set_title('Two-Time FDT: τ_3 Component (Last Row)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("="*70)

    # ========== Comprehensive Comparison Plot: All g^K Functions ==========
    print("\n" + "="*70)
    print("Comprehensive Comparison: g^K(τ), g^K[-1,:], and Convolutions")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Construct proper time axes for t ≤ 0
    # Two-time grid: use the actual time grid from evolution object
    dt = evolution.delta_t
    time_grid_two_time = evolution.time_grid  # Already goes from -T_max to 0

    # One-time tau grid: centered at 0, extract only negative part (t ≤ 0)
    # For equilibrium g^K(t,t') = g^K(t-t'), so tau = t - t'
    # When t' = 0 (last column), tau = t, so we want tau ≤ 0
    center_idx = len(tau_grid_fft) // 2
    tau_negative_mask = tau_grid_fft <= 0
    tau_axis_negative = tau_grid_fft[tau_negative_mask]

    # Prepare data for all four functions (only t ≤ 0)
    # 1. g^K(τ) from equilibrium (one-time) - extract τ ≤ 0
    gk_tau_tau2_neg = np.imag(gk_tau.trace(2)[tau_negative_mask]) / 2
    gk_tau_tau3_neg = np.imag(gk_tau.trace(3)[tau_negative_mask]) / 2

    # 2. g^K[-1,:] from two-time (already for t ≤ 0)
    gk_two_time_tau2 = np.imag(gk_two_time.trace(2)[t_last, :]) / 2
    gk_two_time_tau3 = np.imag(gk_two_time.trace(3)[t_last, :]) / 2

    # 3. One-time convolution (extract τ ≤ 0 part)
    # The convolution was done on the full tau grid and then shifted
    # Extract only the negative tau part
    gk_conv_one_time_tau2_full = gk_conv_analytical_tau2[:N]
    gk_conv_one_time_tau3_full = gk_conv_analytical_tau3[:N]
    tau_axis_conv_full = tau_axis[:N]
    conv_negative_mask = tau_axis_conv_full <= 0
    gk_conv_one_time_tau2_neg = gk_conv_one_time_tau2_full[conv_negative_mask]
    gk_conv_one_time_tau3_neg = gk_conv_one_time_tau3_full[conv_negative_mask]
    tau_axis_conv_neg = tau_axis_conv_full[conv_negative_mask]

    # 4. Two-time convolution last row (already for t ≤ 0)
    gk_conv_two_time_tau2 = np.imag(gk_conv_last_row.trace(2)[0, :]) / 2
    gk_conv_two_time_tau3 = np.imag(gk_conv_last_row.trace(3)[0, :]) / 2

    # Plot τ_2 component
    ax = axes[0]
    ax.plot(tau_axis_negative, gk_tau_tau2_neg, linewidth=2.5, color='black',
            label='g^K(τ) equilibrium (one-time)', alpha=0.9)
    ax.plot(time_grid_two_time, gk_two_time_tau2, linewidth=2, color='green',
            label='g^K[-1,:] (two-time last row)', alpha=0.8, linestyle='-.')
    ax.plot(tau_axis_conv_neg, gk_conv_one_time_tau2_neg, linewidth=2, color='blue',
            label='g^R ⊛ f - f ⊛ g^A (one-time)', alpha=0.7, linestyle='--')
    ax.plot(time_grid_two_time, gk_conv_two_time_tau2, linewidth=1.5, color='red',
            label='dt*(g^R[-1,:] @ f - f[-1,:] @ g^A) (two-time)', alpha=0.7, linestyle=':')
    ax.set_xlabel('t (time, t ≤ 0)', fontsize=12)
    ax.set_ylabel('Im[g^K]  (τ_2 component)', fontsize=12)
    ax.set_title('Comprehensive g^K Comparison: τ_2 Component', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_grid_two_time[0], 0])

    # Plot τ_3 component
    ax = axes[1]
    ax.plot(tau_axis_negative, gk_tau_tau3_neg, linewidth=2.5, color='black',
            label='g^K(τ) equilibrium (one-time)', alpha=0.9)
    ax.plot(time_grid_two_time, gk_two_time_tau3, linewidth=2, color='green',
            label='g^K[-1,:] (two-time last row)', alpha=0.8, linestyle='-.')
    ax.plot(tau_axis_conv_neg, gk_conv_one_time_tau3_neg, linewidth=2, color='blue',
            label='g^R ⊛ f - f ⊛ g^A (one-time)', alpha=0.7, linestyle='--')
    ax.plot(time_grid_two_time, gk_conv_two_time_tau3, linewidth=1.5, color='red',
            label='dt*(g^R[-1,:] @ f - f[-1,:] @ g^A) (two-time)', alpha=0.7, linestyle=':')
    ax.set_xlabel('t (time, t ≤ 0)', fontsize=12)
    ax.set_ylabel('Im[g^K]  (τ_3 component)', fontsize=12)
    ax.set_title('Comprehensive g^K Comparison: τ_3 Component', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_grid_two_time[0], 0])

    plt.tight_layout()
    plt.show()

    # Print diagnostic information about the grids
    print(f"\nGrid information:")
    print(f"  dt = {dt:.6f}")
    print(f"  Two-time grid: {n_t_two_time} points, range [{time_grid_two_time[0]:.3f}, {time_grid_two_time[-1]:.3f}]")
    print(f"  One-time τ ≤ 0: {len(tau_axis_negative)} points, range [{tau_axis_negative[0]:.3f}, {tau_axis_negative[-1]:.3f}]")
    print(f"  One-time conv τ ≤ 0: {len(tau_axis_conv_neg)} points, range [{tau_axis_conv_neg[0]:.3f}, {tau_axis_conv_neg[-1]:.3f}]")

    print(f"\nTime grid alignment check (last 5 points):")
    print(f"  Two-time t: {time_grid_two_time[-5:]}")
    print(f"  One-time τ: {tau_axis_negative[-5:]}")
    print(f"  Conv τ:     {tau_axis_conv_neg[-5:]}")

    print(f"\nValue comparisons at t=0 (last index):")
    print(f"  g^K(τ=0) τ_2: {gk_tau_tau2_neg[-1]:.6e}")
    print(f"  g^K(τ=0) τ_3: {gk_tau_tau3_neg[-1]:.6e}")
    print(f"  g^K[-1,-1] τ_2: {gk_two_time_tau2[-1]:.6e}")
    print(f"  g^K[-1,-1] τ_3: {gk_two_time_tau3[-1]:.6e}")
    print(f"  One-time conv τ_2: {gk_conv_one_time_tau2_neg[-1]:.6e}")
    print(f"  One-time conv τ_3: {gk_conv_one_time_tau3_neg[-1]:.6e}")
    print(f"  Two-time conv τ_2: {gk_conv_two_time_tau2[-1]:.6e}")
    print(f"  Two-time conv τ_3: {gk_conv_two_time_tau3[-1]:.6e}")

    print(f"\nValue comparisons at second-to-last index (-2):")
    print(f"  g^K(τ) τ_2: {gk_tau_tau2_neg[-2]:.6e}")
    print(f"  g^K[-1,-2] τ_2: {gk_two_time_tau2[-2]:.6e}")
    print(f"  Two-time conv τ_2: {gk_conv_two_time_tau2[-2]:.6e}")

    # Check if there's a systematic 1-index offset
    print(f"\nChecking for 1-index offset:")
    print(f"  Does g^K[-1,-1] match g^K(τ)[-2]?")
    print(f"    τ_2: g^K[-1,-1]={gk_two_time_tau2[-1]:.6e}, g^K(τ)[-2]={gk_tau_tau2_neg[-2]:.6e}, diff={abs(gk_two_time_tau2[-1]-gk_tau_tau2_neg[-2]):.6e}")
    print(f"    τ_3: g^K[-1,-1]={gk_two_time_tau3[-1]:.6e}, g^K(τ)[-2]={gk_tau_tau3_neg[-2]:.6e}, diff={abs(gk_two_time_tau3[-1]-gk_tau_tau3_neg[-2]):.6e}")

    print("="*70)

    # ========== Plot Convolution vs g^K (One-Time) ==========
    if True:
        print("\nGenerating one-time convolution comparison plots...")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Use only the first N points for tau_axis
        tau_axis_conv = tau_axis[:N]

        # Extract equilibrium g^K traces
        gk_tau2_imag_eq = np.imag(gk_tau.trace(2)[:N]) / 2
        gk_tau3_imag_eq = np.imag(gk_tau.trace(3)[:N]) / 2

        # Plot 1: τ_2 component
        ax = axes[0]
        ax.plot(tau_axis_conv, gk_tau2_imag_eq, linewidth=2, color='black',
                label='g^K(τ) from equilibrium', alpha=0.9)
        ax.plot(tau_axis_conv, gk_conv_analytical_tau2, linewidth=2, color='blue',
                label='g^R ⊛ f - f ⊛ g^A', alpha=0.7, linestyle='--')
        ax.set_xlabel('τ (relative time)', fontsize=12)
        ax.set_ylabel('Im[g^K(τ)]  (τ_2 component)', fontsize=12)
        ax.set_title('FDT Verification: τ_2 Component', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: τ_3 component
        ax = axes[1]
        ax.plot(tau_axis_conv, gk_tau3_imag_eq, linewidth=2, color='black',
                label='g^K(τ) from equilibrium', alpha=0.9)
        ax.plot(tau_axis_conv, gk_conv_analytical_tau3, linewidth=2, color='red',
                label='g^R ⊛ f - f ⊛ g^A', alpha=0.7, linestyle='--')
        ax.set_xlabel('τ (relative time)', fontsize=12)
        ax.set_ylabel('Im[g^K(τ)]  (τ_3 component)', fontsize=12)
        ax.set_title('FDT Verification: τ_3 Component', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print()
    print("All functions extracted successfully!")
    print("="*70)

    # ========== Load and Analyze Simulated State (if exists) ==========
    print("\n" + "="*70)
    print("Checking for Simulated State")
    print("="*70)

    try:
        import pickle
        with open('simulated_state.pkl', 'rb') as f:
            simulated_state = pickle.load(f)

        print("Simulated state loaded successfully!")
        print(f"  g^R shape: {simulated_state.gr.data.shape}")
        print(f"  g^K shape: {simulated_state.gk.data.shape}")

        # Get two-time Green's functions from simulated state
        gr_sim = simulated_state.gr
        gk_sim = simulated_state.gk

        # Compute g^A from g^R using involution
        ga_sim = -gr_sim.involution()

        # Get time grid
        n_t_sim = gr_sim.data.shape[-1]
        time_grid_sim = evolution.time_grid

        print(f"\nComputing convolution from simulated state...")
        print(f"  dt * (g^R[-1,:] @ f - f[-1,:] @ g^A)")

        # Extract last rows for convolution
        gr_sim_last_row = NambuKeldyshTensor(gr_sim.data[:, :, -1:, :])
        f_last_row = NambuKeldyshTensor(f_two_time.data[:, :, -1:, :])

        # Compute convolution: dt * (gr[-1,:] @ f - f[-1,:] @ ga)
        dt = evolution.delta_t
        gk_sim_conv_raw = gr_sim_last_row @ f_two_time - f_last_row @ ga_sim
        gk_sim_conv = NambuKeldyshTensor(gk_sim_conv_raw.data * dt)

        # Extract traces for plotting
        gk_sim_last_row_tau2 = np.imag(gk_sim.trace(2)[-1, :]) / 2
        gk_sim_last_row_tau3 = np.imag(gk_sim.trace(3)[-1, :]) / 2

        gk_sim_conv_tau2 = np.imag(gk_sim_conv.trace(2)[0, :]) / 2
        gk_sim_conv_tau3 = np.imag(gk_sim_conv.trace(3)[0, :]) / 2

        # Equilibrium for reference
        gk_eq_last_row_tau2 = np.imag(gk_two_time.trace(2)[-1, :]) / 2
        gk_eq_last_row_tau3 = np.imag(gk_two_time.trace(3)[-1, :]) / 2

        print(f"\nPlotting comparison: simulated g^K[-1,:], convolution, and equilibrium...")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot τ_2 component
        ax = axes[0]
        ax.plot(time_grid_sim, gk_sim_last_row_tau2, linewidth=2.5, color='purple',
                label='g^K[-1,:] simulated', alpha=0.9)
        ax.plot(time_grid_sim, gk_sim_conv_tau2, linewidth=2, color='orange',
                label='dt*(g^R[-1,:] @ f - f[-1,:] @ g^A) simulated', alpha=0.8, linestyle='--')
        ax.plot(time_grid_two_time, gk_eq_last_row_tau2, linewidth=1.5, color='green',
                label='g^K[-1,:] equilibrium', alpha=0.6, linestyle=':')
        ax.set_xlabel('t\' (time)', fontsize=12)
        ax.set_ylabel('Im[g^K(t_max, t\')]  (τ_2)', fontsize=12)
        ax.set_title('FDT Verification for Simulated State: τ_2 Component', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid_sim[0], 0])

        # Plot τ_3 component
        ax = axes[1]
        ax.plot(time_grid_sim, gk_sim_last_row_tau3, linewidth=2.5, color='purple',
                label='g^K[-1,:] simulated', alpha=0.9)
        ax.plot(time_grid_sim, gk_sim_conv_tau3, linewidth=2, color='orange',
                label='dt*(g^R[-1,:] @ f - f[-1,:] @ g^A) simulated', alpha=0.8, linestyle='--')
        ax.plot(time_grid_two_time, gk_eq_last_row_tau3, linewidth=1.5, color='green',
                label='g^K[-1,:] equilibrium', alpha=0.6, linestyle=':')
        ax.set_xlabel('t\' (time)', fontsize=12)
        ax.set_ylabel('Im[g^K(t_max, t\')]  (τ_3)', fontsize=12)
        ax.set_title('FDT Verification for Simulated State: τ_3 Component', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid_sim[0], 0])

        plt.tight_layout()
        plt.show()

        # Compute and print error metrics
        print(f"\nError analysis:")
        print(f"  τ_2: max|g^K - conv| = {np.max(np.abs(gk_sim_last_row_tau2 - gk_sim_conv_tau2)):.6e}")
        print(f"  τ_3: max|g^K - conv| = {np.max(np.abs(gk_sim_last_row_tau3 - gk_sim_conv_tau3)):.6e}")
        print(f"  τ_2: max|g^K_sim - g^K_eq| = {np.max(np.abs(gk_sim_last_row_tau2 - gk_eq_last_row_tau2)):.6e}")
        print(f"  τ_3: max|g^K_sim - g^K_eq| = {np.max(np.abs(gk_sim_last_row_tau3 - gk_eq_last_row_tau3)):.6e}")

        # ========== Gap vs Time ==========
        print(f"\nComputing gap vs time...")

        gap_eq = initial_state.get_gap_history()
        gap_sim = simulated_state.get_gap_history()

        print(f"  Equilibrium gap: {np.mean(np.real(gap_eq)):.6f} (mean over window)")
        print(f"  Simulated gap at t_max: {np.real(gap_sim[-1]):.6f}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_grid_sim, np.real(gap_eq), linewidth=2, color='green',
                label='equilibrium', linestyle='--')
        ax.plot(time_grid_sim, np.real(gap_sim), linewidth=2, color='purple',
                label='simulated')
        ax.set_xlabel('t', fontsize=12)
        ax.set_ylabel('Δ(t)', fontsize=12)
        ax.set_title('Gap vs Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("\nSimulated state analysis complete!")

    except FileNotFoundError:
        print("simulated_state.pkl not found - skipping simulated state analysis")
        print("Run test_real_time.py to generate the simulated state")

    print("="*70)

    # ========== Normalization Checks at Last Time Slice ==========
    print("\n" + "="*70)
    print("Normalization Checks at t = t_max (t1 = -1)")
    print("="*70)

    # Check initial state
    print("\nChecking initial state normalization constraints...")
    n_t_init = initial_state.gr.data.shape[-1]

    gr_errors_init, gr_totals_init = initial_state.check_gr_normalization(-1)
    gk_errors_init, gk_totals_init, gk_components_init = initial_state.check_keldysh_normalization(-1)

    print(f"  g^R max error: {np.max(gr_errors_init):.6e}")
    print(f"  g^R mean error: {np.mean(gr_errors_init[gr_errors_init > 0]):.6e}")
    print(f"  g^K FDT max error: {np.max(gk_errors_init):.6e}")
    print(f"  g^K FDT mean error: {np.mean(gk_errors_init):.6e}")

    # Check simulated state if available
    if 'simulated_state' in locals():
        print("\nChecking simulated state normalization constraints...")

        gr_errors_sim, gr_totals_sim = simulated_state.check_gr_normalization(-1)
        gk_errors_sim, gk_totals_sim, gk_components_sim = simulated_state.check_keldysh_normalization(-1)

        print(f"  g^R max error: {np.max(gr_errors_sim):.6e}")
        print(f"  g^R mean error: {np.mean(gr_errors_sim[gr_errors_sim > 0]):.6e}")
        print(f"  g^K FDT max error: {np.max(gk_errors_sim):.6e}")
        print(f"  g^K FDT mean error: {np.mean(gk_errors_sim):.6e}")

    # Plot errors vs t2 index
    print("\nPlotting normalization errors vs t2 index...")

    if 'simulated_state' in locals():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes = axes.reshape(2, 1)

    # Initial state g^R errors
    ax = axes[0, 0] if 'simulated_state' in locals() else axes[0, 0]
    ax.semilogy(gr_errors_init, linewidth=2, color='blue', marker='.', markersize=3)
    ax.set_xlabel('t2 index', fontsize=12)
    ax.set_ylabel('error (log scale)', fontsize=12)
    ax.set_title('Initial: g^R Normalization at t1=-1', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Initial state g^K errors
    ax = axes[1, 0] if 'simulated_state' in locals() else axes[1, 0]
    ax.semilogy(gk_errors_init, linewidth=2, color='green', marker='.', markersize=3)
    ax.set_xlabel('t2 index', fontsize=12)
    ax.set_ylabel('error (log scale)', fontsize=12)
    ax.set_title('Initial: g^K FDT Constraint at t1=-1', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if 'simulated_state' in locals():
        ax = axes[0, 1]
        ax.semilogy(gr_errors_sim, linewidth=2, color='purple', marker='.', markersize=3)
        ax.set_xlabel('t2 index', fontsize=12)
        ax.set_ylabel('error (log scale)', fontsize=12)
        ax.set_title('Simulated: g^R Normalization at t1=-1', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.semilogy(gk_errors_sim, linewidth=2, color='orange', marker='.', markersize=3)
        ax.set_xlabel('t2 index', fontsize=12)
        ax.set_ylabel('error (log scale)', fontsize=12)
        ax.set_title('Simulated: g^K FDT Constraint at t1=-1', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Analyze Pauli component contributions
    print("\n" + "="*70)
    print("Pauli Component Breakdown of g^K FDT Violation")
    print("="*70)

    pauli_names = ['τ₀', 'τ₁', 'τ₂', 'τ₃']

    # Plot individual Pauli component errors vs t2
    if 'simulated_state' in locals():
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        axes = axes.reshape(3, 1)

    # Initial state: total Pauli components
    ax = axes[0, 0] if 'simulated_state' in locals() else axes[0, 0]
    for pauli_idx in range(4):
        ax.plot(np.abs(gk_totals_init[pauli_idx, :]), label=pauli_names[pauli_idx], linewidth=1.5)
    ax.set_xlabel('t2 index', fontsize=12)
    ax.set_ylabel('|violation|', fontsize=12)
    ax.set_title('Initial: g^K FDT total violation by Pauli component', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Initial state: g^R @ g^K contribution
    ax = axes[1, 0] if 'simulated_state' in locals() else axes[1, 0]
    for pauli_idx in range(4):
        ax.plot(np.abs(gk_components_init['gr_gk_conv'][pauli_idx, :]), label=pauli_names[pauli_idx], linewidth=1.5)
    ax.set_xlabel('t2 index', fontsize=12)
    ax.set_ylabel('|g^R @ g^K|', fontsize=12)
    ax.set_title('Initial: g^R @ g^K convolution by Pauli component', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Initial state: g^K @ g^A contribution
    ax = axes[2, 0] if 'simulated_state' in locals() else axes[2, 0]
    for pauli_idx in range(4):
        ax.plot(np.abs(gk_components_init['gk_ga_conv'][pauli_idx, :]), label=pauli_names[pauli_idx], linewidth=1.5)
    ax.set_xlabel('t2 index', fontsize=12)
    ax.set_ylabel('|g^K @ g^A|', fontsize=12)
    ax.set_title('Initial: g^K @ g^A convolution by Pauli component', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if 'simulated_state' in locals():
        # Simulated state: total Pauli components
        ax = axes[0, 1]
        for pauli_idx in range(4):
            ax.plot(np.abs(gk_totals_sim[pauli_idx, :]), label=pauli_names[pauli_idx], linewidth=1.5)
        ax.set_xlabel('t2 index', fontsize=12)
        ax.set_ylabel('|violation|', fontsize=12)
        ax.set_title('Simulated: g^K FDT total violation by Pauli component', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Simulated state: g^R @ g^K contribution
        ax = axes[1, 1]
        for pauli_idx in range(4):
            ax.plot(np.abs(gk_components_sim['gr_gk_conv'][pauli_idx, :]), label=pauli_names[pauli_idx], linewidth=1.5)
        ax.set_xlabel('t2 index', fontsize=12)
        ax.set_ylabel('|g^R @ g^K|', fontsize=12)
        ax.set_title('Simulated: g^R @ g^K convolution by Pauli component', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Simulated state: g^K @ g^A contribution
        ax = axes[2, 1]
        for pauli_idx in range(4):
            ax.plot(np.abs(gk_components_sim['gk_ga_conv'][pauli_idx, :]), label=pauli_names[pauli_idx], linewidth=1.5)
        ax.set_xlabel('t2 index', fontsize=12)
        ax.set_ylabel('|g^K @ g^A|', fontsize=12)
        ax.set_title('Simulated: g^K @ g^A convolution by Pauli component', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Print comparison
        print(f"\n  Initial state max g^K FDT error:   {np.max(gk_errors_init):.6e} (at t2={np.argmax(gk_errors_init)})")
        print(f"  Simulated state max g^K FDT error: {np.max(gk_errors_sim):.6e} (at t2={np.argmax(gk_errors_sim)})")
        print(f"  Error increase during evolution:   {np.max(gk_errors_sim) / np.max(gk_errors_init):.2f}x")

    plt.tight_layout()
    plt.show()

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
