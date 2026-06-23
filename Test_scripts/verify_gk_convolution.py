"""
Verify g^K satisfies FDT relation in two-time domain.

Loads initial state from initial_state.pkl and verifies:
g^K(t,t') = ∫ dt'' [g^R(t,t'') f(t'',t') - f(t,t'') g^A(t'',t')]

Uses the StateObject.check_fdt() method with precise convolution.

Can also verify FDT for evolved states after real-time evolution by loading simulated_state.pkl.

Usage:
    1. Run test_real_time.py to generate initial_state.pkl and simulated_state.pkl
    2. Run this script to verify FDT and plot gap evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from nambu_keldysh_class import NambuKeldyshTensor
from usadel_keldysh_evolution import UsadelKeldyshEvolution


def plot_gap_vs_time(time_grid, gap_eq, gap_evolved=None, filename='gap_evolution.png'):
    """
    Plot gap evolution over time.

    Args:
        time_grid: Array of time values
        gap_eq: Gap history for equilibrium state (array)
        gap_evolved: Optional gap history for evolved state (array)
        filename: Output filename for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot real part
    ax1.plot(time_grid, np.real(gap_eq), 'b-', linewidth=2, label='Equilibrium', alpha=0.8)
    if gap_evolved is not None:
        ax1.plot(time_grid, np.real(gap_evolved), 'r--', linewidth=2, label='Evolved', alpha=0.8)
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Re[Δ(t)]', fontsize=12)
    ax1.set_title('Gap Evolution - Real Part', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time_grid[0], time_grid[-1]])

    # Plot imaginary part
    ax2.plot(time_grid, np.imag(gap_eq), 'b-', linewidth=2, label='Equilibrium', alpha=0.8)
    if gap_evolved is not None:
        ax2.plot(time_grid, np.imag(gap_evolved), 'r--', linewidth=2, label='Evolved', alpha=0.8)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Im[Δ(t)]', fontsize=12)
    ax2.set_title('Gap Evolution - Imaginary Part', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time_grid[0], time_grid[-1]])

    plt.tight_layout()
    plt.show()
    # plt.savefig(filename, dpi=150, bbox_inches='tight')
    # print(f"  Saved: {filename}")


def plot_normalization_checks(state, time_grid, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, state_name='State', t1_idx=-1):
    """
    Plot g^R and g^K normalization checks for the last row.

    Args:
        state: StateObject with Green's functions
        time_grid: Array of time values
        thermal_dist: Thermal distribution f(t,t')
        thermal_integral: Integral of thermal distribution F(t,t')
        state_name: Name for plot title (e.g., 'Equilibrium', 'Evolved')
        t1_idx: Time index to check (default -1 for last row)
    """
    print(f"\nChecking {state_name} normalization relations for last row (t1_idx={t1_idx})...")

    # ========== g^R Normalization ==========
    print("\n  Computing g^R normalization...")
    gr_errors, gr_totals = state.check_gr_normalization(t1_idx)

    # Extract Pauli components from totals
    gr_tau0 = gr_totals[0, :]
    gr_tau1 = gr_totals[1, :]
    gr_tau2 = gr_totals[2, :]
    gr_tau3 = gr_totals[3, :]

    print(f"    Maximum g^R normalization error: {np.max(gr_errors):.6e}")

    # ========== g^K Normalization ==========
    print("\n  Computing g^K normalization...")
    gk_errors, gk_totals, gk_components = state.check_keldysh_normalization(t1_idx, thermal_dist, thermal_integral, thermal_sum_left=thermal_sum_left, thermal_sum_right=thermal_sum_right)

    # Extract Pauli components from totals
    gk_tau0 = gk_totals[0, :]
    gk_tau1 = gk_totals[1, :]
    gk_tau2 = gk_totals[2, :]
    gk_tau3 = gk_totals[3, :]

    print(f"    Maximum g^K normalization error: {np.max(gk_errors):.6e}")

    # ========== Plot g^R Normalization ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pauli_names = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']
    gr_components = [gr_tau0, gr_tau1, gr_tau2, gr_tau3]

    for ax, pauli_name, component in zip(axes.flat, pauli_names, gr_components):
        # Plot real and imaginary parts
        ax.plot(time_grid[:len(component)], np.real(component), 'b-', linewidth=2,
                label='Real part', alpha=0.7)
        ax.plot(time_grid[:len(component)], np.imag(component), 'r-', linewidth=2,
                label='Imaginary part', alpha=0.7)

        ax.set_xlabel("t₂ (time)", fontsize=10)
        ax.set_ylabel(f'Normalization violation - {pauli_name}', fontsize=10)
        ax.set_title(f'g^R Normalization: {pauli_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.tight_layout()
    plt.suptitle(f'{state_name} State: g^R Normalization Check (Last Row)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.show()

    # ========== Plot g^K Normalization ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gk_components_list = [gk_tau0, gk_tau1, gk_tau2, gk_tau3]

    for ax, pauli_name, component in zip(axes.flat, pauli_names, gk_components_list):
        # Plot real and imaginary parts
        ax.plot(time_grid, np.real(component), 'b-', linewidth=2,
                label='Real part', alpha=0.7)
        ax.plot(time_grid, np.imag(component), 'r-', linewidth=2,
                label='Imaginary part', alpha=0.7)

        ax.set_xlabel("t₂ (time)", fontsize=10)
        ax.set_ylabel(f'Normalization violation - {pauli_name}', fontsize=10)
        ax.set_title(f'g^K Normalization: {pauli_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.tight_layout()
    plt.suptitle(f'{state_name} State: g^K Normalization Check (Last Row)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.show()

    # Print statistics
    print(f"\n  g^R normalization errors by Pauli component:")
    for pauli_name, component in zip(pauli_names, gr_components):
        max_err = np.max(np.abs(component))
        print(f"    {pauli_name}: max error = {max_err:.6e}")

    print(f"\n  g^K normalization errors by Pauli component:")
    for pauli_name, component in zip(pauli_names, gk_components_list):
        max_err = np.max(np.abs(component))
        print(f"    {pauli_name}: max error = {max_err:.6e}")


def plot_thermal_integrals(time_grid, thermal_dist, thermal_integral):
    """
    Plot integrals of f(t,t'') and f(t'',t') vs t'' along with thermal integral F(t,t').

    Properly accounts for integration limits:
    - ∫dt'' f(t,t'') integrates from -T_max up to t' (respects causality for each t')
    - ∫dt'' f(t'',t') integrates from -T_max up to t (fixed upper limit)

    Args:
        time_grid: Array of time values
        thermal_dist: Thermal distribution f(t,t') tensor
        thermal_integral: Thermal integral F(t,t') tensor
    """
    print("\nPlotting thermal distribution integrals...")

    # Extract tau_0 component (identity) from f and F
    f_tau0 = thermal_dist.trace(0) / 2  # Shape: (Nt, Nt)
    F_tau0 = thermal_integral.trace(0) / 2  # Shape: (Nt, Nt)

    # Pick last row: t = 0 (end of time grid)
    t_idx = -1
    t_value = time_grid[t_idx]

    F_row = F_tau0[t_idx, :]  # F(t, t'') vs t''

    # Time step
    dt = time_grid[1] - time_grid[0]
    Nt = len(time_grid)

    # ========== Use NambuKeldyshTensor convolution operations ==========
    # Extract f as a row for the last time: f(t, t') with t = t_idx
    # Convert negative index to positive to avoid slicing issues
    t_idx_pos = Nt + t_idx if t_idx < 0 else t_idx
    f_row_tensor = thermal_dist[t_idx_pos:t_idx_pos+1, :]  # Shape: (2, 2, 1, Nt)

    # Create ones tensor as a row vector (all ones)
    # Shape: (2, 2, 1, Nt) - identity in Nambu space
    ones_row_data = np.ones((1, Nt), dtype=complex)
    ones_row = NambuKeldyshTensor(ones_row_data, pauli_channel=0)

    # Create causality matrix: ones_data[i, j] = 1 if i <= j, else 0
    # This enforces integration up to each t'
    row_indices = np.arange(Nt)[:, np.newaxis]  # Shape (Nt, 1)
    col_indices = np.arange(Nt)[np.newaxis, :]  # Shape (1, Nt)
    ones_causal_data = (row_indices <= col_indices).astype(complex)
    ones_causal = NambuKeldyshTensor(ones_causal_data, pauli_channel=0)

    # Create filter function to suppress early-time artifacts
    # Zero out first 1/5 of time points (same as in precise_convolution_left)
    filter_data = np.append(np.zeros(Nt//5), np.ones(Nt - Nt//5))
    filter_function = NambuKeldyshTensor(filter_data, pauli_channel=0)

    # Compute convolutions using @ operator
    # 1. ones @ f: integrates over first index (sums columns)
    # This gives ∫dt'' f(t'',t') for each t'
    # Apply filter to suppress early-time artifacts
    ones_at_f_filtered = (ones_row @ thermal_dist) * filter_function
    ones_at_f = ones_at_f_filtered * dt
    f_col_integral_trace = ones_at_f.trace(0) / 2  # Extract tau_0 component
    # Handle shape: could be (1, Nt) or (Nt,)
    if f_col_integral_trace.ndim == 2:
        f_col_integral_conv = f_col_integral_trace[0, :]  # Flatten to 1D
    elif f_col_integral_trace.ndim == 1:
        f_col_integral_conv = f_col_integral_trace  # Already 1D
    else:
        # Unexpected shape
        f_col_integral_conv = f_col_integral_trace.flatten()

    # 2. f @ ones_causal: integrates with causality (respects t <= t')
    # This gives ∫_{-T_max}^{t'} dt'' f(t,t'') for each t'
    f_at_ones = (f_row_tensor @ ones_causal) * dt
    f_row_integral_trace = f_at_ones.trace(0) / 2  # Extract tau_0 component
    # Handle shape: could be (1, Nt) or (Nt,)
    if f_row_integral_trace.ndim == 2:
        f_row_integral_conv = f_row_integral_trace[0, :]  # Flatten to 1D
    elif f_row_integral_trace.ndim == 1:
        f_row_integral_conv = f_row_integral_trace  # Already 1D
    else:
        # Unexpected shape
        f_row_integral_conv = f_row_integral_trace.flatten()

    # Create single plot with all integrals
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot f @ ones_causal: ∫_{-T_max}^{t'} dt'' f(t,t'') with causality
    ax.plot(time_grid, np.real(f_row_integral_conv), 'b-', linewidth=2,
            label='f @ ones_causal (∫ up to t\') - Real', alpha=0.8)
    ax.plot(time_grid, np.imag(f_row_integral_conv), 'b--', linewidth=2,
            label='f @ ones_causal (∫ up to t\') - Imag', alpha=0.8)

    # Plot ones @ f: ∫_{-T_max}^{t} dt'' f(t'',t') (full column sum)
    # Make red lines THICKER to be more visible
    ax.plot(time_grid, np.real(f_col_integral_conv), 'r-', linewidth=4,
            label='ones @ f (∫ all t\'\') - Real', alpha=0.9)
    ax.plot(time_grid, np.imag(f_col_integral_conv), 'r--', linewidth=4,
            label='ones @ f (∫ all t\'\') - Imag', alpha=0.9)

    # Plot -F(t,t') (analytic thermal integral)
    # F(t,t') = ∫_{-∞}^{t-t'} f(τ) dτ in the code
    # But on finite grid: F(t,t') = ∫_{-T_max-t'}^{t-t'} f(τ) dτ + corrections
    ax.plot(time_grid, -np.real(F_row), 'g-', linewidth=2,
            label='-F(t,t\') (analytic) - Real', alpha=0.7)
    ax.plot(time_grid, -np.imag(F_row), 'g--', linewidth=2,
            label='-F(t,t\') (analytic) - Imag', alpha=0.7)

    ax.set_xlabel('t\' (time)', fontsize=12)
    ax.set_ylabel('Thermal Integrals', fontsize=12)
    ax.set_title(f'Thermal Distribution Integrals (t = {t_value:.3f})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_grid[0], time_grid[-1]])

    plt.tight_layout()
    plt.show()

    # Print comparison statistics
    print(f"  Convolution operations:")
    print(f"    f @ ones_causal: ∫_{'{-T_max}'}^{'{t\'}'} dt'' f(t,t'') (respects causality)")
    print(f"    ones @ f: ∫_{'{-T_max}'}^{'{t}'} dt'' f(t'',t') = {t_value:.3f} (full column sum)")
    print(f"\n  Comparing convolution integrals with -F(t,t'):")

    diff_row_real = np.max(np.abs(np.real(f_row_integral_conv) - (-np.real(F_row))))
    diff_row_imag = np.max(np.abs(np.imag(f_row_integral_conv) - (-np.imag(F_row))))
    print(f"    f @ ones_causal vs -F(t,t'):")
    print(f"      Maximum difference (Real): {diff_row_real:.6e}")
    print(f"      Maximum difference (Imag): {diff_row_imag:.6e}")

    diff_col_real = np.max(np.abs(np.real(f_col_integral_conv) - (-np.real(F_row))))
    diff_col_imag = np.max(np.abs(np.imag(f_col_integral_conv) - (-np.imag(F_row))))
    print(f"    ones @ f vs -F(t,t'):")
    print(f"      Maximum difference (Real): {diff_col_real:.6e}")
    print(f"      Maximum difference (Imag): {diff_col_imag:.6e}")

    print(f"\n    Note: F is computed from -∞ with corrections, causing discrepancy at boundaries")
    print(f"    -F(t,t'=t) at diagonal: {-F_row[-1]:.6f}")
    print()


def plot_fdt_check(state, time_grid, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, state_name='State', time_index=-1):
    """
    Plot FDT relation check: g^K vs (g^R @ f - f @ g^A).

    Shows actual g^K, FDT prediction, and error for each Pauli component.

    Args:
        state: StateObject with Green's functions
        time_grid: Array of time values
        thermal_dist: Thermal distribution f(t,t')
        thermal_integral: Integral of thermal distribution F(t,t')
        state_name: Name for plot title (e.g., 'Equilibrium', 'Evolved')
        time_index: Time index to check (default -1 for last row)
    """
    print(f"\n{state_name} State: FDT Check for row at time_index={time_index}")

    # Compute FDT using precise convolution
    gk_fdt_row, gk_actual_row, error_row, max_error = state.check_fdt(
        thermal_dist,
        thermal_integral,
        time_index=time_index,
        thermal_sum_left=thermal_sum_left,
        thermal_sum_right=thermal_sum_right
    )

    print(f"  Maximum absolute error: {max_error:.6e}")

    # Extract Pauli components
    pauli_names = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']

    # Create figure with 2x2 grid (one subplot per Pauli component)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx, pauli_name in enumerate(pauli_names):
        # Extract Pauli components using trace
        gk_actual_pauli = gk_actual_row.trace(pauli_idx) / 2  # Shape: (1, Nt)
        gk_fdt_pauli = gk_fdt_row.trace(pauli_idx) / 2
        error_pauli = error_row.trace(pauli_idx) / 2

        # Flatten to 1D arrays
        gk_actual_pauli = gk_actual_pauli[0, :]
        gk_fdt_pauli = gk_fdt_pauli[0, :]
        error_pauli = error_pauli[0, :]

        # Get subplot position (2x2 grid)
        ax = axes.flat[pauli_idx]

        # Plot actual g^K (solid lines)
        ax.plot(time_grid, np.real(gk_actual_pauli), 'b-', linewidth=2,
                label='Actual (Real)', alpha=0.8)
        ax.plot(time_grid, np.imag(gk_actual_pauli), 'r-', linewidth=2,
                label='Actual (Imag)', alpha=0.8)
        # Plot FDT prediction (dashed lines)
        ax.plot(time_grid, np.real(gk_fdt_pauli), 'b--', linewidth=2,
                label='FDT (Real)', alpha=0.6)
        ax.plot(time_grid, np.imag(gk_fdt_pauli), 'r--', linewidth=2,
                label='FDT (Imag)', alpha=0.6)

        ax.set_xlabel('Time t\'', fontsize=10)
        ax.set_ylabel(f'g^K - {pauli_name}', fontsize=10)
        ax.set_title(f'{pauli_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

        # Compute max error for this component
        max_err = np.max(np.abs(error_pauli))
        max_val = np.max(np.abs(gk_actual_pauli))
        rel_err = max_err / (max_val + 1e-10) * 100

        # Add error info as text in corner
        ax.text(0.02, 0.98, f'max err: {max_err:.2e}\nrel: {rel_err:.1f}%',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.suptitle(f'{state_name} State: FDT Relation Check (Row at t={time_grid[time_index]:.3f})',
                 fontsize=14, fontweight='bold', y=0.998)
    plt.subplots_adjust(top=0.97)
    plt.show()

    # Print component-wise statistics
    print(f"\n  FDT errors by Pauli component:")
    for pauli_idx, pauli_name in enumerate(pauli_names):
        gk_actual_pauli = gk_actual_row.trace(pauli_idx) / 2
        error_pauli = error_row.trace(pauli_idx) / 2
        max_err = np.max(np.abs(error_pauli))
        max_val = np.max(np.abs(gk_actual_pauli))
        rel_err = max_err / (max_val + 1e-10) * 100
        print(f"    {pauli_name}: max|error| = {max_err:.6e}, relative = {rel_err:.2f}%")


def main():
    print("="*70)
    print("Verify Equilibrium g^K - FDT Relation in Two-Time Domain")
    print("="*70)
    print()

    # ========== Setup Parameters ==========
    grid_parameters = {
        'time_sampling': 1501,
        'time_duration': 2 * np.pi * 5,
        'eta': 0.2
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

    # ========== Load Initial State ==========
    print("Loading initial state from initial_state.pkl...")
    try:
        with open('initial_state_test.pkl', 'rb') as f:
            equilibrium_state = pickle.load(f)
        print("✓ Initial state loaded")
        print(f"  g^R shape: {equilibrium_state.gr.data.shape}")
        print(f"  g^K shape: {equilibrium_state.gk.data.shape}")
    except FileNotFoundError:
        print("✗ initial_state_test.pkl not found!")
        print("  Please run test_real_time.py first to generate initial_state_test.pkl")
        return

    # ========== Create Evolution Object for Grid Parameters ==========
    print("\nCreating evolution object for grid parameters...")
    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
    print(f"  dt = {evolution.delta_t:.6f}")
    print(f"  Time grid: [{evolution.time_grid[0]:.4f}, {evolution.time_grid[-1]:.4f}]")
    print(f"  BCS coupling: {evolution._get_BCS_coupling():.4f}")
    print()

    # ========== Generate Thermal Distribution and Integral (BEFORE gap extraction) ==========
    print("Generating thermal distribution f(t,t') and integral F(t,t')...")
    evolution.get_thermal_occupation(system_parameters['temperature'])
    thermal_dist = evolution.thermal_dist
    print(f"  Thermal distribution shape: {thermal_dist.data.shape}")

    evolution.get_thermal_integral(system_parameters['temperature'])
    thermal_integral = evolution.thermal_integral
    print(f"  Thermal integral shape: {thermal_integral.data.shape}")

    evolution.get_thermal_sum(system_parameters['temperature'])
    thermal_sum_left = evolution.thermal_sum_left
    thermal_sum_right = evolution.thermal_sum_right
    print(f"  Thermal sum shape: {thermal_sum_left.data.shape}")

    # Verify F(0) is set to BCS value
    F_zero = thermal_integral.data[0, 0, 0, 0]
    bcs_coupling = evolution._get_BCS_coupling()
    F_zero_expected = 1.0 / bcs_coupling + np.log(system_parameters['critical_temperature'] / system_parameters['temperature'])
    print(f"  F(0) = {np.real(F_zero):.6f} (expected BCS value: {F_zero_expected:.6f})")
    print()

    # ========== Plot Thermal Integrals ==========
    time_grid = evolution.time_grid
    plot_thermal_integrals(time_grid, thermal_dist, thermal_integral)

    # ========== Extract and Plot Gap ==========
    # Extract equilibrium gap history
    gap_eq_history = equilibrium_state.get_gap_history()
    gap_eq = gap_eq_history[-1]
    print(f"  Equilibrium gap: Δ = {np.real(gap_eq):.6f}")
    print()

    # Plot equilibrium gap evolution
    print("Plotting equilibrium gap evolution...")
    plot_gap_vs_time(time_grid, gap_eq_history, filename='gap_equilibrium.png')
    print()

    # ========== Check FDT Using StateObject Method ==========
    print("Checking FDT relation using check_fdt() method...")
    print("Computing regularized FDT convolution for last time row...")

    # Use check_fdt to compute FDT prediction with precise convolution
    gk_fdt_row, gk_actual_row, error_row, max_error = equilibrium_state.check_fdt(
        thermal_dist,
        thermal_integral,
        time_index=-1,
        thermal_sum_left=thermal_sum_left,
        thermal_sum_right=thermal_sum_right
    )

    print("✓ FDT check computed with precise convolution")
    print(f"  Maximum absolute error: {max_error:.6e}")
    print()

    # ========== Plot FDT Check for Equilibrium ==========
    print("Plotting FDT check for equilibrium state...")
    plot_fdt_check(equilibrium_state, time_grid, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right,
                   state_name='Equilibrium', time_index=-1)
    print()

    # ========== Check Equilibrium Normalization Relations ==========
    print()
    print("="*70)
    print("Equilibrium State: Normalization Checks")
    print("="*70)
    plot_normalization_checks(equilibrium_state, time_grid, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, state_name='Equilibrium', t1_idx=-1)
    print()

    # ========== Check FDT for Evolved State ==========
    print()
    print("="*70)
    print("Checking FDT for Evolved State (after real-time evolution)")
    print("="*70)
    print()

    # Load evolved state
    try:
        print("Loading evolved state from simulated_state.pkl...")
        with open('simulated_state.pkl', 'rb') as f:
            evolved_state = pickle.load(f)
        print("✓ Evolved state loaded")
        print(f"  g^R shape: {evolved_state.gr.data.shape}")
        print(f"  g^K shape: {evolved_state.gk.data.shape}")

        # Extract gap history from evolved state
        gap_evolved_history = evolved_state.get_gap_history()
        gap_evolved = gap_evolved_history[-1]
        print(f"  Evolved gap: Δ = {np.real(gap_evolved):.6f}")
        print(f"  Gap change: ΔΔ = {np.real(gap_evolved - gap_eq):.6f}")
        print()

        # Plot gap evolution comparison
        print("Plotting gap evolution comparison (equilibrium vs evolved)...")
        plot_gap_vs_time(time_grid, gap_eq_history, gap_evolved_history, filename='gap_evolution_comparison.png')
        print()

        # Check FDT for evolved state
        print("Checking FDT relation for evolved state (last row)...")
        gk_fdt_evolved, gk_actual_evolved, error_evolved, max_error_evolved = evolved_state.check_fdt(
            thermal_dist,
            thermal_integral,
            time_index=-1,
            thermal_sum_left=thermal_sum_left,
            thermal_sum_right=thermal_sum_right
        )

        print("✓ FDT check computed for evolved state")
        print(f"  Maximum absolute error: {max_error_evolved:.6e}")
        print()

        # Compare errors
        print("Comparison: Equilibrium vs Evolved State FDT Errors")
        print(f"  Equilibrium max error: {max_error:.6e}")
        print(f"  Evolved max error:     {max_error_evolved:.6e}")
        print(f"  Ratio (evolved/equilibrium): {max_error_evolved/max_error:.2f}x")
        print()

        # ========== Plot FDT Check for Evolved State ==========
        print("Plotting FDT check for evolved state...")
        plot_fdt_check(evolved_state, time_grid, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right,
                       state_name='Evolved', time_index=-1)
        print()

        # ========== Check Evolved State Normalization Relations ==========
        print()
        print("="*70)
        print("Evolved State: Normalization Checks")
        print("="*70)
        plot_normalization_checks(evolved_state, time_grid, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, state_name='Evolved', t1_idx=-1)
        print()

    except FileNotFoundError:
        print("⚠ simulated_state.pkl not found - skipping evolved state FDT check")
        print("  Run test_real_time.py first to generate evolved state")

    print()
    print("="*70)
    print("FDT Verification Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
