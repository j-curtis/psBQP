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


def plot_normalization_checks(state, time_grid, thermal_dist, thermal_integral, state_name='State', t1_idx=-1):
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
    gk_errors, gk_totals, gk_components = state.check_keldysh_normalization(t1_idx, thermal_dist, thermal_integral)

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


def plot_fdt_check(state, time_grid, thermal_dist, thermal_integral, state_name='State', time_index=-1):
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
        time_index=time_index
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

    # Extract equilibrium gap history
    gap_eq_history = equilibrium_state.get_gap_history()
    gap_eq = gap_eq_history[-1]
    print(f"  Equilibrium gap: Δ = {np.real(gap_eq):.6f}")
    print()

    # Plot equilibrium gap evolution
    print("Plotting equilibrium gap evolution...")
    time_grid = evolution.time_grid
    plot_gap_vs_time(time_grid, gap_eq_history, filename='gap_equilibrium.png')
    print()

    # ========== Generate Thermal Distribution and Integral ==========
    print("Generating thermal distribution f(t,t') and integral F(t,t')...")
    evolution.get_thermal_occupation(system_parameters['temperature'])
    thermal_dist = evolution.thermal_dist
    print(f"  Thermal distribution shape: {thermal_dist.data.shape}")

    evolution.get_thermal_integral(system_parameters['temperature'])
    thermal_integral = evolution.thermal_integral
    print(f"  Thermal integral shape: {thermal_integral.data.shape}")

    # Verify F(0) is set to BCS value
    F_zero = thermal_integral.data[0, 0, 0, 0]
    bcs_coupling = evolution._get_BCS_coupling()
    F_zero_expected = 1.0 / bcs_coupling + np.log(system_parameters['critical_temperature'] / system_parameters['temperature'])
    print(f"  F(0) = {np.real(F_zero):.6f} (expected BCS value: {F_zero_expected:.6f})")
    print()

    # ========== Plot Thermal Distribution Properties ==========
    print("Plotting thermal distribution and integral properties...")

    # Extract f(t,t') and F(t,t') data (tau0 component)
    f_data = thermal_dist.trace(pauli_index=0) / 2.0  # Shape: (N_t, N_t)
    F_data = thermal_integral.trace(pauli_index=0) / 2.0  # Shape: (N_t, N_t)

    # Compute sums: ∫ f(t,t') dt' and ∫ f(t,t') dt
    dt = evolution.delta_t
    sum_over_tprime = np.sum(f_data, axis=1) * dt  # Sum over t' for each t
    sum_over_t = np.sum(f_data, axis=0) * dt  # Sum over t for each t'

    # Extract last row of thermal integral: F(t=-1, t')
    F_last_row = F_data[-1, :]

    # Create single plot with real and imaginary parts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Real parts
    ax1.plot(time_grid, np.real(F_last_row), 'b-', linewidth=2, label='F(t=-1, t\')', alpha=0.8)
    ax1.plot(time_grid, np.real(sum_over_tprime), 'r--', linewidth=2, label='∫ f(t,t\') dt\'', alpha=0.8)
    ax1.plot(time_grid, np.real(sum_over_t), 'g-.', linewidth=2, label='∫ f(t,t\') dt', alpha=0.8)
    ax1.set_xlabel("t'", fontsize=11)
    ax1.set_ylabel("Real part", fontsize=11)
    ax1.set_title("Thermal Integral - Real Part", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time_grid[0], time_grid[-1]])

    # Plot 2: Imaginary parts
    ax2.plot(time_grid, np.imag(F_last_row), 'b-', linewidth=2, label='F(t=-1, t\')', alpha=0.8)
    ax2.plot(time_grid, np.imag(sum_over_tprime), 'r--', linewidth=2, label='∫ f(t,t\') dt\'', alpha=0.8)
    ax2.plot(time_grid, np.imag(sum_over_t), 'g-.', linewidth=2, label='∫ f(t,t\') dt', alpha=0.8)
    ax2.set_xlabel("t'", fontsize=11)
    ax2.set_ylabel("Imaginary part", fontsize=11)
    ax2.set_title("Thermal Integral - Imaginary Part", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time_grid[0], time_grid[-1]])

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"  F(t=-1, t') statistics:")
    print(f"    Real: min={np.min(np.real(F_last_row)):.6e}, max={np.max(np.real(F_last_row)):.6e}")
    print(f"    Imag: min={np.min(np.imag(F_last_row)):.6e}, max={np.max(np.imag(F_last_row)):.6e}")
    print(f"  ∫ f(t,t') dt' statistics:")
    print(f"    Real: min={np.min(np.real(sum_over_tprime)):.6e}, max={np.max(np.real(sum_over_tprime)):.6e}")
    print(f"    Imag: min={np.min(np.imag(sum_over_tprime)):.6e}, max={np.max(np.imag(sum_over_tprime)):.6e}")
    print(f"  ∫ f(t,t') dt statistics:")
    print(f"    Real: min={np.min(np.real(sum_over_t)):.6e}, max={np.max(np.real(sum_over_t)):.6e}")
    print(f"    Imag: min={np.min(np.imag(sum_over_t)):.6e}, max={np.max(np.imag(sum_over_t)):.6e}")
    print()

    # ========== Check FDT Using StateObject Method ==========
    print("Checking FDT relation using check_fdt() method...")
    print("Computing regularized FDT convolution for last time row...")

    # Use check_fdt to compute FDT prediction with precise convolution
    gk_fdt_row, gk_actual_row, error_row, max_error = equilibrium_state.check_fdt(
        thermal_dist,
        thermal_integral,
        time_index=-1
    )

    print("✓ FDT check computed with precise convolution")
    print(f"  Maximum absolute error: {max_error:.6e}")
    print()

    # ========== Plot FDT Check for Equilibrium ==========
    print("Plotting FDT check for equilibrium state...")
    plot_fdt_check(equilibrium_state, time_grid, thermal_dist, thermal_integral,
                   state_name='Equilibrium', time_index=-1)
    print()

    # ========== Check Equilibrium Normalization Relations ==========
    print()
    print("="*70)
    print("Equilibrium State: Normalization Checks")
    print("="*70)
    plot_normalization_checks(equilibrium_state, time_grid, thermal_dist, thermal_integral, state_name='Equilibrium', t1_idx=-1)
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
            time_index=-1
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
        plot_fdt_check(evolved_state, time_grid, thermal_dist, thermal_integral,
                       state_name='Evolved', time_index=-1)
        print()

        # ========== Check Evolved State Normalization Relations ==========
        print()
        print("="*70)
        print("Evolved State: Normalization Checks")
        print("="*70)
        plot_normalization_checks(evolved_state, time_grid, thermal_dist, thermal_integral, state_name='Evolved', t1_idx=-1)
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
