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

    # ========== Compute Errors for Each Pauli Component ==========
    print("Analyzing FDT errors for last row g^K(t_max, t')...")

    time_grid = evolution.time_grid

    # Compute errors for each Pauli component using trace method
    print("\nLast row g^K(t_max, t') errors:")
    for pauli_idx, pauli_name in enumerate(['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']):
        # Use trace to extract Pauli components
        gk_actual_pauli = gk_actual_row.trace(pauli_idx) / 2
        gk_fdt_pauli = gk_fdt_row.trace(pauli_idx) / 2
        error_pauli = error_row.trace(pauli_idx) / 2

        max_error_pauli = np.max(np.abs(error_pauli))
        max_val = np.max(np.abs(gk_actual_pauli))
        rel_error = max_error_pauli / (max_val + 1e-10)

        print(f"  {pauli_name}: max|Δg^K| = {max_error_pauli:.6e}, relative = {rel_error*100:.2f}%")

    # ========== Plot Last Row Elements ==========
    print("\nPlotting last row g^K(t_max, t')...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx, (ax, pauli_name) in enumerate(zip(axes.flat, ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)'])):
        # Extract Pauli components using trace method
        gk_actual_pauli = gk_actual_row.trace(pauli_idx) / 2
        gk_fdt_pauli = gk_fdt_row.trace(pauli_idx) / 2

        # Plot real and imaginary parts
        ax.plot(time_grid, np.real(gk_actual_pauli[0, :]), 'b-', linewidth=2,
                label='g^K actual (Re)', alpha=0.7)
        ax.plot(time_grid, np.real(gk_fdt_pauli[0, :]), 'r--', linewidth=1.5,
                label='FDT prediction (Re)', alpha=0.7)
        ax.plot(time_grid, np.imag(gk_actual_pauli[0, :]), 'c-', linewidth=2,
                label='g^K actual (Im)', alpha=0.7)
        ax.plot(time_grid, np.imag(gk_fdt_pauli[0, :]), 'm--', linewidth=1.5,
                label='FDT prediction (Im)', alpha=0.7)

        ax.set_xlabel("t' (time)", fontsize=10)
        ax.set_ylabel(f'g^K(t_max, t\') - {pauli_name}', fontsize=10)
        ax.set_title(f'Last Row: {pauli_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.tight_layout()
    plt.suptitle('Equilibrium State: FDT Verification', fontsize=14, fontweight='bold', y=1.00)
    plt.show()

    # ========== Diagnostic: Near-Diagonal Error Analysis ==========
    print()
    print("="*70)
    print("Near-Diagonal FDT Error Analysis")
    print("="*70)
    print()

    # Extract distance from diagonal
    n_points = len(time_grid)
    distances = np.abs(np.arange(n_points) - (n_points-1))

    # Error for each Pauli component
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx, (ax, pauli_name) in enumerate(zip(axes.flat, ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)'])):
        error_pauli = error_row.trace(pauli_idx) / 2
        error_magnitude = np.abs(error_pauli[0, :])

        # Plot on log scale
        ax.semilogy(distances, error_magnitude, 'b.-', linewidth=2, markersize=4, label='FDT Error')
        ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='1 step off')
        ax.axvline(x=2, color='g', linestyle='--', alpha=0.5, label='2 steps off')
        ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, label='3 steps off')

        # Print statistics
        near_diag_mask = distances <= 3
        far_diag_mask = distances > 10
        near_error = np.mean(error_magnitude[near_diag_mask])
        far_error = np.mean(error_magnitude[far_diag_mask])
        print(f"{pauli_name}: near-diag error = {near_error:.6e}, far-diag error = {far_error:.6e}, ratio = {near_error/far_error:.1f}x")

        ax.set_xlabel('Distance from diagonal (Δt units)', fontsize=10)
        ax.set_ylabel(f'|FDT Error| - {pauli_name}', fontsize=10)
        ax.set_title(f'FDT Error vs Distance: {pauli_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 50])

    plt.tight_layout()
    plt.suptitle('Equilibrium State: FDT Error vs Distance', fontsize=14, fontweight='bold', y=1.00)
    plt.show()
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

        # Analyze errors for each Pauli component
        print("Evolved state g^K(t_max, t') FDT errors:")
        for pauli_idx, pauli_name in enumerate(['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']):
            gk_actual_pauli = gk_actual_evolved.trace(pauli_idx) / 2
            gk_fdt_pauli = gk_fdt_evolved.trace(pauli_idx) / 2
            error_pauli = error_evolved.trace(pauli_idx) / 2

            max_error_pauli = np.max(np.abs(error_pauli))
            max_val = np.max(np.abs(gk_actual_pauli))
            rel_error = max_error_pauli / (max_val + 1e-10)

            print(f"  {pauli_name}: max|Δg^K| = {max_error_pauli:.6e}, relative = {rel_error*100:.2f}%")

        # ========== Plot Evolved State Last Row ==========
        print("\nPlotting evolved state g^K(t_max, t')...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for pauli_idx, (ax, pauli_name) in enumerate(zip(axes.flat, ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)'])):
            # Extract Pauli components using trace method
            gk_actual_pauli = gk_actual_evolved.trace(pauli_idx) / 2
            gk_fdt_pauli = gk_fdt_evolved.trace(pauli_idx) / 2

            # Plot real and imaginary parts
            ax.plot(time_grid, np.real(gk_actual_pauli[0, :]), 'b-', linewidth=2,
                    label='g^K actual (Re)', alpha=0.7)
            ax.plot(time_grid, np.real(gk_fdt_pauli[0, :]), 'r--', linewidth=1.5,
                    label='FDT prediction (Re)', alpha=0.7)
            ax.plot(time_grid, np.imag(gk_actual_pauli[0, :]), 'c-', linewidth=2,
                    label='g^K actual (Im)', alpha=0.7)
            ax.plot(time_grid, np.imag(gk_fdt_pauli[0, :]), 'm--', linewidth=1.5,
                    label='FDT prediction (Im)', alpha=0.7)

            ax.set_xlabel("t' (time)", fontsize=10)
            ax.set_ylabel(f'g^K(t_max, t\') - {pauli_name}', fontsize=10)
            ax.set_title(f'Evolved State Last Row: {pauli_name}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([time_grid[0], 0])

        plt.tight_layout()
        plt.suptitle('Evolved State: FDT Verification', fontsize=14, fontweight='bold', y=1.00)
        plt.show()

    except FileNotFoundError:
        print("⚠ simulated_state.pkl not found - skipping evolved state FDT check")
        print("  Run test_real_time.py first to generate evolved state")

    print()
    print("="*70)
    print("FDT Verification Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
