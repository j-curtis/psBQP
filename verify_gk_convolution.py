"""
Verify equilibrium g^K satisfies FDT relation in two-time domain.

Computes equilibrium state and verifies:
g^K(t,t') = ∫ dt'' [g^R(t,t'') f(t'',t') - f(t,t'') g^A(t'',t')]

Uses the StateObject.check_fdt() method with precise convolution.
"""

import numpy as np
import matplotlib.pyplot as plt

from nambu_keldysh_class import NambuKeldyshTensor
from usadel_keldysh_evolution import UsadelKeldyshEvolution


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

    # ========== Create Evolution Object and Generate Equilibrium ==========
    print("Creating evolution object...")
    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
    print(f"  dt = {evolution.delta_t:.6f}")
    print(f"  Time grid: [{evolution.time_grid[0]:.4f}, {evolution.time_grid[-1]:.4f}]")
    print(f"  BCS coupling: {evolution._get_BCS_coupling():.4f}")
    print()

    print("Computing equilibrium state...")
    equilibrium_state, gr_tau, gk_tau = evolution.generate_initial_state(Q=0.0)
    print("✓ Equilibrium state computed")
    print(f"  g^R shape: {equilibrium_state.gr.data.shape}")
    print(f"  g^K shape: {equilibrium_state.gk.data.shape}")

    # Extract equilibrium gap
    gap_eq = equilibrium_state.get_gap_history()[-1]
    print(f"  Equilibrium gap: Δ = {np.real(gap_eq):.6f}")
    print()

    # ========== Generate Thermal Distribution and Integral ==========
    print("Generating thermal distribution f(t,t') and integral F(t,t')...")
    evolution.get_thermal_occupation(system_parameters['temperature'])
    thermal_dist = evolution.thermal_dist
    print(f"  Thermal distribution shape: {thermal_dist.data.shape}")

    evolution.get_thermal_integral(system_parameters['temperature'])
    thermal_integral = evolution.thermal_integral
    print(f"  Thermal integral shape: {thermal_integral.data.shape}")
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
    plt.savefig('fdt_verification_last_row.png', dpi=150, bbox_inches='tight')
    print("  Saved: fdt_verification_last_row.png")
    plt.show()

    print()
    print("="*70)
    print("FDT Verification Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
