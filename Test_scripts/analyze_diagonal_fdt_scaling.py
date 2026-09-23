"""
Analyze FDT Error Scaling at the Diagonal

Tests how the FDT error at the diagonal element scales with dt for different system sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usadel_keldysh_evolution import UsadelKeldyshEvolution


def analyze_diagonal_error(N_t_values=[501, 751, 1001]):
    """
    Analyze FDT error at diagonal for different grid sizes.

    Args:
        N_t_values: List of time grid sizes to test
    """

    print("="*80)
    print("DIAGONAL FDT ERROR SCALING ANALYSIS")
    print("="*80)
    print()

    results = []

    for N_t in N_t_values:
        print(f"\n{'='*80}")
        print(f"Testing N_t = {N_t}")
        print(f"{'='*80}\n")

        # Fixed time duration
        T_max = 2 * np.pi * 5

        grid_parameters = {
            'time_sampling': N_t,
            'time_duration': T_max,
            'eta': 0.2
        }

        system_parameters = {
            'critical_temperature': 1.0,
            'temperature': 0.1,
            'eta': 0.2
        }

        print(f"Grid: N_t={N_t}, T_max={T_max:.4f}")

        # Create evolution object
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
        dt = evolution.delta_t
        print(f"Time step: dt = {dt:.6f}")
        print()

        # Generate equilibrium state
        print("Generating equilibrium state...")
        initial_state, gr_tau, gk_tau, J_eq = evolution.generate_initial_state(Q=0.0)
        gap = initial_state.get_gap_history()[-1]
        print(f"  Gap: |Δ| = {np.abs(gap):.6f}")

        # Ensure thermal distributions are computed
        if not hasattr(evolution, 'thermal_dist'):
            evolution.get_thermal_occupation(system_parameters['temperature'])
            evolution.get_thermal_integral(system_parameters['temperature'])
            evolution.get_log_two_time(system_parameters['temperature'])
            evolution.get_thermal_sum(system_parameters['temperature'])

        # Check FDT
        print("Checking FDT...")
        gk_fdt, gk_actual, fdt_error, fdt_max_error = initial_state.check_fdt(
            f_thermal=evolution.thermal_dist,
            f_thermal_integral=evolution.thermal_integral,
            time_index=-1,
            thermal_sum_left=evolution.thermal_sum_left,
            thermal_sum_right=evolution.thermal_sum_right,
            log_two_time=evolution.log_two_time,
            tmax=evolution.tmax
        )

        # Extract diagonal element error for both tau_2 and tau_3
        fdt_error_tau2 = fdt_error.trace(pauli_index=2) / 2  # Shape (1, N_t)
        fdt_error_tau3 = fdt_error.trace(pauli_index=3) / 2  # Shape (1, N_t)

        diagonal_error_tau2 = fdt_error_tau2[0, -1]  # Last element is the diagonal
        diagonal_error_tau3 = fdt_error_tau3[0, -1]  # Last element is the diagonal

        # Also check a few off-diagonal elements for comparison
        off_diag_errors_tau2 = []
        off_diag_errors_tau3 = []
        for offset in [1, 2, 3, 5, 10]:
            if N_t > offset:
                off_diag_errors_tau2.append(np.abs(fdt_error_tau2[0, -(offset+1)]))
                off_diag_errors_tau3.append(np.abs(fdt_error_tau3[0, -(offset+1)]))

        avg_off_diag_error_tau2 = np.mean(off_diag_errors_tau2) if off_diag_errors_tau2 else 0
        avg_off_diag_error_tau3 = np.mean(off_diag_errors_tau3) if off_diag_errors_tau3 else 0
        max_error_tau2 = np.max(np.abs(fdt_error_tau2))
        max_error_tau3 = np.max(np.abs(fdt_error_tau3))

        print(f"\nFDT Error Results:")
        print(f"  τ₂ component:")
        print(f"    Diagonal error:          {np.abs(diagonal_error_tau2):.6e}")
        print(f"    Avg off-diagonal (nearby): {avg_off_diag_error_tau2:.6e}")
        print(f"    Max error:               {max_error_tau2:.6e}")
        print(f"    Diagonal / dt:           {np.abs(diagonal_error_tau2)/dt:.6e}")
        print(f"  τ₃ component:")
        print(f"    Diagonal error:          {np.abs(diagonal_error_tau3):.6e}")
        print(f"    Avg off-diagonal (nearby): {avg_off_diag_error_tau3:.6e}")
        print(f"    Max error:               {max_error_tau3:.6e}")
        print(f"    Diagonal / dt:           {np.abs(diagonal_error_tau3)/dt:.6e}")

        results.append({
            'N_t': N_t,
            'dt': dt,
            'diagonal_error_tau2': np.abs(diagonal_error_tau2),
            'diagonal_error_tau3': np.abs(diagonal_error_tau3),
            'avg_off_diag_error_tau2': avg_off_diag_error_tau2,
            'avg_off_diag_error_tau3': avg_off_diag_error_tau3,
            'max_error_tau2': max_error_tau2,
            'max_error_tau3': max_error_tau3,
            'gap': np.abs(gap)
        })

    # Analysis and plotting
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS")
    print(f"{'='*80}\n")

    # Print results table for τ₂
    print(f"\nτ₂ Component:")
    print(f"{'N_t':<8} {'dt':<12} {'Diagonal':<14} {'Off-diag':<14} {'Diag/dt':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['N_t']:<8} {r['dt']:<12.6e} {r['diagonal_error_tau2']:<14.6e} {r['avg_off_diag_error_tau2']:<14.6e} "
              f"{r['diagonal_error_tau2']/r['dt']:<12.2f}")

    # Print results table for τ₃
    print(f"\nτ₃ Component:")
    print(f"{'N_t':<8} {'dt':<12} {'Diagonal':<14} {'Off-diag':<14} {'Diag/dt':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['N_t']:<8} {r['dt']:<12.6e} {r['diagonal_error_tau3']:<14.6e} {r['avg_off_diag_error_tau3']:<14.6e} "
              f"{r['diagonal_error_tau3']/r['dt']:<12.2f}")

    # Check scaling
    print(f"\nScaling Analysis:")
    if len(results) >= 2:
        ratio_dt = results[0]['dt'] / results[-1]['dt']

        # τ₂ component
        print(f"\nτ₂ Component:")
        ratio_diag_tau2 = results[0]['diagonal_error_tau2'] / results[-1]['diagonal_error_tau2']
        print(f"  Diagonal error ratio (N={results[0]['N_t']}/{results[-1]['N_t']}): {ratio_diag_tau2:.4f}")
        print(f"  dt ratio:                                    {ratio_dt:.4f}")
        print(f"  Expected for O(dt):                          {ratio_dt:.4f}")
        print(f"  Expected for O(dt²):                         {ratio_dt**2:.4f}")
        if np.abs(ratio_diag_tau2 - ratio_dt) / ratio_dt < 0.2:
            print(f"  => τ₂ diagonal error scales as O(dt) ✓")
        elif np.abs(ratio_diag_tau2 - ratio_dt**2) / ratio_dt**2 < 0.2:
            print(f"  => τ₂ diagonal error scales as O(dt²) ✓")

        # τ₃ component
        print(f"\nτ₃ Component:")
        ratio_diag_tau3 = results[0]['diagonal_error_tau3'] / results[-1]['diagonal_error_tau3']
        print(f"  Diagonal error ratio (N={results[0]['N_t']}/{results[-1]['N_t']}): {ratio_diag_tau3:.4f}")
        print(f"  dt ratio:                                    {ratio_dt:.4f}")
        print(f"  Expected for O(dt):                          {ratio_dt:.4f}")
        print(f"  Expected for O(dt²):                         {ratio_dt**2:.4f}")
        if np.abs(ratio_diag_tau3 - ratio_dt) / ratio_dt < 0.2:
            print(f"  => τ₃ diagonal error scales as O(dt) ✓")
        elif np.abs(ratio_diag_tau3 - ratio_dt**2) / ratio_dt**2 < 0.2:
            print(f"  => τ₃ diagonal error scales as O(dt²) ✓")

    # Plot results
    print(f"\nGenerating plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    dt_vals = np.array([r['dt'] for r in results])
    diag_errors_tau2 = np.array([r['diagonal_error_tau2'] for r in results])
    diag_errors_tau3 = np.array([r['diagonal_error_tau3'] for r in results])

    # Plot 1: τ₂ Error vs dt (log-log)
    ax1.loglog(dt_vals, diag_errors_tau2, 'ro-', linewidth=2, markersize=8, label='τ₂ Diagonal')
    if len(dt_vals) >= 2:
        ref_dt = diag_errors_tau2[0] * (dt_vals / dt_vals[0])
        ax1.loglog(dt_vals, ref_dt, 'r--', alpha=0.5, linewidth=1.5, label='O(dt) reference')
    ax1.set_xlabel('Time step dt', fontsize=12)
    ax1.set_ylabel('FDT Error', fontsize=12)
    ax1.set_title('τ₂ Component: Diagonal Error Scaling', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: τ₂ Error / dt
    ax2.semilogx(dt_vals, diag_errors_tau2 / dt_vals, 'ro-', linewidth=2, markersize=8, label='τ₂ Diagonal / dt')
    ax2.set_xlabel('Time step dt', fontsize=12)
    ax2.set_ylabel('Scaled Error', fontsize=12)
    ax2.set_title('τ₂: Diagonal / dt (should be constant if O(dt))', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: τ₃ Error vs dt (log-log)
    ax3.loglog(dt_vals, diag_errors_tau3, 'bo-', linewidth=2, markersize=8, label='τ₃ Diagonal')
    if len(dt_vals) >= 2:
        ref_dt = diag_errors_tau3[0] * (dt_vals / dt_vals[0])
        ax3.loglog(dt_vals, ref_dt, 'b--', alpha=0.5, linewidth=1.5, label='O(dt) reference')
    ax3.set_xlabel('Time step dt', fontsize=12)
    ax3.set_ylabel('FDT Error', fontsize=12)
    ax3.set_title('τ₃ Component: Diagonal Error Scaling', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: τ₃ Error / dt
    ax4.semilogx(dt_vals, diag_errors_tau3 / dt_vals, 'bo-', linewidth=2, markersize=8, label='τ₃ Diagonal / dt')
    ax4.set_xlabel('Time step dt', fontsize=12)
    ax4.set_ylabel('Scaled Error', fontsize=12)
    ax4.set_title('τ₃: Diagonal / dt (should be constant if O(dt))', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'Test_plots/diagonal_fdt_scaling.png'
    os.makedirs('Test_plots', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    results = analyze_diagonal_error(N_t_values=[501, 751, 1001, 1251, 1501])
