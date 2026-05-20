"""
FDT Error Scaling Diagnostic

Tests how FDT errors scale with time step resolution.
Generates equilibrium states at different resolutions, evolves them,
and measures FDT errors in all Pauli channels.

Keeps T_max constant and varies the number of time points.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution


def compute_fdt_errors(state, thermal_dist, thermal_integral):
    """
    Compute maximum FDT error for all Pauli components.

    Args:
        state: StateObject with g^R and g^K
        thermal_dist: Thermal distribution f(t,t')
        thermal_integral: Thermal integral F(t,t')

    Returns:
        dict: Maximum absolute errors for each Pauli component
    """
    gk_fdt_row, gk_actual_row, error_row, max_error = state.check_fdt(
        thermal_dist,
        thermal_integral,
        time_index=-1
    )

    errors = {}
    for pauli_idx, pauli_name in enumerate(['tau0', 'tau1', 'tau2', 'tau3']):
        error_pauli = error_row.trace(pauli_idx) / 2
        errors[pauli_name] = np.max(np.abs(error_pauli))

    return errors


def run_single_resolution(time_sampling, T_max, system_parameters, num_evolution_steps=100):
    """
    Run complete diagnostic for a single time resolution.

    Args:
        time_sampling: Number of time points
        T_max: Maximum time duration (kept constant)
        system_parameters: System parameters dict
        num_evolution_steps: Number of real-time evolution steps

    Returns:
        dict: Results containing dt and errors
    """
    print(f"\n{'='*70}")
    print(f"Testing time_sampling = {time_sampling}")
    print(f"{'='*70}")

    grid_parameters = {
        'time_sampling': time_sampling,
        'time_duration': T_max,
        'eta': system_parameters['eta']
    }

    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
    dt = evolution.delta_t

    print(f"  dt = {dt:.6f}")
    print(f"  Time grid: [{evolution.time_grid[0]:.4f}, {evolution.time_grid[-1]:.4f}]")

    print("\n  Generating equilibrium state...")
    initial_state, _, _ = evolution.generate_initial_state(Q=0.0)
    gap_initial = initial_state.get_gap_history()[-1]
    print(f"    Equilibrium gap: Δ = {np.real(gap_initial):.6f}")

    print(f"\n  Evolving for {num_evolution_steps} timesteps...")
    evolved_state, gaps, currents = evolution.real_time_evolution(
        initial_state,
        num_timesteps=num_evolution_steps
    )
    gap_final = gaps[-1]
    print(f"    Final gap: Δ = {np.real(gap_final):.6f}")
    print(f"    Gap change: ΔΔ = {np.real(gap_final - gap_initial):.6f}")

    print("\n  Generating thermal distribution...")
    evolution.get_thermal_occupation(system_parameters['temperature'])
    thermal_dist = evolution.thermal_dist

    evolution.get_thermal_integral(system_parameters['temperature'])
    thermal_integral = evolution.thermal_integral

    print("\n  Computing FDT errors...")
    errors = compute_fdt_errors(evolved_state, thermal_dist, thermal_integral)

    print("    Maximum FDT errors:")
    for pauli_name, error_val in errors.items():
        print(f"      {pauli_name}: {error_val:.6e}")

    results = {
        'time_sampling': time_sampling,
        'dt': dt,
        'gap_initial': gap_initial,
        'gap_final': gap_final,
        'errors': errors
    }

    return results


def plot_scaling_results(results_list, filename='fdt_error_vs_dt.png'):
    """
    Plot FDT error scaling with dt for all Pauli components.

    Args:
        results_list: List of result dictionaries
        filename: Output filename
    """
    dt_values = [r['dt'] for r in results_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pauli_names = ['tau0', 'tau1', 'tau2', 'tau3']
    pauli_labels = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']

    for ax, pauli_name, pauli_label in zip(axes.flat, pauli_names, pauli_labels):
        errors = [r['errors'][pauli_name] for r in results_list]

        ax.loglog(dt_values, errors, 'bo-', linewidth=2, markersize=8, label='FDT Error')

        # Add reference lines for dt and dt^2 scaling
        dt_min, dt_max = min(dt_values), max(dt_values)
        dt_ref = np.logspace(np.log10(dt_min), np.log10(dt_max), 50)

        # Scale reference lines to match data
        if max(errors) > 0:
            ref_scale = max(errors) / max(dt_ref)
            ax.loglog(dt_ref, ref_scale * dt_ref, 'r--', alpha=0.5, linewidth=1.5, label='∝ dt')
            ax.loglog(dt_ref, ref_scale * dt_ref**2, 'g--', alpha=0.5, linewidth=1.5, label='∝ dt²')

        ax.set_xlabel('Time step dt', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Max |FDT Error| - {pauli_label}', fontsize=10)
        ax.set_title(f'{pauli_label} Error Scaling', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Add data point labels
        for dt_val, err in zip(dt_values, errors):
            if err > 0:
                ax.annotate(f'{int(1/dt_val)}', (dt_val, err),
                           textcoords="offset points", xytext=(0,5),
                           ha='center', fontsize=7, alpha=0.7)

    plt.suptitle('FDT Error Scaling with Time Resolution (T_max constant)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {filename}")
    plt.close()


def save_results(results_list, filename='scaling_results.pkl'):
    """Save results to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(results_list, f)
    print(f"Results saved: {filename}")


def main():
    print("="*70)
    print("FDT Error Scaling Diagnostic")
    print("="*70)
    print()

    # System parameters (kept constant)
    T_max = 2 * np.pi * 5
    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': 0.3,
        'eta': 0.2
    }

    # Time resolutions to test
    time_sampling_values = [1001, 1251, 1501, 1751, 2001]
    num_evolution_steps = 100

    print("Configuration:")
    print(f"  T_max: {T_max:.4f} (constant)")
    print(f"  Temperature: {system_parameters['temperature']}")
    print(f"  Critical temperature: {system_parameters['critical_temperature']}")
    print(f"  Evolution steps: {num_evolution_steps}")
    print(f"  Time resolutions: {time_sampling_values}")
    print()

    # Run diagnostics for all resolutions
    results_list = []

    for time_sampling in time_sampling_values:
        try:
            results = run_single_resolution(
                time_sampling,
                T_max,
                system_parameters,
                num_evolution_steps
            )
            results_list.append(results)
        except Exception as e:
            print(f"\n  ERROR: Failed for time_sampling={time_sampling}")
            print(f"  {type(e).__name__}: {e}")
            continue

    if len(results_list) == 0:
        print("\nNo successful runs. Diagnostic failed.")
        return

    # Plot results
    print("\n" + "="*70)
    print("Generating scaling plots...")
    print("="*70)
    plot_scaling_results(results_list)

    # Save results
    save_results(results_list)

    # Summary table
    print("\n" + "="*70)
    print("Summary Table")
    print("="*70)
    print(f"{'N_t':<8} {'dt':<12} {'τ₀':<12} {'τ₁':<12} {'τ₂':<12} {'τ₃':<12}")
    print("-" * 68)
    for r in results_list:
        print(f"{r['time_sampling']:<8} {r['dt']:<12.6f} "
              f"{r['errors']['tau0']:<12.6e} {r['errors']['tau1']:<12.6e} "
              f"{r['errors']['tau2']:<12.6e} {r['errors']['tau3']:<12.6e}")

    print("\n" + "="*70)
    print("Diagnostic Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
