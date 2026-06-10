"""
Equilibrium sweep script for superconducting gap and current vs temperature and vector potential.

This script computes equilibrium states and extracts gap and current observables.
"""

import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from usadel_keldysh_evolution import UsadelKeldyshEvolution


def compute_equilibrium_gap_vs_temperature(
    T_min=0.1,
    T_max=1.3,
    n_points=20,
    critical_temperature=1.0,
    eta=0.2,
    time_sampling=1501,
    time_duration=2 * np.pi * 5,
    save_results=True,
    save_dir='equilibrium_results'
):
    """
    Compute equilibrium gap as a function of temperature.

    Uses UsadelKeldyshEvolution to generate equilibrium states in time domain.

    Args:
        T_min: Minimum temperature
        T_max: Maximum temperature
        n_points: Number of temperature points
        critical_temperature: Critical temperature T_c
        eta: Damping parameter η
        time_sampling: Number of time grid points
        time_duration: Time duration T_max
        save_results: Whether to save results to file
        save_dir: Directory for saving results

    Returns:
        dict: {'temperatures': array, 'gaps': array, 'gap_errors': array}
    """
    print("\n" + "="*70)
    print("EQUILIBRIUM GAP vs TEMPERATURE SWEEP")
    print("="*70)

    # Temperature array
    temperatures = np.linspace(T_min, T_max, n_points)

    # Storage arrays
    gaps = []
    gap_errors = []

    # Grid parameters (time domain)
    grid_parameters = {
        'time_sampling': time_sampling,
        'time_duration': time_duration,
        'eta': eta
    }

    print(f"\nScan parameters:")
    print(f"  Temperature range: [{T_min:.2f}, {T_max:.2f}]")
    print(f"  Number of points: {n_points}")
    print(f"  Critical temperature: {critical_temperature:.2f}")
    print(f"  Damping η: {eta:.3f}")
    print(f"  Time sampling: {time_sampling}")
    print(f"  Time duration: {time_duration:.1f}")
    print()

    # Loop over temperatures
    for i, T in enumerate(temperatures):
        print(f"\n[{i+1}/{n_points}] Computing equilibrium at T = {T:.4f}")

        # System parameters
        system_parameters = {
            'critical_temperature': critical_temperature,
            'temperature': T,
            'eta': eta
        }

        try:
            # Create evolution object (as in code_testing.ipynb)
            evolution = UsadelKeldyshEvolution(
                grid_parameters=grid_parameters,
                system_parameters=system_parameters,
                optimization_parameters=None,
                sigma_scatterings=None
            )

            # Generate equilibrium state (returns state, gr_tau, gk_tau, current)
            initial_state, gr_tau, gk_tau, J_eq = evolution.generate_initial_state(Q=0.0)

            # Extract gap from state
            gap_history = initial_state.get_gap_history()
            gap_eq = gap_history[-1]  # Last time point (t=0)
            gap_magnitude = np.abs(gap_eq)

            # Estimate error from variation in last 10% of points
            n_tail = max(1, len(gap_history) // 10)
            gap_std = np.std(np.abs(gap_history[-n_tail:]))

            gaps.append(gap_eq)
            gap_errors.append(gap_std)

            print(f"  → Gap: {gap_magnitude:.6f} (Re: {np.real(gap_eq):.6f}, Im: {np.imag(gap_eq):.6f})")
            print(f"  → Error estimate: {gap_std:.6e}")

        except Exception as e:
            print(f"  ERROR: Failed to compute equilibrium at T={T:.4f}")
            print(f"  {str(e)}")
            gaps.append(np.nan)
            gap_errors.append(np.nan)

    # Convert to arrays
    gaps = np.array(gaps)
    gap_errors = np.array(gap_errors)

    # Results dictionary
    results = {
        'temperatures': temperatures,
        'gaps': gaps,
        'gap_errors': gap_errors,
        'parameters': {
            'T_min': T_min,
            'T_max': T_max,
            'n_points': n_points,
            'critical_temperature': critical_temperature,
            'eta': eta,
            'time_sampling': time_sampling,
            'time_duration': time_duration
        }
    }

    # Save results
    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as text file for easy copy-paste
        txt_filename = os.path.join(save_dir, f'gap_vs_T_sweep_{timestamp}.txt')

        with open(txt_filename, 'w') as f:
            f.write("# Equilibrium Gap vs Temperature\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# T_c = {critical_temperature}, eta = {eta}\n")
            f.write(f"# time_sampling = {time_sampling}, time_duration = {time_duration}\n")
            f.write("#\n")
            f.write("# Columns: Temperature  Re(Gap)  Im(Gap)  |Gap|  Error\n")
            f.write("#" + "-"*68 + "\n")

            for T, gap, err in zip(temperatures, gaps, gap_errors):
                if not np.isnan(gap):
                    f.write(f"{T:12.6f}  {np.real(gap):12.6e}  {np.imag(gap):12.6e}  {np.abs(gap):12.6e}  {err:12.6e}\n")
                else:
                    f.write(f"{T:12.6f}  {'NaN':>12s}  {'NaN':>12s}  {'NaN':>12s}  {'NaN':>12s}\n")

        # Also save pickle for compatibility
        pkl_filename = os.path.join(save_dir, f'gap_vs_T_sweep_{timestamp}.pkl')
        with open(pkl_filename, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n{'='*70}")
        print(f"Text file saved to: {txt_filename}")
        print(f"Pickle file saved to: {pkl_filename}")
        print(f"{'='*70}\n")

    return results


def compute_equilibrium_current_and_gap_vs_vector_potential(
    A_min=0.0,
    A_max=0.9,
    n_points=20,
    temperature=0.3,
    critical_temperature=1.0,
    eta=0.2,
    time_sampling=1501,
    time_duration=2 * np.pi * 5,
    save_results=True,
    save_dir='equilibrium_results'
):
    """
    Compute equilibrium gap and current as a function of vector potential.

    Uses UsadelKeldyshEvolution to generate equilibrium states in time domain.

    Args:
        A_min: Minimum vector potential
        A_max: Maximum vector potential
        n_points: Number of vector potential points
        temperature: Temperature for sweep
        critical_temperature: Critical temperature T_c
        eta: Damping parameter η
        time_sampling: Number of time grid points
        time_duration: Time duration T_max
        save_results: Whether to save results to file
        save_dir: Directory for saving results

    Returns:
        dict: {'vector_potentials': array, 'gaps': array, 'currents': array, 'gap_errors': array}
    """
    print("\n" + "="*70)
    print("EQUILIBRIUM GAP & CURRENT vs VECTOR POTENTIAL SWEEP")
    print("="*70)

    # Vector potential array
    vector_potentials = np.linspace(A_min, A_max, n_points)

    # Storage arrays
    gaps = []
    currents = []
    gap_errors = []

    # Grid parameters (time domain)
    grid_parameters = {
        'time_sampling': time_sampling,
        'time_duration': time_duration,
        'eta': eta
    }

    # System parameters
    system_parameters = {
        'critical_temperature': critical_temperature,
        'temperature': temperature,
        'eta': eta
    }

    print(f"\nScan parameters:")
    print(f"  Vector potential range: [{A_min:.2f}, {A_max:.2f}]")
    print(f"  Number of points: {n_points}")
    print(f"  Temperature: {temperature:.2f}")
    print(f"  Critical temperature: {critical_temperature:.2f}")
    print(f"  Damping η: {eta:.3f}")
    print(f"  Time sampling: {time_sampling}")
    print(f"  Time duration: {time_duration:.1f}")
    print()

    # Loop over vector potentials
    for i, A in enumerate(vector_potentials):
        print(f"\n[{i+1}/{n_points}] Computing equilibrium at A = {A:.4f}")

        try:
            # Create evolution object (as in code_testing.ipynb)
            evolution = UsadelKeldyshEvolution(
                grid_parameters=grid_parameters,
                system_parameters=system_parameters,
                optimization_parameters=None,
                sigma_scatterings=None
            )

            # Generate equilibrium state (returns state, gr_tau, gk_tau, current)
            initial_state, gr_tau, gk_tau, current_eq = evolution.generate_initial_state(Q=A)

            # Extract gap from state
            gap_history = initial_state.get_gap_history()
            gap_eq = gap_history[-1]  # Last time point (t=0)
            gap_magnitude = np.abs(gap_eq)

            # Estimate gap error from variation in last 10% of points
            n_tail = max(1, len(gap_history) // 10)
            gap_std = np.std(np.abs(gap_history[-n_tail:]))

            # Current magnitude
            current_magnitude = np.abs(current_eq)

            gaps.append(gap_eq)
            gap_errors.append(gap_std)
            currents.append(current_eq)

            print(f"  → Gap: {gap_magnitude:.6f} (Re: {np.real(gap_eq):.6f}, Im: {np.imag(gap_eq):.6f})")
            print(f"  → Current: {current_magnitude:.6f} (Re: {np.real(current_eq):.6f}, Im: {np.imag(current_eq):.6f})")
            print(f"  → Gap error: {gap_std:.6e}")

        except Exception as e:
            print(f"  ERROR: Failed to compute equilibrium at A={A:.4f}")
            print(f"  {str(e)}")
            gaps.append(np.nan)
            gap_errors.append(np.nan)
            currents.append(np.nan)

    # Convert to arrays
    gaps = np.array(gaps)
    gap_errors = np.array(gap_errors)
    currents = np.array(currents)

    # Results dictionary
    results = {
        'vector_potentials': vector_potentials,
        'gaps': gaps,
        'currents': currents,
        'gap_errors': gap_errors,
        'parameters': {
            'A_min': A_min,
            'A_max': A_max,
            'n_points': n_points,
            'temperature': temperature,
            'critical_temperature': critical_temperature,
            'eta': eta,
            'time_sampling': time_sampling,
            'time_duration': time_duration
        }
    }

    # Save results
    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as text file for easy copy-paste
        txt_filename = os.path.join(save_dir, f'gap_current_vs_A_sweep_{timestamp}.txt')

        with open(txt_filename, 'w') as f:
            f.write("# Equilibrium Gap and Current vs Vector Potential\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# T = {temperature}, T_c = {critical_temperature}, eta = {eta}\n")
            f.write(f"# time_sampling = {time_sampling}, time_duration = {time_duration}\n")
            f.write("#\n")
            f.write("# Columns: A  Re(Gap)  Im(Gap)  |Gap|  Re(Current)  Im(Current)  |Current|  Gap_Error\n")
            f.write("#" + "-"*100 + "\n")

            for A, gap, current, err in zip(vector_potentials, gaps, currents, gap_errors):
                if not np.isnan(gap):
                    f.write(f"{A:12.6f}  {np.real(gap):12.6e}  {np.imag(gap):12.6e}  {np.abs(gap):12.6e}  "
                           f"{np.real(current):12.6e}  {np.imag(current):12.6e}  {np.abs(current):12.6e}  {err:12.6e}\n")
                else:
                    f.write(f"{A:12.6f}  {'NaN':>12s}  {'NaN':>12s}  {'NaN':>12s}  "
                           f"{'NaN':>12s}  {'NaN':>12s}  {'NaN':>12s}  {'NaN':>12s}\n")

        # Also save pickle for compatibility
        pkl_filename = os.path.join(save_dir, f'gap_current_vs_A_sweep_{timestamp}.pkl')
        with open(pkl_filename, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n{'='*70}")
        print(f"Text file saved to: {txt_filename}")
        print(f"Pickle file saved to: {pkl_filename}")
        print(f"{'='*70}\n")

    return results


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("EQUILIBRIUM SWEEPS FOR SUPERCONDUCTING VORTICES")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # Common parameters
    critical_temperature = 1.0
    eta = 0.2
    time_sampling = 1501
    time_duration = 2 * np.pi * 30
    save_dir = 'equilibrium_results'

    # ========== Temperature Sweep (Gap) ==========
    print("\n" + "#"*70)
    print("# SWEEP 1: Gap vs Temperature")
    print("#"*70)

    temp_results = compute_equilibrium_gap_vs_temperature(
        T_min=0.1,
        T_max=1.1,
        n_points=30,
        critical_temperature=critical_temperature,
        eta=eta,
        time_sampling=time_sampling,
        time_duration=time_duration,
        save_results=True,
        save_dir=save_dir
    )

    # ========== Vector Potential Sweep (Gap & Current) ==========
    print("\n" + "#"*70)
    print("# SWEEP 2: Gap & Current vs Vector Potential")
    print("#"*70)

    A_results = compute_equilibrium_current_and_gap_vs_vector_potential(
        A_min=0.0,
        A_max=0.9,
        n_points=20,
        temperature=0.3,
        critical_temperature=critical_temperature,
        eta=eta,
        time_sampling=time_sampling,
        time_duration=time_duration,
        save_results=True,
        save_dir=save_dir
    )

    # ========== Summary ==========
    print("\n" + "="*70)
    print("SWEEP SUMMARY")
    print("="*70)

    print("\nTemperature Sweep (Gap):")
    print(f"  Temperature range: [{temp_results['parameters']['T_min']:.2f}, {temp_results['parameters']['T_max']:.2f}]")
    print(f"  Points computed: {len(temp_results['temperatures'])}")
    valid_gaps = ~np.isnan(temp_results['gaps'])
    print(f"  Successful: {np.sum(valid_gaps)}/{len(temp_results['temperatures'])}")
    if np.sum(valid_gaps) > 0:
        print(f"  Max gap: {np.max(np.abs(temp_results['gaps'][valid_gaps])):.6f}")
        max_gap_idx = np.argmax(np.abs(temp_results['gaps'][valid_gaps]))
        print(f"    at T = {temp_results['temperatures'][valid_gaps][max_gap_idx]:.4f}")

    print("\nVector Potential Sweep (Gap & Current):")
    print(f"  Vector potential range: [{A_results['parameters']['A_min']:.2f}, {A_results['parameters']['A_max']:.2f}]")
    print(f"  Temperature: {A_results['parameters']['temperature']:.2f}")
    print(f"  Points computed: {len(A_results['vector_potentials'])}")
    valid_data = ~np.isnan(A_results['gaps'])
    print(f"  Successful: {np.sum(valid_data)}/{len(A_results['vector_potentials'])}")
    if np.sum(valid_data) > 0:
        print(f"  Max gap: {np.max(np.abs(A_results['gaps'][valid_data])):.6f}")
        print(f"  Max current: {np.max(np.abs(A_results['currents'][valid_data])):.6f}")

    print("\n" + "="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    print("Results saved in directory:", save_dir)
    print("Use data_analysis.py to plot the results.")
