"""
Static test suite for equilibrium states in Keldysh formalism.

This module provides comprehensive testing for equilibrium/static states:
- Normalization checks (g^R and g^K)
- FDT relation verification
- Current conservation (should be zero for zero vector potential)
- Time translation invariance

All tests save plots to Test_plots/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution
from Test_scripts.general_comparison_file import load_state, compare_tensor_rows


def test_normalization(state, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, time_grid, save_dir='Test_plots', verbose=True):
    """
    Test g^R and g^K normalization relations.

    Calls StateObject methods to check normalization constraints and plots results.

    Args:
        state: StateObject with Green's functions
        thermal_dist: Thermal distribution f(t,t')
        thermal_integral: Integral of thermal distribution F(t,t')
        thermal_sum_left: Pre-computed left thermal sum
        thermal_sum_right: Pre-computed right thermal sum
        time_grid: Array of time values
        save_dir: Directory to save plots
        verbose: Print detailed output

    Returns:
        dict: {
            'gr_max_error': float,
            'gk_max_error': float,
            'gr_totals': array,
            'gk_totals': array
        }
    """
    if verbose:
        print("\n" + "="*70)
        print("Normalization Tests")
        print("="*70)

    # g^R normalization check
    if verbose:
        print("\n  Testing g^R normalization...")
    gr_errors, gr_totals = state.check_gr_normalization(t1_idx=-1)
    gr_max_error = np.max(gr_errors)

    if verbose:
        print(f"    Max g^R normalization error: {gr_max_error:.6e}")

    # g^K normalization check
    if verbose:
        print("\n  Testing g^K normalization...")
    gk_errors, gk_totals, gk_components = state.check_keldysh_normalization(-1, thermal_dist, thermal_integral, thermal_sum_left=thermal_sum_left, thermal_sum_right=thermal_sum_right)
    gk_max_error = np.max(gk_errors)

    if verbose:
        print(f"    Max g^K normalization error: {gk_max_error:.6e}")

    # Plot g^R normalization
    pauli_names = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx in range(4):
        ax = axes.flat[pauli_idx]
        component = gr_totals[pauli_idx, :]

        # Plot real and imaginary parts
        ax.plot(time_grid, np.real(component), 'b-', linewidth=2, label='Real part', alpha=0.7)
        ax.plot(time_grid, np.imag(component), 'r-', linewidth=2, label='Imaginary part', alpha=0.7)

        ax.set_xlabel("t₂ (time)", fontsize=10)
        ax.set_ylabel(f'Normalization violation - {pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'g^R Normalization: {pauli_names[pauli_idx]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.tight_layout()
    plt.suptitle('g^R Normalization Check (Last Row)', fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.97)

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    output_path_gr = os.path.join(save_dir, 'normalization_gr.png')
    plt.savefig(output_path_gr, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Plot saved to: {output_path_gr}")

    # Plot g^K normalization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx in range(4):
        ax = axes.flat[pauli_idx]
        component = gk_totals[pauli_idx, :]

        # Plot real and imaginary parts
        ax.plot(time_grid, np.real(component), 'b-', linewidth=2, label='Real part', alpha=0.7)
        ax.plot(time_grid, np.imag(component), 'r-', linewidth=2, label='Imaginary part', alpha=0.7)

        ax.set_xlabel("t₂ (time)", fontsize=10)
        ax.set_ylabel(f'Normalization violation - {pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'g^K Normalization: {pauli_names[pauli_idx]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.tight_layout()
    plt.suptitle('g^K Normalization Check (Last Row)', fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.97)

    # Save plot
    output_path_gk = os.path.join(save_dir, 'normalization_gk.png')
    plt.savefig(output_path_gk, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Plot saved to: {output_path_gk}")

    return {
        'gr_max_error': gr_max_error,
        'gk_max_error': gk_max_error,
        'gr_totals': gr_totals,
        'gk_totals': gk_totals
    }


def test_fdt(state, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, time_grid, save_dir='Test_plots', time_index=-1, verbose=True):
    """
    Test FDT relation: g^K = g^R @ f - f @ g^A.

    Args:
        state: StateObject with Green's functions
        thermal_dist: Thermal distribution f(t,t')
        thermal_integral: Integral of thermal distribution F(t,t')
        thermal_sum_left: Pre-computed left thermal sum
        thermal_sum_right: Pre-computed right thermal sum
        time_grid: Array of time values
        save_dir: Directory to save plots
        time_index: Time index to check (default -1 for last row)
        verbose: Print detailed output

    Returns:
        dict: {
            'max_error': float,
            'gk_actual': NambuKeldyshTensor,
            'gk_fdt': NambuKeldyshTensor,
            'error': NambuKeldyshTensor
        }
    """
    if verbose:
        print("\n" + "="*70)
        print("FDT Relation Test")
        print("="*70)

    # Compute FDT check
    if verbose:
        print("\n  Computing FDT relation check...")
    gk_fdt_row, gk_actual_row, error_row, max_error = state.check_fdt(
        thermal_dist, thermal_integral, time_index=time_index, thermal_sum_left=thermal_sum_left, thermal_sum_right=thermal_sum_right
    )

    if verbose:
        print(f"    Maximum absolute error: {max_error:.6e}")

    # Plot comparison
    pauli_names = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx in range(4):
        # Extract Pauli components
        gk_actual_pauli = gk_actual_row.trace(pauli_idx) / 2
        gk_fdt_pauli = gk_fdt_row.trace(pauli_idx) / 2
        error_pauli = error_row.trace(pauli_idx) / 2

        # Flatten to 1D arrays
        if gk_actual_pauli.ndim == 2:
            gk_actual_pauli = gk_actual_pauli[0, :]
            gk_fdt_pauli = gk_fdt_pauli[0, :]
            error_pauli = error_pauli[0, :]

        ax = axes.flat[pauli_idx]

        # Plot actual g^K (solid lines)
        ax.plot(time_grid, np.real(gk_actual_pauli), 'b-', linewidth=2, label='Actual (Real)', alpha=0.8)
        ax.plot(time_grid, np.imag(gk_actual_pauli), 'r-', linewidth=2, label='Actual (Imag)', alpha=0.8)

        # Plot FDT prediction (dashed lines)
        ax.plot(time_grid, np.real(gk_fdt_pauli), 'b--', linewidth=2, label='FDT (Real)', alpha=0.6)
        ax.plot(time_grid, np.imag(gk_fdt_pauli), 'r--', linewidth=2, label='FDT (Imag)', alpha=0.6)

        ax.set_xlabel('Time t\'', fontsize=10)
        ax.set_ylabel(f'g^K - {pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_names[pauli_idx]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

        # Compute error info
        max_err = np.max(np.abs(error_pauli))
        max_val = np.max(np.abs(gk_actual_pauli))
        rel_err = max_err / (max_val + 1e-10) * 100

        # Add error info as text
        ax.text(0.02, 0.98, f'max err: {max_err:.2e}\nrel: {rel_err:.1f}%',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.suptitle(f'FDT Relation Check (Row at t={time_grid[time_index]:.3f})',
                 fontsize=14, fontweight='bold', y=0.998)
    plt.subplots_adjust(top=0.97)

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, 'fdt_check.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Plot saved to: {output_path}")

    return {
        'max_error': max_error,
        'gk_actual': gk_actual_row,
        'gk_fdt': gk_fdt_row,
        'error': error_row
    }


def test_current(state, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, time_grid, save_dir='Test_plots', A_history=None, verbose=True):
    """
    Test current conservation for zero vector potential.

    Current should be zero in equilibrium with no external field.

    Args:
        state: StateObject with Green's functions
        thermal_dist: Thermal distribution f(t,t')
        thermal_integral: Integral of thermal distribution F(t,t')
        thermal_sum_left: Pre-computed left thermal sum
        thermal_sum_right: Pre-computed right thermal sum
        time_grid: Array of time values
        save_dir: Directory to save plots
        A_history: Optional vector potential history (default None = zeros)
        verbose: Print detailed output

    Returns:
        dict: {
            'current': complex,
            'passed': bool,
            'threshold': float
        }
    """
    if verbose:
        print("\n" + "="*70)
        print("Current Conservation Test")
        print("="*70)

    # Set A_history to zeros if not provided
    if A_history is None:
        A_history = np.zeros(len(time_grid), dtype=complex)
        if verbose:
            print("\n  Testing with zero vector potential...")

    # Compute current
    if verbose:
        print("  Computing current...")

    try:
        current = state.get_current_at_time_t(A_history, thermal_dist, thermal_integral, time_index=-1, thermal_sum_left=thermal_sum_left, thermal_sum_right=thermal_sum_right)
    except (IndexError, TypeError) as e:
        if verbose:
            print(f"    ⚠ Warning: Current calculation failed with error: {e}")
            print(f"    This indicates a bug in state_object_class.py")
            print(f"    Skipping current test...")

        # Return failure result
        return {
            'current': np.nan,
            'passed': False,
            'threshold': 1e-6,
            'error': str(e)
        }

    # Check threshold
    threshold = 1e-6
    current_magnitude = np.abs(current)
    passed = current_magnitude < threshold

    if verbose:
        print(f"    Current: {current:.6e}")
        print(f"    Magnitude: {current_magnitude:.6e}")
        print(f"    Threshold: {threshold:.6e}")
        print(f"    Status: {'✓ PASSED' if passed else '✗ FAILED'}")

    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Bar plot for real and imaginary parts
    components = ['Real', 'Imaginary', 'Magnitude']
    values = [np.real(current), np.imag(current), current_magnitude]
    colors = ['blue', 'red', 'green']

    bars = ax.bar(components, np.abs(values), color=colors, alpha=0.7)

    # Add threshold line
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0e})')

    ax.set_ylabel('Current (absolute value)', fontsize=12)
    ax.set_title('Current Test (Zero Vector Potential)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add pass/fail text
    status_text = 'PASSED' if passed else 'FAILED'
    status_color = 'green' if passed else 'red'
    ax.text(0.5, 0.95, f'Status: {status_text}', transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=status_color,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, 'current_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Plot saved to: {output_path}")

    return {
        'current': current,
        'passed': passed,
        'threshold': threshold
    }


def test_time_translation_invariance(state, time_grid, save_dir='Test_plots', num_rows=10, verbose=True):
    """
    Test time translation invariance for equilibrium state.

    Checks that g(t_i, t_j) ≈ g(t_{i-1}, t_{j-1}) for the last num_rows rows,
    verifying that equilibrium state only depends on τ = t - t'.

    Uses shift operator to vectorize comparison. Ignores first 10 elements
    due to shift zeroing.

    Args:
        state: StateObject with Green's functions
        time_grid: Array of time values
        save_dir: Directory to save plots
        num_rows: Number of rows to check (default 10)
        verbose: Print detailed output

    Returns:
        dict: {
            'max_diff_gr': float,
            'max_diff_gk': float,
            'passed': bool,
            'threshold': float
        }
    """
    if verbose:
        print("\n" + "="*70)
        print("Time Translation Invariance Test")
        print("="*70)
        print(f"\n  Testing last {num_rows} rows...")

    threshold = 1e-8
    max_diffs_gr = []
    max_diffs_gk = []

    # Test g^R
    if verbose:
        print("\n  Testing g^R time translation invariance...")

    for i in range(1, num_rows + 1):
        # Extract consecutive rows
        row_i = state.gr[-i, :]
        row_prev = state.gr[-(i+1), :]

        # Shift previous row
        row_prev_shifted = row_prev.shift(-1, axis=0)

        # Compute max difference across all Pauli components
        # Ignore first 10 elements due to shift zeroing
        max_diff = 0.0
        for pauli_idx in range(4):
            comp_i = row_i.trace(pauli_idx) / 2
            comp_prev = row_prev_shifted.trace(pauli_idx) / 2

            # Handle shape
            if comp_i.ndim == 2:
                comp_i = comp_i[0, :]
                comp_prev = comp_prev[0, :]
            elif comp_i.ndim == 0:
                # Scalar - make it 1D
                comp_i = np.array([comp_i])
                comp_prev = np.array([comp_prev])

            # Ignore first 10 elements (shift boundary effects)
            if len(comp_i) > 10:
                comp_i = comp_i[10:]
                comp_prev = comp_prev[10:]

            diff = np.max(np.abs(comp_i - comp_prev))
            max_diff = max(max_diff, diff)

        max_diffs_gr.append(max_diff)

    if verbose:
        print(f"    Max g^R difference: {np.max(max_diffs_gr):.6e}")

    # Test g^K
    if verbose:
        print("\n  Testing g^K time translation invariance...")

    for i in range(1, num_rows + 1):
        # Extract consecutive rows
        row_i = state.gk[-i, :]
        row_prev = state.gk[-(i+1), :]

        # Shift previous row
        row_prev_shifted = row_prev.shift(-1, axis=0)

        # Compute max difference across all Pauli components
        # Ignore first 10 elements due to shift zeroing
        max_diff = 0.0
        for pauli_idx in range(4):
            comp_i = row_i.trace(pauli_idx) / 2
            comp_prev = row_prev_shifted.trace(pauli_idx) / 2

            # Handle shape
            if comp_i.ndim == 2:
                comp_i = comp_i[0, :]
                comp_prev = comp_prev[0, :]
            elif comp_i.ndim == 0:
                # Scalar - make it 1D
                comp_i = np.array([comp_i])
                comp_prev = np.array([comp_prev])

            # Ignore first 10 elements (shift boundary effects)
            if len(comp_i) > 10:
                comp_i = comp_i[10:]
                comp_prev = comp_prev[10:]

            diff = np.max(np.abs(comp_i - comp_prev))
            max_diff = max(max_diff, diff)

        max_diffs_gk.append(max_diff)

    if verbose:
        print(f"    Max g^K difference: {np.max(max_diffs_gk):.6e}")

    # Check if passed
    max_diff_gr = np.max(max_diffs_gr)
    max_diff_gk = np.max(max_diffs_gk)
    passed = (max_diff_gr < threshold) and (max_diff_gk < threshold)

    if verbose:
        print(f"\n  Overall max difference: {max(max_diff_gr, max_diff_gk):.6e}")
        print(f"  Threshold: {threshold:.6e}")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    row_indices = np.arange(1, num_rows + 1)

    # Plot g^R differences
    ax1.plot(row_indices, max_diffs_gr, 'bo-', linewidth=2, markersize=8, label='Max difference')
    ax1.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0e})')
    ax1.set_xlabel('Row index from last', fontsize=12)
    ax1.set_ylabel('Max difference', fontsize=12)
    ax1.set_title('g^R Time Translation Invariance', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot g^K differences
    ax2.plot(row_indices, max_diffs_gk, 'ro-', linewidth=2, markersize=8, label='Max difference')
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0e})')
    ax2.set_xlabel('Row index from last', fontsize=12)
    ax2.set_ylabel('Max difference', fontsize=12)
    ax2.set_title('g^K Time Translation Invariance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.suptitle(f'Time Translation Invariance Test (Last {num_rows} Rows)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.94)

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, 'time_translation_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Plot saved to: {output_path}")

    return {
        'max_diff_gr': max_diff_gr,
        'max_diff_gk': max_diff_gk,
        'passed': passed,
        'threshold': threshold
    }


def test_state(filename, save_dir='Test_plots', verbose=True):
    """
    Run comprehensive test suite on equilibrium state.

    Loads state from pickle file (with auto-generation fallback) and runs:
    1. Normalization tests (g^R and g^K)
    2. FDT relation check
    3. Current conservation test
    4. Time translation invariance test

    All plots saved to save_dir (default: Test_plots/).

    Args:
        filename: Path to state pickle file
        save_dir: Directory to save plots (default 'Test_plots')
        verbose: Print detailed output

    Returns:
        dict: Results from all tests
    """
    print("="*70)
    print("STATIC STATE TEST SUITE")
    print("="*70)

    # Load state
    print(f"\nLoading state from: {filename}")
    state = load_state(filename)

    # Extract parameters from state
    # Use defaults if state doesn't have these attributes
    try:
        temperature = state.temperature if hasattr(state, 'temperature') else 0.3
        critical_temperature = state.critical_temperature if hasattr(state, 'critical_temperature') else 1.0
    except:
        temperature = 0.3
        critical_temperature = 1.0

    # Create grid parameters from state shape
    N_t = state.gr.data.shape[2]
    T_max = state.T_max if hasattr(state, 'T_max') else 2 * np.pi * 5
    dt = state.dt if hasattr(state, 'dt') else T_max / (N_t - 1)

    grid_parameters = {
        'time_sampling': N_t,
        'time_duration': T_max,
        'eta': 0.25
    }

    system_parameters = {
        'critical_temperature': critical_temperature,
        'temperature': temperature,
        'eta': 0.25
    }

    print(f"\nGrid parameters:")
    print(f"  Time points: {grid_parameters['time_sampling']}")
    print(f"  Time duration: {grid_parameters['time_duration']:.4f}")
    print(f"  dt: {dt:.6f}")

    print(f"\nSystem parameters:")
    print(f"  Critical temperature: {system_parameters['critical_temperature']}")
    print(f"  Temperature: {system_parameters['temperature']}")

    # Create evolution object for thermal distributions
    print("\nCreating evolution object for thermal distributions...")
    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)

    # Generate thermal distributions
    print("Generating thermal occupation and integral...")
    evolution.get_thermal_occupation(temperature)
    evolution.get_thermal_integral(temperature)
    evolution.get_thermal_sum(temperature)
    thermal_dist = evolution.thermal_dist
    thermal_integral = evolution.thermal_integral
    thermal_sum_left = evolution.thermal_sum_left
    thermal_sum_right = evolution.thermal_sum_right
    time_grid = evolution.time_grid

    print(f"  Thermal distribution shape: {thermal_dist.data.shape}")
    print(f"  Time grid: [{time_grid[0]:.4f}, {time_grid[-1]:.4f}]")

    # Run tests
    results = {}

    # 1. Normalization test
    results['normalization'] = test_normalization(
        state, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, time_grid, save_dir, verbose
    )

    # 2. FDT test
    results['fdt'] = test_fdt(
        state, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, time_grid, save_dir, time_index=-1, verbose=verbose
    )

    # 3. Current test
    results['current'] = test_current(
        state, thermal_dist, thermal_integral, thermal_sum_left, thermal_sum_right, time_grid, save_dir, A_history=None, verbose=verbose
    )

    # 4. Time translation invariance test
    results['time_translation'] = test_time_translation_invariance(
        state, time_grid, save_dir, num_rows=10, verbose=verbose
    )

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\n✓ Normalization:")
    print(f"    g^R max error: {results['normalization']['gr_max_error']:.6e}")
    print(f"    g^K max error: {results['normalization']['gk_max_error']:.6e}")

    print(f"\n✓ FDT:")
    print(f"    Max error: {results['fdt']['max_error']:.6e}")

    print(f"\n{'✓' if results['current']['passed'] else '✗'} Current:")
    print(f"    Current: {results['current']['current']:.6e}")
    print(f"    Passed: {results['current']['passed']}")

    print(f"\n{'✓' if results['time_translation']['passed'] else '✗'} Time Translation Invariance:")
    print(f"    g^R max diff: {results['time_translation']['max_diff_gr']:.6e}")
    print(f"    g^K max diff: {results['time_translation']['max_diff_gk']:.6e}")
    print(f"    Passed: {results['time_translation']['passed']}")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)

    return results


if __name__ == "__main__":
    """Run test suite on default state file."""
    # Default test on initial_state_test.pkl
    test_state('initial_state_test.pkl')
