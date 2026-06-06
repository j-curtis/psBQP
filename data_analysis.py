import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from demler_tools.file_manager import io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution


# ============================================================================
# Data Loading and Parameter Extraction
# ============================================================================

def load_job_data(timestamp, job_index):
    """
    Load simulation results using demler_tools.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index (run_index)

    Returns:
        tuple: (input_kwargs, save_data)
            input_kwargs: Dict with system_parameters, grid_parameters, etc.
            save_data: Dict with final_state, gaps, currents, vector_potentials, times
    """
    input_kwargs, save_data = io.get_results(timestamp=timestamp, run_index=job_index)
    return input_kwargs, save_data


def _get_system_parameters(input_kwargs):
    """
    Extract system parameters from input_kwargs.

    Args:
        input_kwargs: Input parameters dict from io.get_results()

    Returns:
        dict: system_parameters with critical_temperature, temperature, eta

    Raises:
        KeyError: If required parameters are missing
    """
    return input_kwargs['system_parameters']


def _get_grid_parameters(save_data, input_kwargs):
    """
    Extract grid parameters from state.

    Args:
        save_data: Saved data dict containing final_state
        input_kwargs: Input parameters dict (for eta fallback)

    Returns:
        dict: grid_parameters with time_sampling, time_duration, eta

    Raises:
        AttributeError: If state lacks required attributes
    """
    state = save_data['final_state']
    eta = state.eta if hasattr(state, 'eta') else _get_system_parameters(input_kwargs)['eta']
    return {'time_sampling': state.gr.data.shape[2], 'time_duration': state.T_max, 'eta': eta}


# ============================================================================
# Normalization and FDT Checks
# ============================================================================

def check_normalizations(timestamp, job_index, save_plot=False, save_dir='analysis_plots'):
    """
    Check normalization relations for saved state.

    Args:
        timestamp: Timestamp to analyze
        job_index: Job index
        save_plot: Whether to save plots (default False)
        save_dir: Directory for plots

    Returns:
        dict: {'gr_max_error': float, 'gk_max_error': float, 'gr_totals': array, 'gk_totals': array}
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index)
    state = save_data['final_state']
    system_params = _get_system_parameters(input_kwargs)
    grid_params = _get_grid_parameters(save_data, input_kwargs)

    evolution = UsadelKeldyshEvolution(grid_params, system_params)
    evolution.get_thermal_occupation(system_params['temperature'])
    evolution.get_thermal_integral(system_params['temperature'])

    gr_errors, gr_totals = state.check_gr_normalization(t1_idx=-1)
    gk_errors, gk_totals, _ = state.check_keldysh_normalization(-1, evolution.thermal_dist, evolution.thermal_integral)

    if save_plot:
        _plot_normalization(gr_totals, 'gr', evolution.time_grid, save_dir, timestamp)
        _plot_normalization(gk_totals, 'gk', evolution.time_grid, save_dir, timestamp)

    return {'gr_max_error': np.max(gr_errors), 'gk_max_error': np.max(gk_errors), 'gr_totals': gr_totals, 'gk_totals': gk_totals}


def check_fdt(timestamp, job_index, save_plot=False, save_dir='analysis_plots'):
    """
    Check FDT relation for saved state.

    Args:
        timestamp: Timestamp to analyze
        job_index: Job index
        save_plot: Whether to save plots (default False)
        save_dir: Directory for plots

    Returns:
        dict: {'max_error': float, 'gk_actual': NambuKeldyshTensor, 'gk_fdt': NambuKeldyshTensor, 'error': NambuKeldyshTensor}
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index)
    state = save_data['final_state']
    system_params = _get_system_parameters(input_kwargs)
    grid_params = _get_grid_parameters(save_data, input_kwargs)

    evolution = UsadelKeldyshEvolution(grid_params, system_params)
    evolution.get_thermal_occupation(system_params['temperature'])
    evolution.get_thermal_integral(system_params['temperature'])

    gk_fdt_row, gk_actual_row, error_row, max_error = state.check_fdt(evolution.thermal_dist, evolution.thermal_integral, time_index=-1)

    if save_plot:
        _plot_fdt(gk_actual_row, gk_fdt_row, error_row, evolution.time_grid, save_dir, timestamp)

    return {'max_error': max_error, 'gk_actual': gk_actual_row, 'gk_fdt': gk_fdt_row, 'error': error_row}


def check_time_translational_invariance(timestamp, job_index, num_rows=10, threshold=1e-8, save_plot=False, save_dir='analysis_plots'):
    """
    Check time-translation invariance for equilibrium state.

    Args:
        timestamp: Timestamp to analyze
        job_index: Job index
        num_rows: Number of rows to check (default 10)
        threshold: Threshold for passing test (default 1e-8)
        save_plot: Whether to save plots (default False)
        save_dir: Directory for plots

    Returns:
        dict: {'max_diff_gr': float, 'max_diff_gk': float, 'passed': bool, 'threshold': float}
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index)
    state = save_data['final_state']
    time_grid = np.linspace(-state.T_max, 0, state.gr.data.shape[2])

    max_diffs_gr = _compute_time_translation_diffs(state.gr, num_rows)
    max_diffs_gk = _compute_time_translation_diffs(state.gk, num_rows)

    max_diff_gr, max_diff_gk = np.max(max_diffs_gr), np.max(max_diffs_gk)
    passed = (max_diff_gr < threshold) and (max_diff_gk < threshold)

    if save_plot:
        _plot_time_translation(max_diffs_gr, max_diffs_gk, num_rows, threshold, save_dir, timestamp)

    return {'max_diff_gr': max_diff_gr, 'max_diff_gk': max_diff_gk, 'passed': passed, 'threshold': threshold}


def _compute_time_translation_diffs(tensor, num_rows):
    """Compute time-translation invariance differences for a tensor."""
    max_diffs = []
    for i in range(1, num_rows + 1):
        row_i = tensor[-i, :]
        row_prev = tensor[-(i+1), :]
        row_prev_shifted = row_prev.shift(-1, axis=0)

        max_diff = 0.0
        for pauli_idx in range(4):
            comp_i = row_i.trace(pauli_idx) / 2
            comp_prev = row_prev_shifted.trace(pauli_idx) / 2
            if comp_i.ndim == 2:
                comp_i = comp_i[0, :]
                comp_prev = comp_prev[0, :]
            if len(comp_i) > 10:
                comp_i = comp_i[10:]
                comp_prev = comp_prev[10:]
            diff = np.max(np.abs(comp_i - comp_prev))
            max_diff = max(max_diff, diff)
        max_diffs.append(max_diff)
    return max_diffs


# ============================================================================
# Tensor Comparison Functions (from general_comparison_file.py)
# ============================================================================

def compare_tensor_rows(tensor1, tensor2, row_index=-1):
    """
    Compare rows of two NambuKeldyshTensor objects.

    Args:
        tensor1: First NambuKeldyshTensor to compare
        tensor2: Second NambuKeldyshTensor to compare
        row_index: Row index to extract (default -1 for last row)

    Returns:
        dict: Statistics for each Pauli component with structure:
              {
                  'pauli_0': {'real': {...}, 'imag': {...}},
                  'pauli_1': {'real': {...}, 'imag': {...}},
                  'pauli_2': {'real': {...}, 'imag': {...}},
                  'pauli_3': {'real': {...}, 'imag': {...}}
              }
              where each {...} contains:
                  - max_diff: Maximum absolute difference
                  - max_diff_index: Index of maximum difference
                  - mean_diff: Mean absolute difference
                  - rms_diff: RMS difference
    """
    row1 = tensor1[row_index, :]
    row2 = tensor2[row_index, :]

    results = {}
    pauli_names = ['pauli_0', 'pauli_1', 'pauli_2', 'pauli_3']

    for pauli_idx in range(4):
        comp1 = row1.trace(pauli_index=pauli_idx) / 2
        comp2 = row2.trace(pauli_index=pauli_idx) / 2

        if comp1.ndim == 2:
            comp1 = comp1[0, :]
            comp2 = comp2[0, :]

        real_diff = np.abs(np.real(comp1) - np.real(comp2))
        max_diff_idx_real = np.argmax(real_diff)
        real_stats = {'max_diff': real_diff[max_diff_idx_real], 'max_diff_index': max_diff_idx_real, 'mean_diff': np.mean(real_diff), 'rms_diff': np.sqrt(np.mean(real_diff**2))}

        imag_diff = np.abs(np.imag(comp1) - np.imag(comp2))
        max_diff_idx_imag = np.argmax(imag_diff)
        imag_stats = {'max_diff': imag_diff[max_diff_idx_imag], 'max_diff_index': max_diff_idx_imag, 'mean_diff': np.mean(imag_diff), 'rms_diff': np.sqrt(np.mean(imag_diff**2))}

        results[pauli_names[pauli_idx]] = {'real': real_stats, 'imag': imag_stats}

    return results


def plot_tensor_comparison(tensor1, tensor2, x_values, title, row_index=-1, save_plot=False, save_dir='analysis_plots'):
    """
    Plot comparison of two NambuKeldyshTensor objects.

    Creates a 2x2 grid showing all 4 Pauli components with deviation markers.

    Args:
        tensor1: First NambuKeldyshTensor to plot
        tensor2: Second NambuKeldyshTensor to plot
        x_values: Array of x-values for plotting
        title: Plot title
        row_index: Row index to extract (default -1 for last row)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
    """
    row1 = tensor1[row_index, :]
    row2 = tensor2[row_index, :]

    pauli_labels = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx in range(4):
        comp1 = row1.trace(pauli_index=pauli_idx) / 2
        comp2 = row2.trace(pauli_index=pauli_idx) / 2

        if comp1.ndim == 2:
            comp1 = comp1[0, :]
            comp2 = comp2[0, :]

        ax = axes.flat[pauli_idx]

        ax.plot(x_values, np.real(comp1), 'b-', linewidth=2, label='Tensor 1 (Real)', alpha=0.8)
        ax.plot(x_values, np.imag(comp1), 'b--', linewidth=2, label='Tensor 1 (Imag)', alpha=0.8)
        ax.plot(x_values, np.real(comp2), 'r-', linewidth=2, label='Tensor 2 (Real)', alpha=0.8)
        ax.plot(x_values, np.imag(comp2), 'r--', linewidth=2, label='Tensor 2 (Imag)', alpha=0.8)

        real_diff = np.abs(np.real(comp1) - np.real(comp2))
        imag_diff = np.abs(np.imag(comp1) - np.imag(comp2))
        max_real_idx = np.argmax(real_diff)
        max_imag_idx = np.argmax(imag_diff)
        max_idx = max_real_idx if real_diff[max_real_idx] > imag_diff[max_imag_idx] else max_imag_idx
        max_x = x_values[max_idx]

        ax.axvline(max_x, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Max deviation')

        ax.set_xlabel('Time t\'', fontsize=10)
        ax.set_ylabel(f'{pauli_labels[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_labels[pauli_idx]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([x_values[0], x_values[-1]])

    plt.tight_layout()
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.97)

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = title.replace(' ', '_').lower()
        plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# Gap and Current Plotting Functions
# ============================================================================

def plot_gap(timestamp, job_index, save_plot=False, save_dir='analysis_plots'):
    """
    Plot gap vs time for a specific job.

    Displays input parameters before plotting.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index)
    gaps = save_data['gaps']

    if 'times' in save_data:
        times = save_data['times']
    else:
        state = save_data['final_state']
        N_t = len(gaps)
        dt = state.dt
        times = np.linspace(0, N_t * dt, N_t)

    print(f"\n{'='*60}")
    print(f"Input Parameters (timestamp={timestamp}, job_index={job_index}):")
    print(f"{'='*60}")
    for key, value in input_kwargs.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, np.real(gaps), 'b-', linewidth=2, label='Real(Δ)')
    ax.plot(times, np.imag(gaps), 'r-', linewidth=2, label='Imag(Δ)')
    ax.plot(times, np.abs(gaps), 'k--', linewidth=1.5, label='|Δ|', alpha=0.5)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Gap Δ(t)', fontsize=12)
    ax.set_title(f'Gap vs Time (job {job_index})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'gap_vs_time_{timestamp}_job{job_index}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_current(timestamp, job_index, save_plot=False, save_dir='analysis_plots'):
    """
    Plot current vs time for a specific job.

    Displays input parameters before plotting.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index)
    currents = save_data['currents']

    if 'times' in save_data:
        times = save_data['times']
    else:
        state = save_data['final_state']
        N_t = len(currents)
        dt = state.dt
        times = np.linspace(0, N_t * dt, N_t)

    print(f"\n{'='*60}")
    print(f"Input Parameters (timestamp={timestamp}, job_index={job_index}):")
    print(f"{'='*60}")
    for key, value in input_kwargs.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, np.real(currents), 'b-', linewidth=2, label='Real(J)')
    ax.plot(times, np.imag(currents), 'r-', linewidth=2, label='Imag(J)')
    ax.plot(times, np.abs(currents), 'k--', linewidth=1.5, label='|J|', alpha=0.5)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Current J(t)', fontsize=12)
    ax.set_title(f'Current vs Time (job {job_index})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'current_vs_time_{timestamp}_job{job_index}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_equilibrated_gap_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir='analysis_plots'):
    """
    Plot equilibrated gap vs parameter for multiple timestamps.

    Computes average gap and standard deviation over last n_average timesteps.

    Args:
        timestamps: List of timestamps
        parameter: 'temperature' or 'vector_potential' (which parameter to plot against)
        n_average: Number of final timesteps to average (default 100)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots

    Returns:
        dict: {'parameter_values': array, 'gap_means': array, 'gap_errors': array}
    """
    parameter_values = []
    gap_means = []
    gap_errors = []

    for timestamp in timestamps:
        job_no = io.recover_job_no(timestamp=timestamp)

        for job_idx in range(job_no):
            input_kwargs, save_data = load_job_data(timestamp, job_idx)
            gaps = save_data['gaps']

            gap_eq = np.mean(gaps[-n_average:])
            gap_std = np.std(gaps[-n_average:])

            gap_means.append(gap_eq)
            gap_errors.append(gap_std)

            if parameter == 'temperature':
                param_value = input_kwargs['system_parameters']['temperature']
            elif parameter == 'vector_potential':
                vector_potentials = save_data['vector_potentials']
                param_value = np.max(np.abs(vector_potentials))
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            parameter_values.append(param_value)

    parameter_values = np.array(parameter_values)
    gap_means = np.array(gap_means)
    gap_errors = np.array(gap_errors)

    sort_idx = np.argsort(parameter_values)
    parameter_values = parameter_values[sort_idx]
    gap_means = gap_means[sort_idx]
    gap_errors = gap_errors[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(parameter_values, np.abs(gap_means), yerr=gap_errors, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Equilibrated Gap |Δ|', fontsize=12)
    ax.set_title(f'Equilibrated Gap vs {parameter.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'gap_vs_{parameter}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {'parameter_values': parameter_values, 'gap_means': gap_means, 'gap_errors': gap_errors}


def plot_equilibrated_current_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir='analysis_plots'):
    """
    Plot equilibrated current vs parameter for multiple timestamps.

    Computes average current and standard deviation over last n_average timesteps.

    Args:
        timestamps: List of timestamps
        parameter: 'temperature' or 'vector_potential' (which parameter to plot against)
        n_average: Number of final timesteps to average (default 100)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots

    Returns:
        dict: {'parameter_values': array, 'current_means': array, 'current_errors': array}
    """
    parameter_values = []
    current_means = []
    current_errors = []

    for timestamp in timestamps:
        job_no = io.recover_job_no(timestamp=timestamp)

        for job_idx in range(job_no):
            input_kwargs, save_data = load_job_data(timestamp, job_idx)
            currents = save_data['currents']

            current_eq = np.mean(currents[-n_average:])
            current_std = np.std(currents[-n_average:])

            current_means.append(current_eq)
            current_errors.append(current_std)

            if parameter == 'temperature':
                param_value = input_kwargs['system_parameters']['temperature']
            elif parameter == 'vector_potential':
                vector_potentials = save_data['vector_potentials']
                param_value = np.max(np.abs(vector_potentials))
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            parameter_values.append(param_value)

    parameter_values = np.array(parameter_values)
    current_means = np.array(current_means)
    current_errors = np.array(current_errors)

    sort_idx = np.argsort(parameter_values)
    parameter_values = parameter_values[sort_idx]
    current_means = current_means[sort_idx]
    current_errors = current_errors[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(parameter_values, np.abs(current_means), yerr=current_errors, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Equilibrated Current |J|', fontsize=12)
    ax.set_title(f'Equilibrated Current vs {parameter.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'current_vs_{parameter}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {'parameter_values': parameter_values, 'current_means': current_means, 'current_errors': current_errors}


# ============================================================================
# Internal Plotting Functions
# ============================================================================

def _plot_normalization(totals, gf_type, time_grid, save_dir, timestamp):
    """Plot normalization check for gr or gk."""
    pauli_names = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx in range(4):
        ax = axes.flat[pauli_idx]
        component = totals[pauli_idx, :]
        ax.plot(time_grid, np.real(component), 'b-', linewidth=2, label='Real', alpha=0.7)
        ax.plot(time_grid, np.imag(component), 'r-', linewidth=2, label='Imag', alpha=0.7)
        ax.set_xlabel("t₂", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'g^{gf_type.upper()}: {pauli_names[pauli_idx]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.tight_layout()
    plt.suptitle(f'g^{gf_type.upper()} Normalization (timestamp {timestamp})', fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.97)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'norm_{gf_type}_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_fdt(gk_actual, gk_fdt, error, time_grid, save_dir, timestamp):
    """Plot FDT comparison."""
    pauli_names = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx in range(4):
        gk_actual_pauli = gk_actual.trace(pauli_idx) / 2
        gk_fdt_pauli = gk_fdt.trace(pauli_idx) / 2
        error_pauli = error.trace(pauli_idx) / 2

        if gk_actual_pauli.ndim == 2:
            gk_actual_pauli = gk_actual_pauli[0, :]
            gk_fdt_pauli = gk_fdt_pauli[0, :]
            error_pauli = error_pauli[0, :]

        ax = axes.flat[pauli_idx]
        ax.plot(time_grid, np.real(gk_actual_pauli), 'b-', linewidth=2, label='Actual (Real)', alpha=0.8)
        ax.plot(time_grid, np.imag(gk_actual_pauli), 'r-', linewidth=2, label='Actual (Imag)', alpha=0.8)
        ax.plot(time_grid, np.real(gk_fdt_pauli), 'b--', linewidth=2, label='FDT (Real)', alpha=0.6)
        ax.plot(time_grid, np.imag(gk_fdt_pauli), 'r--', linewidth=2, label='FDT (Imag)', alpha=0.6)

        ax.set_xlabel('Time t\'', fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_names[pauli_idx]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

        max_err = np.max(np.abs(error_pauli))
        max_val = np.max(np.abs(gk_actual_pauli))
        rel_err = max_err / (max_val + 1e-10) * 100
        ax.text(0.02, 0.98, f'max err: {max_err:.2e}\nrel: {rel_err:.1f}%', transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.suptitle(f'FDT Check (timestamp {timestamp})', fontsize=14, fontweight='bold', y=0.998)
    plt.subplots_adjust(top=0.97)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'fdt_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_time_translation(max_diffs_gr, max_diffs_gk, num_rows, threshold, save_dir, timestamp):
    """Plot time-translation invariance check."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    row_indices = np.arange(1, num_rows + 1)

    ax1.plot(row_indices, max_diffs_gr, 'bo-', linewidth=2, markersize=8, label='Max difference')
    ax1.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0e})')
    ax1.set_xlabel('Row index from last', fontsize=12)
    ax1.set_ylabel('Max difference', fontsize=12)
    ax1.set_title('g^R Time Translation Invariance', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.plot(row_indices, max_diffs_gk, 'ro-', linewidth=2, markersize=8, label='Max difference')
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0e})')
    ax2.set_xlabel('Row index from last', fontsize=12)
    ax2.set_ylabel('Max difference', fontsize=12)
    ax2.set_title('g^K Time Translation Invariance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.suptitle(f'Time Translation Invariance (timestamp {timestamp})', fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.94)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'tti_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
