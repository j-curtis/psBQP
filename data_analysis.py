"""
Data analysis tools for Keldysh-Usadel simulations.

COMPATIBILITY WITH REDUCED STATES (save_full_state=False):
- check_normalizations: ✓ Works (checks last timestep)
- check_fdt: ✓ Works (checks last timestep)
- check_time_translational_invariance: ✗ Requires full time history (returns None if reduced)
- check_state_evolution: ✗ Requires full time history (returns None if reduced)
- plot_gap: ✓ Works (plots saved gap array)
- plot_current: ✓ Works (plots saved current array)
- plot_equilibrated_gap_vs_parameter: ✓ Works (uses saved arrays)
- plot_equilibrated_current_vs_parameter: ✓ Works (uses saved arrays)
- partial_fourier_transform: ✗ Requires full two-time tensor (N_t, N_t)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

from demler_tools.file_manager import io, path_management

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution

# Configure matplotlib to use LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}'

# ============================================================================
# Data Loading and Parameter Extraction
# ============================================================================

def load_job_data(timestamp, job_index, running_machine='laptop'):
    """
    Load simulation results using demler_tools.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index (run_index)
        running_machine: Machine identifier ('laptop' or 'cluster_euler')

    Returns:
        tuple: (input_kwargs, save_data)
            input_kwargs: Dict with system_parameters, grid_parameters, etc.
            save_data: Dict with final_state, gaps, currents, vector_potentials, times
    """
    # Initialize path_management if not already done
    if not hasattr(path_management, 'project_data'):
        path_management.initialize(project_name='psBQP-keldysh')

    # Get all kwargs from the timestamp
    kwargs_array = io.recover_full_calculation_arguments(timestamp=timestamp)
    input_kwargs = kwargs_array[job_index]

    # Construct the result file path
    result_file = os.path.join(
        path_management.default_simulation_data_path(running_machine=running_machine),
        str(timestamp),
        path_management.relative_path_raw_result_file().format(job_index)
    )

    # Try main path first, then archive (only for cluster)
    if not os.path.exists(result_file):
        if running_machine == 'cluster_euler':
            result_file = os.path.join(
                path_management.default_autoarchive_path(running_machine=running_machine),
                str(timestamp),
                path_management.relative_path_raw_result_file().format(job_index)
            )

        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Result file not found for timestamp {timestamp}, job {job_index}")

    # Load the result data
    with open(result_file, 'rb') as f:
        save_data = pickle.load(f)

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

    # Compute max error per Pauli channel from totals
    # gr_totals and gk_totals have shape (4, N_t) for 4 Pauli components
    gr_max_errors_per_channel = np.zeros(4)
    gk_max_errors_per_channel = np.zeros(4)

    for pauli_idx in range(4):
        # For gr: τ0 should integrate to 1, others to 0
        expected_value = 1.0 if pauli_idx == 0 else 0.0
        gr_max_errors_per_channel[pauli_idx] = np.max(np.abs(gr_totals[pauli_idx, :] - expected_value))

        # For gk: compute error from the totals
        gk_max_errors_per_channel[pauli_idx] = np.max(np.abs(gk_totals[pauli_idx, :]))

    _plot_normalization(gr_totals, 'gr', evolution.time_grid, save_plot, save_dir, timestamp)
    _plot_normalization(gk_totals, 'gk', evolution.time_grid, save_plot, save_dir, timestamp)

    return {'gr_max_error': gr_max_errors_per_channel, 'gk_max_error': gk_max_errors_per_channel, 'gr_totals': gr_totals, 'gk_totals': gk_totals}


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

    # Compute max error per Pauli channel
    max_errors_per_channel = np.zeros(4)
    for pauli_idx in range(4):
        error_pauli = error_row.trace(pauli_idx) / 2
        if error_pauli.ndim == 2:
            error_pauli = error_pauli[0, :]
        max_errors_per_channel[pauli_idx] = np.max(np.abs(error_pauli))

    _plot_fdt(gk_actual_row, gk_fdt_row, error_row, evolution.time_grid, save_plot, save_dir, timestamp)

    return {'max_error': max_errors_per_channel, 'gk_actual': gk_actual_row, 'gk_fdt': gk_fdt_row, 'error': error_row}


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
        dict: {'max_diff_gr': array, 'max_diff_gk': array, 'passed': bool, 'threshold': float}
            Returns None if state has insufficient time history
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index)
    state = save_data['final_state']
    n_times = state.gr.data.shape[2]

    # Check if state has enough time history
    if n_times < 2:
        print(f"Warning: State only has {n_times} timestep. Time-translation invariance check requires multiple timesteps.")
        print("This state was likely saved with save_full_state=False. Skipping check.")
        return None

    time_grid = np.linspace(-state.T_max, 0, n_times)

    max_diffs_gr = _compute_time_translation_diffs(state.gr, num_rows)
    max_diffs_gk = _compute_time_translation_diffs(state.gk, num_rows)

    max_diff_gr, max_diff_gk = np.max(max_diffs_gr), np.max(max_diffs_gk)
    passed = (max_diff_gr < threshold) and (max_diff_gk < threshold)

    _plot_time_translation(state.gr, state.gk, time_grid, num_rows, threshold, save_plot, save_dir, timestamp)

    return {'max_diff_gr': max_diffs_gr, 'max_diff_gk': max_diffs_gk, 'passed': passed, 'threshold': threshold}


def check_state_evolution(timestamp, job_index, save_plot=False, save_dir='analysis_plots'):
    """
    Compare state evolution between initial (t=0, index 0) and final (t=-1, index -1) times.

    Extracts rows at earliest and latest times for both g^R and g^K, then computes
    differences across all 4 Pauli components to show how much the state has evolved.

    Note: Returns None if state has only one timestep (saved with save_full_state=False).

    Args:
        timestamp: Timestamp to analyze
        job_index: Job index
        save_plot: Whether to save plots (default False)
        save_dir: Directory for plots

    Returns:
        dict: {
            'gr_max_diff': array (4,) - max difference per Pauli component for g^R
            'gk_max_diff': array (4,) - max difference per Pauli component for g^K
            'gr_initial': NambuKeldyshTensor - g^R at earliest time
            'gr_final': NambuKeldyshTensor - g^R at latest time
            'gk_initial': NambuKeldyshTensor - g^K at earliest time
            'gk_final': NambuKeldyshTensor - g^K at latest time
        }
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index)
    state = save_data['final_state']
    n_times = state.gr.data.shape[2]

    # Check if state has enough time history
    if n_times < 2:
        print(f"Warning: State only has {n_times} timestep. State evolution check requires multiple timesteps.")
        print("This state was likely saved with save_full_state=False. Skipping check.")
        return None

    time_grid = np.linspace(-state.T_max, 0, n_times)

    # Extract rows at initial (index 0) and final (index -1) times
    gr_initial = state.gr[0, :]  # t = -T_max (earliest)
    gr_final = state.gr[-1, :]   # t = 0 (latest)
    gk_initial = state.gk[0, :]
    gk_final = state.gk[-1, :]

    # Compute differences for each Pauli component
    gr_max_diff = np.zeros(4)
    gk_max_diff = np.zeros(4)

    for pauli_idx in range(4):
        # g^R differences
        gr_init_pauli = gr_initial.trace(pauli_idx) / 2
        gr_final_pauli = gr_final.trace(pauli_idx) / 2
        if gr_init_pauli.ndim == 2:
            gr_init_pauli = gr_init_pauli[0, :]
            gr_final_pauli = gr_final_pauli[0, :]
        gr_diff = np.abs(gr_final_pauli - gr_init_pauli)
        gr_max_diff[pauli_idx] = np.max(gr_diff)

        # g^K differences
        gk_init_pauli = gk_initial.trace(pauli_idx) / 2
        gk_final_pauli = gk_final.trace(pauli_idx) / 2
        if gk_init_pauli.ndim == 2:
            gk_init_pauli = gk_init_pauli[0, :]
            gk_final_pauli = gk_final_pauli[0, :]
        gk_diff = np.abs(gk_final_pauli - gk_init_pauli)
        gk_max_diff[pauli_idx] = np.max(gk_diff)

    # Plot comparison
    _plot_state_evolution(gr_initial, gr_final, gk_initial, gk_final,
                         time_grid, save_plot, save_dir, timestamp)

    return {
        'gr_max_diff': gr_max_diff,
        'gk_max_diff': gk_max_diff,
        'gr_initial': gr_initial,
        'gr_final': gr_final,
        'gk_initial': gk_initial,
        'gk_final': gk_final
    }


def _compute_time_translation_diffs(tensor, num_rows):
    """
    Compute time-translation invariance differences for a tensor.

    Compares each row shifted by -i with the reference (last row shifted by -1).
    For time-translation invariance: g(t_i, t') shifted by -i should equal g(t_j, t') shifted by -j.
    """
    max_diffs = []

    # Reference: last row shifted by -1
    row_ref = tensor[-1, :]
    row_ref_shifted = row_ref.shift(-1, axis=0)

    for i in range(1, num_rows + 1):
        row_i = tensor[-i, :]
        row_i_shifted = row_i.shift(-i, axis=0)

        max_diff = 0.0
        for pauli_idx in range(4):
            comp_i = row_i_shifted.trace(pauli_idx) / 2
            comp_ref = row_ref_shifted.trace(pauli_idx) / 2
            if comp_i.ndim == 2:
                comp_i = comp_i[0, :]
                comp_ref = comp_ref[0, :]
            # Skip first few points to avoid edge effects from shifting
            if len(comp_i) > 10:
                comp_i = comp_i[10:]
                comp_ref = comp_ref[10:]
            diff = np.max(np.abs(comp_i - comp_ref))
            max_diff = max(max_diff, diff)
        max_diffs.append(max_diff)
    return max_diffs


# ============================================================================
# Fourier Transform Functions
# ============================================================================

def partial_fourier_transform(g_two_time, time_grid):
    """
    Transform Green's function from two-time g(t,t') to mixed Wigner g(ε,t).

    Modified Fourier transform to avoid interpolation of midpoints:
        ĝ(ε; t) = ∫ dτ e^{i(2ε)τ} ĝ(t, t-τ)

    where τ = t - t'. This avoids the τ/2 midpoint problem by working
    directly with grid points and modifying the Fourier kernel.

    For each fixed t = t_i on the grid:
    - Extract g(t_i, t_j) for all j
    - Define τ_j = t_i - t_j (relative time)
    - FFT with kernel e^{i(2ε)τ} (factor of 2 in exponent)
    - Energy grid scaled accordingly: ε_grid = 2 × (standard FFT frequencies)

    Args:
        g_two_time: NambuKeldyshTensor with shape (2, 2, N_t, N_t)
                    representing g(t_i, t_j) in two-time form
        time_grid: 1D array of time points [t_0, ..., t_{N_t-1}]
                   typically from evolution.time_grid or np.linspace(-T_max, 0, N_t)

    Returns:
        dict: {
            'energy_grid': 1D array of energy points ε (N_t points)
            'com_time_grid': 1D array of times t (N_t points, same as input)
            'g_mixed': NambuKeldyshTensor with shape (2, 2, N_ε, N_t)
                       representing g(ε_n, t_i) in mixed Wigner representation
        }

    Example:
        >>> # Load data
        >>> input_kwargs, save_data = load_job_data(timestamp, job_index)
        >>> state = save_data['final_state']
        >>> evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
        >>> time_grid = evolution.time_grid
        >>>
        >>> # Transform retarded Green's function
        >>> result_gr = partial_fourier_transform(state.gr, time_grid)
        >>> energy_grid = result_gr['energy_grid']
        >>> g_mixed_gr = result_gr['g_mixed']
        >>>
        >>> # Transform Keldysh Green's function
        >>> result_gk = partial_fourier_transform(state.gk, time_grid)
    """
    # Setup
    N_t = len(time_grid)
    dt = (time_grid[-1] - time_grid[0]) / (N_t - 1)

    # Pauli matrices for reconstruction
    pauli = {
        0: np.array([[1, 0], [0, 1]]),      # τ₀ (identity)
        1: np.array([[0, 1], [1, 0]]),      # τ₁ (σ_x)
        2: np.array([[0, -1j], [1j, 0]]),   # τ₂ (σ_y)
        3: np.array([[1, 0], [0, -1]])      # τ₃ (σ_z)
    }

    # Initialize result array: (2, 2, N_ε, N_t)
    result_data = np.zeros((2, 2, N_t, N_t), dtype=complex)

    # Process each Pauli component
    for pauli_idx in range(4):
        # Extract Pauli component
        g_pauli = g_two_time.trace(pauli_idx) / 2.0

        # Handle both 2D and higher-dimensional cases
        if g_pauli.ndim == 2:
            g_pauli_data = g_pauli
        else:
            # If there are extra dimensions, take first element
            g_pauli_data = g_pauli[0, :]

        # For each fixed t = t_i, extract g(t_i, t_j) and FFT
        g_energy = np.zeros((N_t, N_t), dtype=complex)

        for i_t in range(N_t):
            # Extract row: g(t_i, t_j) for all j
            # This gives g as function of t_j for fixed t_i
            g_row = g_pauli_data[i_t, :]

            # Relative time: τ_j = t_i - t_j
            # We need to reorder so τ is uniformly spaced
            # Since t_j = time_grid, τ_j = t_i - time_grid
            # For FFT, we want τ in ascending order

            # Reverse the row so τ goes from negative to positive
            # τ ranges from (t_i - t_{N_t-1}) to (t_i - t_0)
            g_tau = g_row[::-1]  # Now indexed by increasing τ

            # FFT with modified kernel e^{i(2ε)τ}
            # Standard FFT gives e^{iωτ}, we want e^{i(2ε)τ}
            # So energy will be: ε = ω/2
            #
            # Use ifft for positive exponent convention
            g_fft = np.fft.ifft(g_tau) * N_t  # Remove 1/N normalization

            # Normalize: multiply by dτ to approximate integral
            g_fft *= dt

            # Shift to match energy_grid ordering
            g_energy[i_t, :] = np.fft.fftshift(g_fft)

        # Add Pauli structure to result
        # Transpose to (N_ε, N_t) for proper indexing
        g_transposed = g_energy.T

        # Add Pauli contribution
        for i in range(2):
            for j in range(2):
                result_data[i, j, :, :] += pauli[pauli_idx][i, j] * g_transposed

    # Create energy grid with factor of 2 modification
    # Standard FFT frequencies
    freq = np.fft.fftfreq(N_t, d=dt)  # Frequency in 1/time units
    # Energy grid with 2× factor: ε = 2 × (2πf)
    energy_grid = 2 * 2 * np.pi * freq
    energy_grid = np.fft.fftshift(energy_grid)  # Shift to center

    # Create NambuKeldyshTensor
    g_mixed = NambuKeldyshTensor(result_data)

    # Return result
    return {
        'energy_grid': energy_grid,
        'com_time_grid': time_grid,
        'g_mixed': g_mixed
    }


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

    pauli_labels = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

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

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_labels[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_labels[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([x_values[0], x_values[-1]])

    plt.tight_layout()

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

def plot_gap(timestamp, job_index=None, save_plot=False, save_dir='analysis_plots'):
    """
    Plot gap vs time for all jobs or a specific job.

    Displays input parameters before plotting.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index (default None plots all jobs in folder)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
    """
    # Determine which jobs to plot
    if job_index is None:
        job_no = io.recover_job_no(timestamp=timestamp)
        job_indices = range(job_no)
        plot_all = True
    else:
        job_indices = [job_index]
        plot_all = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for job_idx in job_indices:
        input_kwargs, save_data = load_job_data(timestamp, job_idx)
        gaps = save_data['gaps']

        if 'times' in save_data:
            times = save_data['times']
        else:
            state = save_data['final_state']
            N_t = len(gaps)
            dt = state.dt
            times = np.linspace(0, N_t * dt, N_t)

        print(f"\n{'='*60}")
        print(f"Input Parameters (timestamp={timestamp}, job_index={job_idx}):")
        print(f"{'='*60}")
        for key, value in input_kwargs.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        label = f'Job {job_idx}' if plot_all else None
        # Plot Real part
        axes[0].plot(times, np.real(gaps), linewidth=2, label=label, alpha=0.7)
        # Plot Imag part
        axes[1].plot(times, np.imag(gaps), linewidth=2, label=label, alpha=0.7)
        # Plot Magnitude
        axes[2].plot(times, np.abs(gaps), linewidth=2, label=label, alpha=0.7)

    # Configure axes
    title_suffix = 'All Jobs' if plot_all else f'Job {job_index}'

    axes[0].set_xlabel(r'Time $t$', fontsize=12)
    axes[0].set_ylabel(r'Re($\Delta(t)$)', fontsize=12)
    axes[0].set_title(f'Real($\Delta$) - {title_suffix}', fontsize=13)
    if plot_all:
        axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'Time $t$', fontsize=12)
    axes[1].set_ylabel(r'Im($\Delta(t)$)', fontsize=12)
    axes[1].set_title(f'Imag($\Delta$) - {title_suffix}', fontsize=13)
    if plot_all:
        axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel(r'Time $t$', fontsize=12)
    axes[2].set_ylabel(r'$|\Delta(t)|$', fontsize=12)
    axes[2].set_title(f'$|\Delta|$ - {title_suffix}', fontsize=13)
    if plot_all:
        axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        job_str = f'job{job_index}' if job_index is not None else 'all_jobs'
        plt.savefig(os.path.join(save_dir, f'gap_vs_time_{timestamp}_{job_str}.png'), dpi=150, bbox_inches='tight')

    plt.show()


def plot_current(timestamp, job_index=None, save_plot=False, save_dir='analysis_plots'):
    """
    Plot current vs time for all jobs or a specific job.

    Displays input parameters before plotting.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index (default None plots all jobs in folder)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
    """
    # Determine which jobs to plot
    if job_index is None:
        job_no = io.recover_job_no(timestamp=timestamp)
        job_indices = range(job_no)
        plot_all = True
    else:
        job_indices = [job_index]
        plot_all = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for job_idx in job_indices:
        input_kwargs, save_data = load_job_data(timestamp, job_idx)
        currents = save_data['currents']

        if 'times' in save_data:
            times = save_data['times']
        else:
            state = save_data['final_state']
            N_t = len(currents)
            dt = state.dt
            times = np.linspace(0, N_t * dt, N_t)

        print(f"\n{'='*60}")
        print(f"Input Parameters (timestamp={timestamp}, job_index={job_idx}):")
        print(f"{'='*60}")
        for key, value in input_kwargs.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        label = f'Job {job_idx}' if plot_all else None
        # Plot Real part
        axes[0].plot(times, np.real(currents), linewidth=2, label=label, alpha=0.7)
        # Plot Imag part
        axes[1].plot(times, np.imag(currents), linewidth=2, label=label, alpha=0.7)
        # Plot Magnitude
        axes[2].plot(times, np.abs(currents), linewidth=2, label=label, alpha=0.7)

    # Configure axes
    title_suffix = 'All Jobs' if plot_all else f'Job {job_index}'

    axes[0].set_xlabel(r'Time $t$', fontsize=12)
    axes[0].set_ylabel(r'Re($J(t)$)', fontsize=12)
    axes[0].set_title(f'Real($J$) - {title_suffix}', fontsize=13)
    if plot_all:
        axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'Time $t$', fontsize=12)
    axes[1].set_ylabel(r'Im($J(t)$)', fontsize=12)
    axes[1].set_title(f'Imag($J$) - {title_suffix}', fontsize=13)
    if plot_all:
        axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel(r'Time $t$', fontsize=12)
    axes[2].set_ylabel(r'$|J(t)|$', fontsize=12)
    axes[2].set_title(f'$|J|$ - {title_suffix}', fontsize=13)
    if plot_all:
        axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        job_str = f'job{job_index}' if job_index is not None else 'all_jobs'
        plt.savefig(os.path.join(save_dir, f'current_vs_time_{timestamp}_{job_str}.png'), dpi=150, bbox_inches='tight')

    plt.show()


def plot_equilibrated_gap_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir='analysis_plots', combine_timestamps=False, x_external=None, y_external=None):
    """
    Plot equilibrated gap vs parameter for multiple timestamps.

    Computes average gap and standard deviation over last n_average timesteps.

    Args:
        timestamps: List of timestamps
        parameter: 'temperature' or 'vector_potential' (which parameter to plot against)
        n_average: Number of final timesteps to average (default 100)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
        combine_timestamps: If True, combine all timestamps into single curve (default False)
        x_external: Optional external x-values for comparison (default None)
        y_external: Optional external y-values for comparison (default None)

    Returns:
        dict: {'parameter_values': array, 'gap_means': array, 'gap_errors': array}
    """
    # Store data per timestamp for separate plotting
    timestamp_data = {}
    all_parameter_values = []
    all_gap_means = []
    all_gap_errors = []

    for timestamp in timestamps:
        job_no = io.recover_job_no(timestamp=timestamp)

        parameter_values = []
        gap_means = []
        gap_errors = []

        for job_idx in range(job_no):
            input_kwargs, save_data = load_job_data(timestamp, job_idx)
            gaps = save_data['gaps']

            gap_eq = np.mean(gaps[-n_average:])
            gap_std = np.std(gaps[-n_average:])

            gap_means.append(gap_eq)
            gap_errors.append(gap_std)
            all_gap_means.append(gap_eq)
            all_gap_errors.append(gap_std)

            if parameter == 'temperature':
                param_value = input_kwargs['system_parameters']['temperature']
            elif parameter == 'vector_potential':
                vector_potentials = save_data['vector_potentials']
                param_value = np.max(np.abs(vector_potentials))
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            parameter_values.append(param_value)
            all_parameter_values.append(param_value)

        # Sort data for this timestamp
        if len(parameter_values) > 0:
            sort_idx = np.argsort(parameter_values)
            timestamp_data[timestamp] = {
                'parameter_values': np.array(parameter_values)[sort_idx],
                'gap_means': np.array(gap_means)[sort_idx],
                'gap_errors': np.array(gap_errors)[sort_idx]
            }

    # Convert to arrays for combined analysis
    all_parameter_values = np.array(all_parameter_values)
    all_gap_means = np.array(all_gap_means)
    all_gap_errors = np.array(all_gap_errors)

    if len(all_parameter_values) > 0:
        max_idx = np.argmax(np.abs(all_gap_means))
        max_gap = np.abs(all_gap_means[max_idx])
        max_param = all_parameter_values[max_idx]
    else:
        max_gap = 0
        max_param = 0

    fig, ax = plt.subplots(figsize=(8, 5))

    if combine_timestamps:
        # Old behavior: combine all into single curve
        sort_idx = np.argsort(all_parameter_values)
        ax.errorbar(all_parameter_values[sort_idx], np.abs(all_gap_means[sort_idx]),
                   yerr=all_gap_errors[sort_idx], fmt='o-', capsize=5,
                   linewidth=2, markersize=8, label='All data')
    else:
        # New behavior: plot each timestamp separately
        for i, (timestamp, data) in enumerate(timestamp_data.items()):
            label = f'{timestamp}' if len(timestamps) > 1 else 'Data'
            ax.errorbar(data['parameter_values'], np.abs(data['gap_means']),
                       yerr=data['gap_errors'], fmt='o-', capsize=5,
                       linewidth=2, markersize=8, label=label, alpha=0.7)

    # Plot external data if provided
    if x_external is not None and y_external is not None:
        ax.plot(x_external, np.abs(y_external), 'kx--', linewidth=2, markersize=10,
               label='External data', alpha=0.8)

    ax.axvline(max_param, color='gray', linestyle=':', linewidth=2, alpha=0.7,
              label=f'Max at {parameter.replace("_", " ")}={max_param:.3f}')
    ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(r'Equilibrated Gap $|\Delta|$', fontsize=12)
    ax.set_title(f'Equilibrated Gap vs {parameter.replace("_", " ").title()}' + '\n' +
                f'Max $|\\Delta|$ = {max_gap:.4f} at {parameter.replace("_", " ")} = {max_param:.3f}',
                fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'gap_vs_{parameter}.png'), dpi=150, bbox_inches='tight')

    plt.show()

    return {'parameter_values': all_parameter_values, 'gap_means': all_gap_means,
            'gap_errors': all_gap_errors, 'max_gap': max_gap, 'max_parameter': max_param,
            'timestamp_data': timestamp_data}


def plot_equilibrated_current_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir='analysis_plots', combine_timestamps=False, x_external=None, y_external=None):
    """
    Plot equilibrated current vs parameter for multiple timestamps.

    Computes average current and standard deviation over last n_average timesteps.

    Args:
        timestamps: List of timestamps
        parameter: 'temperature' or 'vector_potential' (which parameter to plot against)
        n_average: Number of final timesteps to average (default 100)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
        combine_timestamps: If True, combine all timestamps into single curve (default False)
        x_external: Optional external x-values for comparison (default None)
        y_external: Optional external y-values for comparison (default None)

    Returns:
        dict: {'parameter_values': array, 'current_means': array, 'current_errors': array}
    """
    # Store data per timestamp for separate plotting
    timestamp_data = {}
    all_parameter_values = []
    all_current_means = []
    all_current_errors = []

    for timestamp in timestamps:
        job_no = io.recover_job_no(timestamp=timestamp)

        parameter_values = []
        current_means = []
        current_errors = []

        for job_idx in range(job_no):
            input_kwargs, save_data = load_job_data(timestamp, job_idx)
            currents = save_data['currents']

            current_eq = np.mean(currents[-n_average:])
            current_std = np.std(currents[-n_average:])

            current_means.append(current_eq)
            current_errors.append(current_std)
            all_current_means.append(current_eq)
            all_current_errors.append(current_std)

            if parameter == 'temperature':
                param_value = input_kwargs['system_parameters']['temperature']
            elif parameter == 'vector_potential':
                vector_potentials = save_data['vector_potentials']
                param_value = np.max(np.abs(vector_potentials))
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            parameter_values.append(param_value)
            all_parameter_values.append(param_value)

        # Sort data for this timestamp
        if len(parameter_values) > 0:
            sort_idx = np.argsort(parameter_values)
            timestamp_data[timestamp] = {
                'parameter_values': np.array(parameter_values)[sort_idx],
                'current_means': np.array(current_means)[sort_idx],
                'current_errors': np.array(current_errors)[sort_idx]
            }

    # Convert to arrays for combined analysis
    all_parameter_values = np.array(all_parameter_values)
    all_current_means = np.array(all_current_means)
    all_current_errors = np.array(all_current_errors)

    if len(all_parameter_values) > 0:
        max_idx = np.argmax(np.abs(all_current_means))
        max_current = np.abs(all_current_means[max_idx])
        max_param = all_parameter_values[max_idx]
    else:
        max_current = 0
        max_param = 0

    fig, ax = plt.subplots(figsize=(8, 5))

    if combine_timestamps:
        # Old behavior: combine all into single curve
        sort_idx = np.argsort(all_parameter_values)
        ax.errorbar(all_parameter_values[sort_idx], np.abs(all_current_means[sort_idx]),
                   yerr=all_current_errors[sort_idx], fmt='o-', capsize=5,
                   linewidth=2, markersize=8, label='All data')
    else:
        # New behavior: plot each timestamp separately
        for i, (timestamp, data) in enumerate(timestamp_data.items()):
            label = f'{timestamp}' if len(timestamps) > 1 else 'Data'
            ax.errorbar(data['parameter_values'], np.abs(data['current_means']),
                       yerr=data['current_errors'], fmt='o-', capsize=5,
                       linewidth=2, markersize=8, label=label, alpha=0.7)

    # Plot external data if provided
    if x_external is not None and y_external is not None:
        ax.plot(x_external, np.abs(y_external), 'kx--', linewidth=2, markersize=10,
               label='External data', alpha=0.8)

    ax.axvline(max_param, color='gray', linestyle=':', linewidth=2, alpha=0.7,
              label=f'Max at {parameter.replace("_", " ")}={max_param:.3f}')
    ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(r'Equilibrated Current $|J|$', fontsize=12)
    ax.set_title(f'Equilibrated Current vs {parameter.replace("_", " ").title()}' + '\n' +
                f'Max $|J|$ = {max_current:.4f} at {parameter.replace("_", " ")} = {max_param:.3f}',
                fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'current_vs_{parameter}.png'), dpi=150, bbox_inches='tight')

    plt.show()

    return {'parameter_values': all_parameter_values, 'current_means': all_current_means,
            'current_errors': all_current_errors, 'max_current': max_current, 'max_parameter': max_param,
            'timestamp_data': timestamp_data}


# ============================================================================
# Internal Plotting Functions
# ============================================================================

def _plot_normalization(totals, gf_type, time_grid, save_plot, save_dir, timestamp):
    """Plot normalization check for gr or gk."""
    pauli_names = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for pauli_idx in range(4):
        ax = axes.flat[pauli_idx]
        component = totals[pauli_idx, :]
        ax.plot(time_grid, np.real(component), 'b-', linewidth=2, label='Real', alpha=0.7)
        ax.plot(time_grid, np.imag(component), 'r-', linewidth=2, label='Imag', alpha=0.7)
        ax.set_xlabel(r"$t_2$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'$g^{{{gf_type}}}$: {pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'norm_{gf_type}_{timestamp}.png'), dpi=150, bbox_inches='tight')

    plt.show()


def _plot_fdt(gk_actual, gk_fdt, error, time_grid, save_plot, save_dir, timestamp):
    """Plot FDT comparison."""
    pauli_names = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

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

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

        max_err = np.max(np.abs(error_pauli))
        max_val = np.max(np.abs(gk_actual_pauli))
        rel_err = max_err / (max_val + 1e-10) * 100
        ax.text(0.02, 0.98, f'max err: {max_err:.2e}\nrel: {rel_err:.1f}%', transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'fdt_{timestamp}.png'), dpi=150, bbox_inches='tight')

    plt.show()


def _plot_time_translation(gr_tensor, gk_tensor, time_grid, num_rows, threshold, save_plot, save_dir, timestamp):
    """Plot time-translation invariance check by overlaying last num_rows rows."""
    from matplotlib import cm

    pauli_names = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']

    # Create colormap (darkest blue = most recent time = largest index)
    blues = cm.get_cmap('Blues', num_rows + 2)

    # Plot g^R (real part only)
    fig_gr, axes_gr = plt.subplots(2, 2, figsize=(10, 7))
    for pauli_idx in range(4):
        ax = axes_gr.flat[pauli_idx]

        for i in range(1, num_rows + 1):
            row = gr_tensor[-i, :].shift(i, axis=0)
            comp = row.trace(pauli_idx) / 2
            if comp.ndim == 2:
                comp = comp[0, :]

            # Darker color for more recent times (larger i)
            color = blues(num_rows + 2 - i)
            label = f'$t = {time_grid[-i]:.2f}$' if i <= 5 else None
            ax.plot(time_grid, np.real(comp), '-', color=color, linewidth=1.5, alpha=0.8, label=label)

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'$g^r$: {pauli_names[pauli_idx]} (Real)', fontsize=11)
        if pauli_idx == 0:
            ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'tti_gr_{timestamp}.png'), dpi=150, bbox_inches='tight')

    plt.show()

    # Plot g^K (imaginary part only)
    fig_gk, axes_gk = plt.subplots(2, 2, figsize=(10, 7))
    for pauli_idx in range(4):
        ax = axes_gk.flat[pauli_idx]

        for i in range(1, num_rows + 1):
            row = gk_tensor[-i, :].shift(i, axis=0)
            comp = row.trace(pauli_idx) / 2
            if comp.ndim == 2:
                comp = comp[0, :]

            # Darker color for more recent times (larger i)
            color = blues(num_rows + 2 - i)
            label = f'$t = {time_grid[-i]:.2f}$' if i <= 5 else None
            ax.plot(time_grid, np.imag(comp), '-', color=color, linewidth=1.5, alpha=0.8, label=label)

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'$g^k$: {pauli_names[pauli_idx]} (Imag)', fontsize=11)
        if pauli_idx == 0:
            ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'tti_gk_{timestamp}.png'), dpi=150, bbox_inches='tight')

    plt.show()


def _plot_state_evolution(gr_initial, gr_final, gk_initial, gk_final, time_grid, save_plot, save_dir, timestamp):
    """
    Plot state evolution comparison between initial and final times.

    Shows overlay of initial (t=0, dashed) and final (t=-1, solid) states
    for all 4 Pauli components of both g^R and g^K.
    """
    pauli_names = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']

    # Plot g^R evolution
    fig_gr, axes_gr = plt.subplots(2, 2, figsize=(10, 7))
    for pauli_idx in range(4):
        # Extract Pauli components
        gr_init_pauli = gr_initial.trace(pauli_idx) / 2
        gr_final_pauli = gr_final.trace(pauli_idx) / 2

        if gr_init_pauli.ndim == 2:
            gr_init_pauli = gr_init_pauli[0, :]
            gr_final_pauli = gr_final_pauli[0, :]

        ax = axes_gr.flat[pauli_idx]

        # Plot initial state (dashed)
        ax.plot(time_grid, np.real(gr_init_pauli), 'b--', linewidth=2, label='Initial (Real)', alpha=0.7)
        ax.plot(time_grid, np.imag(gr_init_pauli), 'r--', linewidth=2, label='Initial (Imag)', alpha=0.7)

        # Plot final state (solid)
        ax.plot(time_grid, np.real(gr_final_pauli), 'b-', linewidth=2, label='Final (Real)', alpha=0.8)
        ax.plot(time_grid, np.imag(gr_final_pauli), 'r-', linewidth=2, label='Final (Imag)', alpha=0.8)

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'$g^R$: {pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

        # Add max difference annotation
        diff = np.abs(gr_final_pauli - gr_init_pauli)
        max_diff = np.max(diff)
        max_diff_idx = np.argmax(diff)
        max_diff_time = time_grid[max_diff_idx]

        ax.axvline(max_diff_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.text(0.02, 0.98, f'max diff: {max_diff:.2e}\nat $t\'$={max_diff_time:.2f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'$g^R$ State Evolution: Initial (t=0) vs Final (t=-1)', fontsize=12)
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'state_evolution_gr_{timestamp}.png'), dpi=150, bbox_inches='tight')

    plt.show()

    # Plot g^K evolution
    fig_gk, axes_gk = plt.subplots(2, 2, figsize=(10, 7))
    for pauli_idx in range(4):
        # Extract Pauli components
        gk_init_pauli = gk_initial.trace(pauli_idx) / 2
        gk_final_pauli = gk_final.trace(pauli_idx) / 2

        if gk_init_pauli.ndim == 2:
            gk_init_pauli = gk_init_pauli[0, :]
            gk_final_pauli = gk_final_pauli[0, :]

        ax = axes_gk.flat[pauli_idx]

        # Plot initial state (dashed)
        ax.plot(time_grid, np.real(gk_init_pauli), 'b--', linewidth=2, label='Initial (Real)', alpha=0.7)
        ax.plot(time_grid, np.imag(gk_init_pauli), 'r--', linewidth=2, label='Initial (Imag)', alpha=0.7)

        # Plot final state (solid)
        ax.plot(time_grid, np.real(gk_final_pauli), 'b-', linewidth=2, label='Final (Real)', alpha=0.8)
        ax.plot(time_grid, np.imag(gk_final_pauli), 'r-', linewidth=2, label='Final (Imag)', alpha=0.8)

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'$g^K$: {pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], time_grid[-1]])

        # Add max difference annotation
        diff = np.abs(gk_final_pauli - gk_init_pauli)
        max_diff = np.max(diff)
        max_diff_idx = np.argmax(diff)
        max_diff_time = time_grid[max_diff_idx]

        ax.axvline(max_diff_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.text(0.02, 0.98, f'max diff: {max_diff:.2e}\nat $t\'$={max_diff_time:.2f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'$g^K$ State Evolution: Initial (t=0) vs Final (t=-1)', fontsize=12)
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'state_evolution_gk_{timestamp}.png'), dpi=150, bbox_inches='tight')

    plt.show()
