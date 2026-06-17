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

    # Construct paths for kwargs and result files
    simulation_data_path = path_management.default_simulation_data_path(running_machine=running_machine)
    kwargs_file = os.path.join(simulation_data_path, str(timestamp), 'inputs', path_management.args_input_filetag())
    result_file = os.path.join(simulation_data_path, str(timestamp), path_management.relative_path_raw_result_file().format(job_index))

    # Try main path first, then archive (only for cluster)
    if not os.path.exists(kwargs_file) or not os.path.exists(result_file):
        if running_machine == 'cluster_euler':
            archive_path = path_management.default_autoarchive_path(running_machine=running_machine)
            if not os.path.exists(kwargs_file):
                kwargs_file = os.path.join(archive_path, str(timestamp), 'inputs', path_management.args_input_filetag())
            if not os.path.exists(result_file):
                result_file = os.path.join(archive_path, str(timestamp), path_management.relative_path_raw_result_file().format(job_index))

    # Check if files exist after trying archive
    if not os.path.exists(kwargs_file):
        raise FileNotFoundError(f"Kwargs file not found for timestamp {timestamp}")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found for timestamp {timestamp}, job {job_index}")

    # Load input kwargs from the initial state job using io
    kwargs_array = io.recover_full_calculation_arguments(full_path=kwargs_file)
    input_kwargs = kwargs_array[job_index]

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

    # Use last dimension of gr to get time grid size (works for both full and reduced states)
    # For full state: shape is (2, 2, N_t, N_t), so shape[-1] = N_t
    # For reduced state: shape is (2, 2, 1, N_t), so shape[-1] = N_t
    time_sampling = state.gr.data.shape[-1]

    grid_params = {'time_sampling': time_sampling, 'time_duration': state.T_max, 'eta': eta}

    # If state has dt stored (e.g., from reduced state), include it
    if hasattr(state, 'dt') and state.dt is not None:
        grid_params['dt'] = state.dt

    return grid_params


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
    gk_errors, gk_totals, gk_components = state.check_keldysh_normalization(-1, evolution.thermal_dist, evolution.thermal_integral)

    # Compute max error per Pauli channel from totals
    # gr_totals and gk_totals have shape (4, N_t) for 4 Pauli components
    gr_max_errors_per_channel = np.zeros(4)
    gk_max_errors_per_channel = np.zeros(4)
    gr_max_error_indices = np.zeros(4, dtype=int)
    gk_max_error_indices = np.zeros(4, dtype=int)

    # Extract error at last element [-1]
    dt = evolution.time_grid[1] - evolution.time_grid[0]
    gr_last_element_errors = np.zeros(4, dtype=complex)
    gk_last_element_errors = np.zeros(4, dtype=complex)

    print("\n" + "="*70)
    print("NORMALIZATION CHECK - Maximum Error Indices")
    print("="*70)

    for pauli_idx in range(4):
        # gr_totals and gk_totals already contain the errors, not raw values
        error_array_gr = np.abs(gr_totals[pauli_idx, :])
        gr_max_error_indices[pauli_idx] = np.argmax(error_array_gr)
        gr_max_errors_per_channel[pauli_idx] = error_array_gr[gr_max_error_indices[pauli_idx]]

        # Error at last element [-1] for g^R
        gr_last_element_errors[pauli_idx] = gr_totals[pauli_idx, -1]

        # For gk: compute error from the totals
        error_array_gk = np.abs(gk_totals[pauli_idx, :])
        gk_max_error_indices[pauli_idx] = np.argmax(error_array_gk)
        gk_max_errors_per_channel[pauli_idx] = error_array_gk[gk_max_error_indices[pauli_idx]]

        # Error at last element [-1] for g^K
        gk_last_element_errors[pauli_idx] = gk_totals[pauli_idx, -1]

        # Print results for this Pauli component
        print(f"\nτ_{pauli_idx} component:")
        print(f"  g^R normalization: max_error = {gr_max_errors_per_channel[pauli_idx]:.4e} at index {gr_max_error_indices[pauli_idx]}")
        print(f"  g^R normalization: error_at_last_element = {gr_last_element_errors[pauli_idx]:.4e}")
        print(f"  g^K normalization: max_error = {gk_max_errors_per_channel[pauli_idx]:.4e} at index {gk_max_error_indices[pauli_idx]}")
        print(f"  g^K normalization: error_at_last_element = {gk_last_element_errors[pauli_idx]:.4e}")

    print("="*70 + "\n")

    # Analyze τ₀ component contributions
    print("\n" + "="*70)
    print("g^K NORMALIZATION - τ₀ COMPONENT ANALYSIS")
    print("="*70)
    pauli_idx = 0  # τ₀ channel

    # Use specific element: t=-1 (last), t'=-2 (second-to-last)
    t_idx = -2

    print(f"\nContributions to τ₀ channel at (t=-1, t'={evolution.time_grid[t_idx]:.2f}):")
    print(f"  Total:                     {gk_totals[pauli_idx, t_idx]:.6e}")
    print(f"  Commutator [τ₃,g^K]:       {gk_components['commutator'][pauli_idx, t_idx]:.6e}")
    print(f"  ∫g^R⊗g^K (pure):           {gk_components['gr_gk_conv_pure'][pauli_idx, t_idx]:.6e}")
    print(f"  ∫g^K⊗g^A (pure):           {gk_components['gk_ga_conv_pure'][pauli_idx, t_idx]:.6e}")
    print(f"  Thermal (gr@f)*τ₃*2:       {gk_components['thermal_gr'][pauli_idx, t_idx]:.6e}")
    print(f"  Thermal 2*τ₃*(f@ga):       {gk_components['thermal_ga'][pauli_idx, t_idx]:.6e}")

    print(f"\nMax absolute values over all t':")
    print(f"  Total:                     {np.max(np.abs(gk_totals[pauli_idx, :])):.6e}")
    print(f"  Commutator:                {np.max(np.abs(gk_components['commutator'][pauli_idx, :])):.6e}")
    print(f"  ∫g^R⊗g^K (pure):           {np.max(np.abs(gk_components['gr_gk_conv_pure'][pauli_idx, :])):.6e}")
    print(f"  ∫g^K⊗g^A (pure):           {np.max(np.abs(gk_components['gk_ga_conv_pure'][pauli_idx, :])):.6e}")
    print(f"  Thermal (gr@f)*τ₃*2:       {np.max(np.abs(gk_components['thermal_gr'][pauli_idx, :])):.6e}")
    print(f"  Thermal 2*τ₃*(f@ga):       {np.max(np.abs(gk_components['thermal_ga'][pauli_idx, :])):.6e}")
    print("="*70 + "\n")

    _plot_normalization(gr_totals, 'gr', evolution.time_grid, save_plot, save_dir, timestamp)
    _plot_normalization(gk_totals, 'gk', evolution.time_grid, save_plot, save_dir, timestamp)

    # Plot individual term contributions to τ₁ channel - COMMENTED OUT
    # _plot_gk_components(gk_components, gk_totals, evolution.time_grid, pauli_channel=0, save_plot=save_plot, save_dir=save_dir, timestamp=timestamp)

    return {'gr_max_error': gr_max_errors_per_channel, 'gk_max_error': gk_max_errors_per_channel,
            'gr_totals': gr_totals, 'gk_totals': gk_totals, 'gk_components': gk_components}


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
    max_error_indices = np.zeros(4, dtype=int)
    last_element_errors = np.zeros(4, dtype=complex)

    # Get time step
    dt = evolution.time_grid[1] - evolution.time_grid[0]

    print("\n" + "="*70)
    print("FDT CHECK - Maximum Error Indices")
    print("="*70)

    for pauli_idx in range(4):
        error_pauli = error_row.trace(pauli_idx) / 2
        if error_pauli.ndim == 2:
            error_pauli = error_pauli[0, :]

        error_array = np.abs(error_pauli)
        max_error_indices[pauli_idx] = np.argmax(error_array)
        max_errors_per_channel[pauli_idx] = error_array[max_error_indices[pauli_idx]]

        # Error at last element [-1]
        last_element_errors[pauli_idx] = error_pauli[-1]

        # Print results for this Pauli component
        print(f"\nτ_{pauli_idx} component:")
        print(f"  max_error = {max_errors_per_channel[pauli_idx]:.4e} at index {max_error_indices[pauli_idx]}")
        print(f"  error_at_last_element = {np.real(last_element_errors[pauli_idx]):.4e} + {np.imag(last_element_errors[pauli_idx]):.4e}j")

    print("="*70 + "\n")

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

    # Create energy grid
    # Standard FFT frequencies
    freq = np.fft.fftfreq(N_t, d=dt)  # Frequency in 1/time units
    # Convert to angular frequency: ω = 2πf
    energy_grid = 2 * np.pi * freq
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
        lines = io.read_contents_readable_file(timestamp)
        job_no = io.recover_job_no(lines)
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
    axes[0].set_title(rf'Real($\Delta$) - {title_suffix}', fontsize=13)
    if plot_all:
        axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'Time $t$', fontsize=12)
    axes[1].set_ylabel(r'Im($\Delta(t)$)', fontsize=12)
    axes[1].set_title(rf'Imag($\Delta$) - {title_suffix}', fontsize=13)
    if plot_all:
        axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel(r'Time $t$', fontsize=12)
    axes[2].set_ylabel(r'$|\Delta(t)|$', fontsize=12)
    axes[2].set_title(rf'$|\Delta|$ - {title_suffix}', fontsize=13)
    if plot_all:
        axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        job_str = f'job{job_index}' if job_index is not None else 'all_jobs'
        plt.savefig(os.path.join(save_dir, f'gap_vs_time_{timestamp}_{job_str}.png'), dpi=150, bbox_inches='tight')

    plt.show()


def extract_conductivity(timestamp, job_index, threshold=1e-6, save_plot=False, save_dir='analysis_plots'):
    """
    Extract and plot optical conductivity σ(ω) = j(ω)/A(ω) from time-domain data.

    Computes the optical conductivity by taking the ratio of Fourier-transformed
    current density to the vector potential. Filters out frequencies where the
    vector potential is below threshold to avoid numerical instabilities.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index
        threshold: Minimum |A(ω)| value for valid conductivity (default 1e-6)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots

    Returns:
        dict: {
            'omega_grid': array - frequency grid in 1/time units
            'sigma': array - complex conductivity σ(ω)
        }
        Returns None if vector potential is zero (equilibrium case)
    """
    # Load data
    input_kwargs, save_data = load_job_data(timestamp, job_index)

    # Check if vector potential is non-zero
    vector_potentials = save_data['vector_potentials']
    if np.allclose(vector_potentials, 0):
        print(f"Vector potential is zero for job {job_index}. Conductivity extraction requires non-zero driving.")
        return None

    # Extract current and time grid
    currents = save_data['currents']

    if 'times' in save_data:
        times = save_data['times']
    else:
        state = save_data['final_state']
        N_t = len(currents)
        dt = state.dt
        times = np.linspace(0, N_t * dt, N_t)

    dt = times[1] - times[0]
    N_t = len(times)

    # Fourier transform current: j(ω)
    j_omega = np.fft.fft(currents)
    j_omega = np.fft.fftshift(j_omega)

    # Fourier transform vector potential: A(ω)
    A_omega = np.fft.fft(vector_potentials)
    A_omega = np.fft.fftshift(A_omega)

    # Construct frequency grid
    freq = np.fft.fftfreq(N_t, d=dt)
    omega_grid = 2 * np.pi * freq
    omega_grid = np.fft.fftshift(omega_grid)

    # Compute conductivity with filtering
    sigma = np.zeros_like(j_omega, dtype=complex)
    mask = np.abs(A_omega) > threshold
    sigma[mask] = j_omega[mask] / A_omega[mask]

    # Find frequency range where A is significant
    omega_indices = np.where(mask)[0]
    if len(omega_indices) > 0:
        omega_min = omega_grid[omega_indices[0]]
        omega_max = omega_grid[omega_indices[-1]]
    else:
        omega_min = omega_max = None

    # Print information
    print(f"\n{'='*60}")
    print(f"Conductivity Extraction (timestamp={timestamp}, job_index={job_index}):")
    print(f"{'='*60}")
    print(f"  Max |A(ω)|: {np.max(np.abs(A_omega)):.3e}")
    print(f"  Threshold: {threshold:.3e}")
    print(f"  Valid frequency range: [{omega_min:.3f}, {omega_max:.3f}]" if omega_min is not None else "  No valid frequencies")
    print(f"  Number of valid frequency points: {np.sum(mask)}/{N_t}")
    print(f"{'='*60}\n")

    # Plot conductivity
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Real part
    axes[0].plot(omega_grid, np.real(sigma), 'b-', linewidth=2, label=r'Re($\sigma(\omega)$)')
    axes[0].set_xlabel(r'Frequency $\omega$', fontsize=12)
    axes[0].set_ylabel(r'Re($\sigma(\omega)$)', fontsize=12)
    axes[0].set_title(r'Real part of conductivity $\sigma(\omega) = j(\omega)/A(\omega)$', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Mark valid frequency range with pale gray dashed lines
    if omega_min is not None:
        axes[0].axvline(omega_min, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='Valid range')
        axes[0].axvline(omega_max, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)

    # Imaginary part
    axes[1].plot(omega_grid, np.imag(sigma), 'r-', linewidth=2, label=r'Im($\sigma(\omega)$)')
    axes[1].set_xlabel(r'Frequency $\omega$', fontsize=12)
    axes[1].set_ylabel(r'Im($\sigma(\omega)$)', fontsize=12)
    axes[1].set_title(r'Imaginary part of conductivity $\sigma(\omega) = j(\omega)/A(\omega)$', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    # Mark valid frequency range with pale gray dashed lines
    if omega_min is not None:
        axes[1].axvline(omega_min, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='Valid range')
        axes[1].axvline(omega_max, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'conductivity_{timestamp}_job{job_index}.png'), dpi=150, bbox_inches='tight')

    plt.show()

    return {'omega_grid': omega_grid, 'sigma': sigma}


def plot_current(timestamp, job_index=None, save_plot=False, save_dir='analysis_plots'):
    """
    Plot current and vector potential vs time for all jobs or a specific job.

    Displays input parameters before plotting.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index (default None plots all jobs in folder)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
    """
    # Determine which jobs to plot
    if job_index is None:
        lines = io.read_contents_readable_file(timestamp)
        job_no = io.recover_job_no(lines)
        job_indices = range(job_no)
        plot_all = True
    else:
        job_indices = [job_index]
        plot_all = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for job_idx in job_indices:
        input_kwargs, save_data = load_job_data(timestamp, job_idx)
        currents = save_data['currents']
        vector_potentials = save_data['vector_potentials']

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
        # Plot Real part of current
        axes[0].plot(times, np.real(currents), linewidth=2, label=label, alpha=0.7, color=f'C{job_idx}')

        # Plot vector potential (real part if complex, otherwise as is)
        if np.iscomplexobj(vector_potentials):
            A_values = np.real(vector_potentials)
        else:
            A_values = vector_potentials

        # Compute dA/dt using gradient
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        dA_dt = np.gradient(A_values, dt)

        # Plot A(t) on left y-axis
        axes[1].plot(times, A_values, linewidth=2, label=label, alpha=0.7, color=f'C{job_idx}')

    # Configure current axis
    title_suffix = 'All Jobs' if plot_all else f'Job {job_index}'

    axes[0].set_xlabel(r'Time $t$', fontsize=12)
    axes[0].set_ylabel(r'Re($J(t)$)', fontsize=12)
    axes[0].set_title(f'Current - {title_suffix}', fontsize=13)
    if plot_all:
        axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Configure vector potential axis (left y-axis)
    axes[1].set_xlabel(r'Time $t$', fontsize=12)
    axes[1].set_ylabel(r'$A(t)$', fontsize=12, color='C0')
    axes[1].set_title(f'Vector Potential and Derivative - {title_suffix}', fontsize=13)
    axes[1].tick_params(axis='y', labelcolor='C0')
    axes[1].grid(True, alpha=0.3)

    # Create second y-axis for dA/dt
    ax_right = axes[1].twinx()

    # Plot dA/dt for all jobs on right y-axis
    for job_idx in job_indices:
        input_kwargs, save_data = load_job_data(timestamp, job_idx)
        vector_potentials = save_data['vector_potentials']

        if 'times' in save_data:
            times = save_data['times']
        else:
            state = save_data['final_state']
            N_t = len(save_data['currents'])
            dt_val = state.dt
            times = np.linspace(0, N_t * dt_val, N_t)

        if np.iscomplexobj(vector_potentials):
            A_values = np.real(vector_potentials)
        else:
            A_values = vector_potentials

        dt_val = times[1] - times[0] if len(times) > 1 else 1.0
        dA_dt = np.gradient(A_values, dt_val)

        label_deriv = f'Job {job_idx} ' + r'($\partial A/\partial t$)' if plot_all else r'$\partial A/\partial t$'
        ax_right.plot(times, dA_dt, linewidth=2, linestyle='--', label=label_deriv, alpha=0.7, color='red')

    ax_right.set_ylabel(r'$\partial A/\partial t$', fontsize=12, color='red')
    ax_right.tick_params(axis='y', labelcolor='red')

    # Combine legends if plotting all jobs
    if plot_all:
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
    else:
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        job_str = f'job{job_index}' if job_index is not None else 'all_jobs'
        plt.savefig(os.path.join(save_dir, f'current_vs_time_{timestamp}_{job_str}.png'), dpi=150, bbox_inches='tight')

    plt.show()


def plot_equilibrated_gap_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir='analysis_plots', combine_timestamps=False, x_external=None, y_external=None, labels_external=None, add_point_labels=False):
    """
    Plot equilibrated gap vs parameter for multiple timestamps.

    Computes average gap and standard deviation over last n_average timesteps.

    Args:
        timestamps: List of timestamps
        parameter: Parameter to plot against. Options:
                   - 'temperature': System temperature
                   - 'vector_potential': Max absolute value of vector potential
                   - 'field_params:KEY': Specific field parameter (e.g., 'field_params:amplitude')
        n_average: Number of final timesteps to average (default 100)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
        combine_timestamps: If True, combine all timestamps into single curve (default False)
        x_external: Optional external x-values for comparison. Can be:
                   - Single array: plots one external dataset
                   - List of arrays: plots multiple external datasets
        y_external: Optional external y-values for comparison. Can be:
                   - Single array: plots one external dataset
                   - List of arrays: plots multiple external datasets (must match x_external length)
        labels_external: Optional labels for external datasets. Can be:
                   - Single string: label for single external dataset
                   - List of strings: labels for multiple external datasets (must match x_external length)
                   - If not provided, defaults to 'External data' or 'External 1', 'External 2', etc.
        add_point_labels: If True, annotate each datapoint with its (x, y) values (default False)

    Returns:
        dict: {'parameter_values': array, 'gap_means': array, 'gap_errors': array}
    """
    # Store data per timestamp for separate plotting
    timestamp_data = {}
    all_parameter_values = []
    all_gap_means = []
    all_gap_errors = []

    for timestamp in timestamps:
        lines = io.read_contents_readable_file(timestamp)
        job_no = io.recover_job_no(lines)

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
            elif parameter.startswith('field_params:'):
                # Extract field parameter name after colon
                field_param_name = parameter.split(':', 1)[1]
                param_value = input_kwargs['field_params'][field_param_name]
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

        # Add point labels if requested
        if add_point_labels:
            for x, y in zip(all_parameter_values[sort_idx], np.abs(all_gap_means[sort_idx])):
                ax.annotate(f'({x:.3f}, {y:.3f})', xy=(x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, alpha=0.7)
    else:
        # New behavior: plot each timestamp separately
        for i, (timestamp, data) in enumerate(timestamp_data.items()):
            label = f'{timestamp}' if len(timestamps) > 1 else 'Data'
            ax.errorbar(data['parameter_values'], np.abs(data['gap_means']),
                       yerr=data['gap_errors'], fmt='o-', capsize=5,
                       linewidth=2, markersize=8, label=label, alpha=0.7)

            # Add point labels if requested
            if add_point_labels:
                for x, y in zip(data['parameter_values'], np.abs(data['gap_means'])):
                    ax.annotate(f'({x:.3f}, {y:.3f})', xy=(x, y), xytext=(5, 5),
                               textcoords='offset points', fontsize=8, alpha=0.7)

    # Plot external data if provided
    if x_external is not None and y_external is not None:
        # Check if x_external and y_external are lists of arrays
        if isinstance(x_external, list) and isinstance(y_external, list):
            # Plot multiple external datasets
            markers = ['x', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h']
            linestyles = ['--', '-.', ':', '--', '-.']
            for idx, (x_ext, y_ext) in enumerate(zip(x_external, y_external)):
                marker = markers[idx % len(markers)]
                linestyle = linestyles[idx % len(linestyles)]
                if labels_external is not None:
                    if isinstance(labels_external, list):
                        label = labels_external[idx]
                    else:
                        label = labels_external
                else:
                    label = f'External {idx+1}' if len(x_external) > 1 else 'External data'
                ax.plot(x_ext, np.abs(y_ext), marker=marker, linestyle=linestyle,
                       linewidth=2, markersize=10, label=label, alpha=0.8)
        else:
            # Single external dataset (backward compatible)
            if labels_external is not None:
                label = labels_external if isinstance(labels_external, str) else labels_external[0]
            else:
                label = 'External data'
            ax.plot(x_external, np.abs(y_external), 'kx--', linewidth=2, markersize=10,
                   label=label, alpha=0.8)

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


def plot_equilibrated_current_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir='analysis_plots', combine_timestamps=False, x_external=None, y_external=None, labels_external=None, add_point_labels=False):
    """
    Plot equilibrated current vs parameter for multiple timestamps.

    Computes average current and standard deviation over last n_average timesteps.

    Args:
        timestamps: List of timestamps
        parameter: Parameter to plot against. Options:
                   - 'temperature': System temperature
                   - 'vector_potential': Max absolute value of vector potential
                   - 'field_params:KEY': Specific field parameter (e.g., 'field_params:amplitude')
        n_average: Number of final timesteps to average (default 100)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots
        combine_timestamps: If True, combine all timestamps into single curve (default False)
        x_external: Optional external x-values for comparison. Can be:
                   - Single array: plots one external dataset
                   - List of arrays: plots multiple external datasets
        y_external: Optional external y-values for comparison. Can be:
                   - Single array: plots one external dataset
                   - List of arrays: plots multiple external datasets (must match x_external length)
        labels_external: Optional labels for external datasets. Can be:
                   - Single string: label for single external dataset
                   - List of strings: labels for multiple external datasets (must match x_external length)
                   - If not provided, defaults to 'External data' or 'External 1', 'External 2', etc.
        add_point_labels: If True, annotate each datapoint with its (x, y) values (default False)

    Returns:
        dict: {'parameter_values': array, 'current_means': array, 'current_errors': array}
    """
    # Store data per timestamp for separate plotting
    timestamp_data = {}
    all_parameter_values = []
    all_current_means = []
    all_current_errors = []

    for timestamp in timestamps:
        lines = io.read_contents_readable_file(timestamp)
        job_no = io.recover_job_no(lines)

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
            elif parameter.startswith('field_params:'):
                # Extract field parameter name after colon
                field_param_name = parameter.split(':', 1)[1]
                param_value = input_kwargs['field_params'][field_param_name]
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

        # Add point labels if requested
        if add_point_labels:
            for x, y in zip(all_parameter_values[sort_idx], np.abs(all_current_means[sort_idx])):
                ax.annotate(f'({x:.3f}, {y:.3f})', xy=(x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, alpha=0.7)
    else:
        # New behavior: plot each timestamp separately
        for i, (timestamp, data) in enumerate(timestamp_data.items()):
            label = f'{timestamp}' if len(timestamps) > 1 else 'Data'
            ax.errorbar(data['parameter_values'], np.abs(data['current_means']),
                       yerr=data['current_errors'], fmt='o-', capsize=5,
                       linewidth=2, markersize=8, label=label, alpha=0.7)

            # Add point labels if requested
            if add_point_labels:
                for x, y in zip(data['parameter_values'], np.abs(data['current_means'])):
                    ax.annotate(f'({x:.3f}, {y:.3f})', xy=(x, y), xytext=(5, 5),
                               textcoords='offset points', fontsize=8, alpha=0.7)

    # Plot external data if provided
    if x_external is not None and y_external is not None:
        # Check if x_external and y_external are lists of arrays
        if isinstance(x_external, list) and isinstance(y_external, list):
            # Plot multiple external datasets
            markers = ['x', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h']
            linestyles = ['--', '-.', ':', '--', '-.']
            for idx, (x_ext, y_ext) in enumerate(zip(x_external, y_external)):
                marker = markers[idx % len(markers)]
                linestyle = linestyles[idx % len(linestyles)]
                if labels_external is not None:
                    if isinstance(labels_external, list):
                        label = labels_external[idx]
                    else:
                        label = labels_external
                else:
                    label = f'External {idx+1}' if len(x_external) > 1 else 'External data'
                ax.plot(x_ext, np.abs(y_ext), marker=marker, linestyle=linestyle,
                       linewidth=2, markersize=10, label=label, alpha=0.8)
        else:
            # Single external dataset (backward compatible)
            if labels_external is not None:
                label = labels_external if isinstance(labels_external, str) else labels_external[0]
            else:
                label = 'External data'
            ax.plot(x_external, np.abs(y_external), 'kx--', linewidth=2, markersize=10,
                   label=label, alpha=0.8)

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

    print(f"\n{'='*70}")
    print(f"PLOTTING {gf_type.upper()} NORMALIZATION - Error Values")
    print(f"{'='*70}")

    for pauli_idx in range(4):
        ax = axes.flat[pauli_idx]
        component = totals[pauli_idx, :]

        # Print what we're plotting (totals already contain errors)
        print(f"\nτ_{pauli_idx} component:")
        print(f"  Plotting range: [{np.real(component).min():.4e}, {np.real(component).max():.4e}]")
        print(f"  Mean error: {np.real(component).mean():.4e}")
        print(f"  Std deviation: {np.real(component).std():.4e}")
        print(f"  Max error: {np.abs(component).max():.4e}")

        ax.plot(time_grid, np.real(component), 'b-', linewidth=2, label='Real', alpha=0.7, marker='o', markevery=10, markersize=4)
        ax.plot(time_grid, np.imag(component), 'r-', linewidth=2, label='Imag', alpha=0.7, marker='s', markevery=10, markersize=4)
        ax.set_xlabel(r"$t_2$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'$g^{{{gf_type}}}$: {pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    print(f"{'='*70}\n")

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


def _plot_gk_components(gk_components, gk_totals, time_grid, pauli_channel=0, save_plot=False, save_dir='analysis_plots', timestamp=''):
    """Plot individual term contributions to specified Pauli channel in g^K normalization."""
    import matplotlib.pyplot as plt

    pauli_idx = pauli_channel

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract components for this Pauli channel
    total = gk_totals[pauli_idx, :]
    comm = gk_components['commutator'][pauli_idx, :]
    gr_gk_pure = gk_components['gr_gk_conv_pure'][pauli_idx, :]
    gk_ga_pure = gk_components['gk_ga_conv_pure'][pauli_idx, :]
    therm_gr = gk_components['thermal_gr'][pauli_idx, :]
    therm_ga = gk_components['thermal_ga'][pauli_idx, :]

    # Plot 1: Real parts of all terms
    ax = axes[0, 0]
    ax.plot(time_grid, np.real(total), 'k-', linewidth=2, label='Total', alpha=0.8, marker='o', markevery=15, markersize=5)
    ax.plot(time_grid, np.real(comm), 'b-', linewidth=1.5, label=r'Commutator', alpha=0.7, marker='s', markevery=15, markersize=4)
    ax.plot(time_grid, np.real(gr_gk_pure), 'r-', linewidth=1.5, label=r'$g^R \otimes g^K$ (pure)', alpha=0.7, marker='^', markevery=15, markersize=4)
    ax.plot(time_grid, np.real(gk_ga_pure), 'g-', linewidth=1.5, label=r'$g^K \otimes g^A$ (pure)', alpha=0.7, marker='v', markevery=15, markersize=4)
    ax.plot(time_grid, np.real(therm_gr), 'm--', linewidth=1.5, label=r'Thermal $gr \otimes f$', alpha=0.7, marker='d', markevery=15, markersize=4)
    ax.plot(time_grid, np.real(therm_ga), 'c--', linewidth=1.5, label=r'Thermal $f \otimes ga$', alpha=0.7, marker='*', markevery=15, markersize=5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(rf'Re[$\tau_{pauli_idx}$ component]', fontsize=12)
    ax.set_title(r'Real Part - Individual Terms', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_grid[0], 0])

    # Plot 2: Imaginary parts of all terms
    ax = axes[0, 1]
    ax.plot(time_grid, np.imag(total), 'k-', linewidth=2, label='Total', alpha=0.8, marker='o', markevery=15, markersize=5)
    ax.plot(time_grid, np.imag(comm), 'b-', linewidth=1.5, label=r'Commutator', alpha=0.7, marker='s', markevery=15, markersize=4)
    ax.plot(time_grid, np.imag(gr_gk_pure), 'r-', linewidth=1.5, label=r'$g^R \otimes g^K$ (pure)', alpha=0.7, marker='^', markevery=15, markersize=4)
    ax.plot(time_grid, np.imag(gk_ga_pure), 'g-', linewidth=1.5, label=r'$g^K \otimes g^A$ (pure)', alpha=0.7, marker='v', markevery=15, markersize=4)
    ax.plot(time_grid, np.imag(therm_gr), 'm--', linewidth=1.5, label=r'Thermal $gr \otimes f$', alpha=0.7, marker='d', markevery=15, markersize=4)
    ax.plot(time_grid, np.imag(therm_ga), 'c--', linewidth=1.5, label=r'Thermal $f \otimes ga$', alpha=0.7, marker='*', markevery=15, markersize=5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(rf'Im[$\tau_{pauli_idx}$ component]', fontsize=12)
    ax.set_title(r'Imaginary Part - Individual Terms', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_grid[0], 0])

    # Plot 3: Absolute value (log scale)
    ax = axes[1, 0]
    ax.semilogy(time_grid, np.abs(total) + 1e-20, 'k-', linewidth=2, label='Total', alpha=0.8, marker='o', markevery=15, markersize=5)
    ax.semilogy(time_grid, np.abs(comm) + 1e-20, 'b-', linewidth=1.5, label='Commutator', alpha=0.7, marker='s', markevery=15, markersize=4)
    ax.semilogy(time_grid, np.abs(gr_gk_pure) + 1e-20, 'r-', linewidth=1.5, label=r'$g^R \otimes g^K$ (pure)', alpha=0.7, marker='^', markevery=15, markersize=4)
    ax.semilogy(time_grid, np.abs(gk_ga_pure) + 1e-20, 'g-', linewidth=1.5, label=r'$g^K \otimes g^A$ (pure)', alpha=0.7, marker='v', markevery=15, markersize=4)
    ax.semilogy(time_grid, np.abs(therm_gr) + 1e-20, 'm--', linewidth=1.5, label=r'Thermal $gr \otimes f$', alpha=0.7, marker='d', markevery=15, markersize=4)
    ax.semilogy(time_grid, np.abs(therm_ga) + 1e-20, 'c--', linewidth=1.5, label=r'Thermal $f \otimes ga$', alpha=0.7, marker='*', markevery=15, markersize=5)
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(rf'|$\tau_{pauli_idx}$ component|', fontsize=12)
    ax.set_title(r'Magnitude (log scale)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([time_grid[0], 0])

    # Plot 4: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"tau_{pauli_idx} Channel Statistics:\n\n"
    stats_text += f"Total:\n"
    stats_text += f"  Re: mean={np.real(total).mean():.4e}, max={np.abs(np.real(total)).max():.4e}\n"
    stats_text += f"  Im: mean={np.imag(total).mean():.4e}, max={np.abs(np.imag(total)).max():.4e}\n\n"

    stats_text += f"Commutator:\n"
    stats_text += f"  Re: mean={np.real(comm).mean():.4e}, max={np.abs(np.real(comm)).max():.4e}\n"
    stats_text += f"  Im: mean={np.imag(comm).mean():.4e}, max={np.abs(np.imag(comm)).max():.4e}\n\n"

    stats_text += f"gR conv gK (pure):\n"
    stats_text += f"  Re: mean={np.real(gr_gk_pure).mean():.4e}, max={np.abs(np.real(gr_gk_pure)).max():.4e}\n"
    stats_text += f"  Im: mean={np.imag(gr_gk_pure).mean():.4e}, max={np.abs(np.imag(gr_gk_pure)).max():.4e}\n\n"

    stats_text += f"gK conv gA (pure):\n"
    stats_text += f"  Re: mean={np.real(gk_ga_pure).mean():.4e}, max={np.abs(np.real(gk_ga_pure)).max():.4e}\n"
    stats_text += f"  Im: mean={np.imag(gk_ga_pure).mean():.4e}, max={np.abs(np.imag(gk_ga_pure)).max():.4e}\n\n"

    stats_text += f"Thermal gr@f:\n"
    stats_text += f"  Re: mean={np.real(therm_gr).mean():.4e}, max={np.abs(np.real(therm_gr)).max():.4e}\n"
    stats_text += f"  Im: mean={np.imag(therm_gr).mean():.4e}, max={np.abs(np.imag(therm_gr)).max():.4e}\n\n"

    stats_text += f"Thermal f@ga:\n"
    stats_text += f"  Re: mean={np.real(therm_ga).mean():.4e}, max={np.abs(np.real(therm_ga)).max():.4e}\n"
    stats_text += f"  Im: mean={np.imag(therm_ga).mean():.4e}, max={np.abs(np.imag(therm_ga)).max():.4e}\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', usetex=False,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'gk_tau{pauli_idx}_components_{timestamp}.png'), dpi=150, bbox_inches='tight')

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


def plot_energy_time_representation(timestamp, job_index, green_function_type='f',
                                     pauli_components=None, plot_every=1, xlim=(-15, 15),
                                     x_external=None, y_external=None, labels_external=None,
                                     save_plot=False, save_dir='analysis_plots'):
    """
    Plot Green's function in energy-time representation.

    Creates a 2-subplot figure showing energy cuts at different times.
    Each time slice is plotted with progressively darker shades of blue
    (darker = later time).

    Args:
        timestamp: Timestamp folder name
        job_index: Job index
        green_function_type: Type of Green's function to plot ('gr', 'gk', or 'f')
        pauli_components: Tuple of two Pauli indices to plot. If None, uses smart defaults:
                         - For 'f': (0, 3) - particle density and particle-hole difference
                         - For 'gr' or 'gk': (2, 3) - anomalous and normal components
                         0=tau_0 (identity), 1=tau_1 (tau_x), 2=tau_2 (tau_y), 3=tau_3 (tau_z)
        plot_every: Plot every N-th time slice (default 1, plot all)
        xlim: Tuple (xmin, xmax) for energy axis limits (default (-15, 15))
              Set to None to use full energy range
        x_external: Optional external x-values (energy) for comparison. Can be:
                   - Single array: plots one external dataset
                   - List of arrays: plots multiple external datasets
        y_external: Optional external y-values for comparison. Can be:
                   - Single array: plots one external dataset
                   - List of arrays: plots multiple external datasets (must match x_external length)
                   - Should be dict with keys matching pauli_components if plotting specific components
        labels_external: Optional labels for external datasets. Can be:
                   - Single string: label for single external dataset
                   - List of strings: labels for multiple external datasets
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots

    Returns:
        dict: {
            'energy_grid': energy array,
            'time_indices': array of plotted time indices,
            'data': list of NambuKeldyshTensor objects at each time
        }

    Example:
        # Plot occupation function f (uses default tau_0 and tau_3 components)
        plot_energy_time_representation('20260615_143022', 0, green_function_type='f',
                                       plot_every=5)

        # Plot retarded Green's function (uses default tau_2 and tau_3)
        plot_energy_time_representation('20260615_143022', 0, green_function_type='gr',
                                       plot_every=10)

        # Plot with external data comparison (e.g., thermal distribution)
        energy_ext = np.linspace(-10, 10, 100)
        f_thermal = -1j * np.tanh(energy_ext / (2 * temperature))
        plot_energy_time_representation('20260615_143022', 0, green_function_type='f',
                                       x_external=energy_ext,
                                       y_external=np.imag(f_thermal),
                                       labels_external='Thermal',
                                       plot_every=5)

        # Override defaults - plot custom components
        plot_energy_time_representation('20260615_143022', 0, green_function_type='f',
                                       pauli_components=(1, 2), plot_every=5)
    """
    # Load data
    input_kwargs, save_data = load_job_data(timestamp, job_index)

    # Map green_function_type to data key
    data_key_map = {
        'gr': 'gr_energy_time',
        'gk': 'gk_energy_time',
        'f': 'f_energy_time'
    }

    if green_function_type not in data_key_map:
        raise ValueError(f"green_function_type must be 'gr', 'gk', or 'f', got '{green_function_type}'")

    data_key = data_key_map[green_function_type]

    # Set smart defaults for pauli_components if not provided
    if pauli_components is None:
        if green_function_type == 'f':
            pauli_components = (0, 3)
        else:
            pauli_components = (2, 3)

    # Check if data exists
    if data_key not in save_data:
        raise ValueError(f"No {green_function_type} energy-time data found. "
                        f"Simulation must be run with track_every_n parameter.")

    # Extract data
    energy_time_list = save_data[data_key]
    time_indices = save_data['energy_time_indices']

    # Get actual times from save_data
    if 'times' in save_data:
        times = save_data['times']
        time_values = times[time_indices]
    else:
        time_values = time_indices

    # Select which times to plot
    plot_indices = np.arange(0, len(energy_time_list), plot_every)

    # Extract energy grid (same for all times)
    energy_grid = energy_time_list[0]['energy_grid']

    # Pauli component labels
    pauli_labels = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']

    # Create colormaps (darker = later time)
    # Blues for real part, Reds for imaginary part
    from matplotlib import cm
    n_plots = len(plot_indices)
    blues = cm.get_cmap('Blues', n_plots + 2)
    reds = cm.get_cmap('Reds', n_plots + 2)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Print information
    print(f"\n{'='*60}")
    print(f"Energy-Time Plot (timestamp={timestamp}, job_index={job_index}):")
    print(f"{'='*60}")
    print(f"  Green's function type: g^{green_function_type}")
    print(f"  Pauli components: {pauli_components}")
    print(f"  Total time slices available: {len(energy_time_list)}")
    print(f"  Plotting every {plot_every} slice(s): {len(plot_indices)} lines")
    print(f"  Time range: [{time_values[0]:.2f}, {time_values[-1]:.2f}]")
    print(f"  Energy grid points: {len(energy_grid)}")
    print(f"{'='*60}\n")

    # Plot both Pauli components
    for subplot_idx, pauli_idx in enumerate(pauli_components):
        ax = axes[subplot_idx]

        # Plot each time slice (both real and imaginary)
        for i, data_idx in enumerate(plot_indices):
            # Extract Green's function at this time
            g_data = energy_time_list[data_idx]
            g_energy = g_data['g_energy']  # NambuKeldyshTensor (2, 2, N_ε)

            # Extract Pauli component
            g_pauli = g_energy.trace(pauli_index=pauli_idx) / 2

            # Extract both real and imaginary parts
            g_real = np.real(g_pauli)
            g_imag = np.imag(g_pauli)

            # Colors: darker for later times (skip first 2 colors to avoid white)
            color_real = blues(i + 2)
            color_imag = reds(i + 2)

            # Time value for label (show only first 5 in legend to avoid clutter)
            time_value = time_values[plot_indices[i]]
            label_real = f'Re, $t = {time_value:.2f}$' if i < 5 else None
            label_imag = f'Im, $t = {time_value:.2f}$' if i < 5 else None

            # Plot real part (blue)
            ax.plot(energy_grid, g_real, '-', color=color_real, linewidth=1.5,
                   alpha=0.8, label=label_real)

            # Plot imaginary part (red)
            ax.plot(energy_grid, g_imag, '-', color=color_imag, linewidth=1.5,
                   alpha=0.8, label=label_imag)

        # Plot external data if provided
        if x_external is not None and y_external is not None:
            # Check if y_external is dict with pauli components as keys
            if isinstance(y_external, dict):
                if pauli_idx in y_external:
                    y_data = y_external[pauli_idx]
                else:
                    y_data = None
            else:
                # Assume same external data for all components
                y_data = y_external

            if y_data is not None:
                # Handle list of external datasets
                if isinstance(x_external, list):
                    markers = ['x', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h']
                    linestyles = ['--', '-.', ':', '--', '-.']
                    y_list = y_data if isinstance(y_data, list) else [y_data]

                    for idx, (x_ext, y_ext) in enumerate(zip(x_external, y_list)):
                        marker = markers[idx % len(markers)]
                        linestyle = linestyles[idx % len(linestyles)]
                        if labels_external is not None:
                            label = labels_external[idx] if isinstance(labels_external, list) else labels_external
                        else:
                            label = f'External {idx+1}' if len(x_external) > 1 else 'External'
                        ax.plot(x_ext, y_ext, marker=marker, linestyle=linestyle,
                               linewidth=2, markersize=8, label=label, alpha=0.8, color='black')
                else:
                    # Single external dataset
                    if labels_external is not None:
                        label = labels_external if isinstance(labels_external, str) else labels_external[0]
                    else:
                        label = 'External'
                    ax.plot(x_external, y_data, 'kx--', linewidth=2, markersize=8,
                           label=label, alpha=0.8)

        # Configure subplot
        ax.set_xlabel(r'Energy $\epsilon$', fontsize=12)
        ax.set_ylabel(f'{pauli_labels[pauli_idx]}', fontsize=12)
        ax.set_title(f'$g^{{{green_function_type}}}$: {pauli_labels[pauli_idx]} (Blue=Re, Red=Im)',
                    fontsize=13)
        ax.grid(True, alpha=0.3)

        # Set x-axis limits if specified
        if xlim is not None:
            ax.set_xlim(xlim)

        # Add legend to first subplot
        if subplot_idx == 0 and (n_plots <= 10 or x_external is not None):
            ax.legend(fontsize=8, loc='best', ncol=2)

    # Overall title
    fig.suptitle(f'Energy-Time Representation: $g^{{{green_function_type}}}$ '
                f'(timestamp={timestamp}, job={job_index})',
                fontsize=14, y=1.02)

    plt.tight_layout()

    # Save if requested
    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'energy_time_{green_function_type}_{timestamp}_job{job_index}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')

    plt.show()

    # Return data for further analysis
    return {
        'energy_grid': energy_grid,
        'time_indices': time_indices[plot_indices],
        'time_values': time_values[plot_indices],
        'data': [energy_time_list[i] for i in plot_indices]
    }


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
        ax.text(0.02, 0.98, rf'max diff: {max_diff:.2e}\nat $t\'$={max_diff_time:.2f}',
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
        ax.text(0.02, 0.98, rf'max diff: {max_diff:.2e}\nat $t\'$={max_diff_time:.2f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'$g^K$ State Evolution: Initial (t=0) vs Final (t=-1)', fontsize=12)
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'state_evolution_gk_{timestamp}.png'), dpi=150, bbox_inches='tight')

    plt.show()
