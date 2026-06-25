"""
Data analysis tools for Keldysh-Usadel simulations.

#* REDUCED STATE COMPATIBILITY (save_full_state=False):
#* ✓ Normalizations, FDT checks, gap/current plots, equilibrium sweeps
#* ✗ Time-translation invariance, state evolution, full Fourier transforms
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle
from demler_tools.file_manager import io, path_management
from demler_tools.data_analysis.image_file_tools import extended_savefig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution

#* Matplotlib LaTeX configuration
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}'


#* ============================================================================
#* DATA LOADING
#* ============================================================================

def load_job_data(timestamp, job_index, running_machine='laptop'):
    """Load raw simulation data (input_kwargs, save_data) using demler_tools."""

    if not hasattr(path_management, 'project_data'):
        path_management.initialize(project_name='psBQP-keldysh')

    simulation_data_path = path_management.default_simulation_data_path(running_machine=running_machine)
    kwargs_file = os.path.join(simulation_data_path, str(timestamp), 'inputs', path_management.args_input_filetag())
    result_file = os.path.join(simulation_data_path, str(timestamp), path_management.relative_path_raw_result_file().format(job_index))

    #* Check archive path for cluster runs if not found
    if not os.path.exists(kwargs_file) or not os.path.exists(result_file):
        if running_machine == 'cluster_euler':
            archive_path = path_management.default_autoarchive_path(running_machine=running_machine)
            if not os.path.exists(kwargs_file):
                kwargs_file = os.path.join(archive_path, str(timestamp), 'inputs', path_management.args_input_filetag())
            if not os.path.exists(result_file):
                result_file = os.path.join(archive_path, str(timestamp), path_management.relative_path_raw_result_file().format(job_index))

    if not os.path.exists(kwargs_file):
        raise FileNotFoundError(f"Kwargs file not found for timestamp {timestamp}")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found for timestamp {timestamp}, job {job_index}")

    kwargs_array = io.recover_full_calculation_arguments(full_path=kwargs_file)
    input_kwargs = kwargs_array[job_index]

    with open(result_file, 'rb') as f:
        save_data = pickle.load(f)

    return input_kwargs, save_data


def load_simulation_data(timestamp, job_index, running_machine='laptop'):
    """
    #* CENTRALIZED LOADER: Returns data + all normalization constants (T_c, J_0, A_0)
    #* Use this for all plotting functions to ensure consistent dimensionless units
    """
    input_kwargs, save_data = load_job_data(timestamp, job_index, running_machine)

    #* Extract system parameters
    T_c = input_kwargs['system_parameters']['critical_temperature']
    gaps = save_data['gaps']
    gap_eq = np.abs(gaps[0])  # Equilibrium gap before driving
    field_type = input_kwargs.get('field_type', 'unknown')

    times = save_data.get('times', None)
    currents = save_data['currents']
    vector_potentials = save_data['vector_potentials']

    #* Current normalization: equilibrium value or max value
    if 'equilibrium_current' in save_data:
        J_0 = np.abs(save_data['equilibrium_current'])
    else:
        J_0 = np.max(np.abs(currents)) if len(currents) > 0 else 1.0
    if J_0 < 1e-12:
        J_0 = 1.0

    J_0 = 1.0
    #* Vector potential normalization: max value
    A_0 = np.max(np.abs(vector_potentials)) if len(vector_potentials) > 0 else 1.0
    if A_0 < 1e-12:
        A_0 = 1.0

    return {
        'input_kwargs': input_kwargs, 'save_data': save_data,
        'T_c': T_c, 'gap_eq': gap_eq, 'J_0': J_0, 'A_0': A_0,
        'field_type': field_type, 'times': times,
        'gaps': gaps, 'currents': currents, 'vector_potentials': vector_potentials
    }


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


def identify_sweep_parameters(timestamp, running_machine='laptop', max_params=10, verbose=True):
    """
    Identify all parameters that vary across jobs in a timestamp.

    Automatically detects parameter sweeps by loading all jobs and comparing
    their input parameters. Works with all nested structures including field_params,
    system_parameters, grid_parameters, etc.

    Args:
        timestamp: Timestamp folder name
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')
        max_params: Maximum number of varying parameters to return (default 10)
        verbose: If True, print detailed information (default True)

    Returns:
        dict: {
            'varying_parameters': List of (param_path, values) tuples,
            'n_jobs': Total number of jobs,
            'constant_parameters': Dict of parameters that are constant across all jobs
        }

    Example:
        >>> sweep_info = identify_sweep_parameters('20240615_143022')
        >>> varying_params = sweep_info['varying_parameters']
        >>> for param_path, values in varying_params:
        >>>     print(f"{param_path}: {values}")
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"IDENTIFYING SWEEP PARAMETERS FOR TIMESTAMP: {timestamp}")
        print(f"{'='*70}\n")

    # Determine number of jobs
    lines = io.read_contents_readable_file(timestamp)
    n_jobs = io.recover_job_no(lines)

    if verbose:
        print(f"Loading {n_jobs} jobs...\n")

    # Load all jobs
    all_kwargs = []
    for job_idx in range(n_jobs):
        input_kwargs, _ = load_job_data(timestamp, job_idx, running_machine=running_machine)
        all_kwargs.append(input_kwargs)

    # Find varying parameters
    varying_params = _find_varying_parameters(all_kwargs, max_params=max_params)

    if verbose:
        if varying_params:
            print(f"Found {len(varying_params)} varying parameter(s):\n")
            for idx, (param_path, values) in enumerate(varying_params):
                print(f"{idx+1}. {param_path}")
                print(f"   Values: {values}")

                # Show value range for numerical parameters
                if all(isinstance(v, (int, float)) for v in values):
                    print(f"   Range: [{min(values):.6g}, {max(values):.6g}]")
                print()
        else:
            print("No varying parameters found - all jobs have identical parameters")

        print(f"{'='*70}\n")

    # Identify constant parameters (from first job, excluding varying ones)
    varying_paths = {param_path for param_path, _ in varying_params}
    all_paths = _explore_all_params(all_kwargs[0])
    constant_params = {}

    for param_path in all_paths:
        if param_path not in varying_paths:
            try:
                value = _extract_param_value(all_kwargs[0], param_path)
                constant_params[param_path] = value
            except (KeyError, TypeError, AttributeError):
                continue

    return {
        'varying_parameters': varying_params,
        'n_jobs': n_jobs,
        'constant_parameters': constant_params
    }


def print_all_parameters(timestamp, job_index=0, running_machine='laptop', show_derived=True):
    """
    Print all simulation parameters in a nicely formatted way.

    Displays input parameters, derived quantities, and state information
    in a hierarchical, easy-to-read format.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index to load (default 0)
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')
        show_derived: If True, also compute and show derived quantities (default True)

    Example:
        print_all_parameters('20240615_143022', job_index=0)
    """
    # Load data
    input_kwargs, save_data = load_job_data(timestamp, job_index, running_machine=running_machine)

    # Compute dt from state
    state = save_data.get('final_state', None)
    dt_computed = None
    if state is not None:
        if hasattr(state, 'dt') and state.dt is not None:
            dt_computed = state.dt
        else:
            time_sampling = state.gr.data.shape[-1]
            T_max = state.T_max
            dt_computed = T_max / (time_sampling - 1)

    # Print header
    print("\n" + "="*80)
    print(f"SIMULATION PARAMETERS")
    print(f"Timestamp: {timestamp}")
    print(f"Job Index: {job_index}")
    print("="*80)

    # Helper function to print nested dictionaries
    def print_dict(d, indent=0, section_name=None):
        """Recursively print dictionary with proper indentation."""
        if section_name:
            print("\n" + " "*indent + f"{'─'*60}")
            print(" "*indent + f"║ {section_name}")
            print(" "*indent + f"{'─'*60}")

        for key, value in d.items():
            if isinstance(value, dict):
                print(" "*indent + f"  {key}:")
                print_dict(value, indent + 4)
            elif isinstance(value, (list, tuple, np.ndarray)):
                if isinstance(value, np.ndarray):
                    if value.size <= 10:
                        # Show small arrays in full
                        print(" "*indent + f"  {key:30s} = {value}")
                    else:
                        # Summarize large arrays
                        print(" "*indent + f"  {key:30s} = array(shape={value.shape}, dtype={value.dtype})")
                        if value.ndim == 1:
                            print(" "*indent + f"  {' '*30}   range=[{np.min(value):.4g}, {np.max(value):.4g}]")
                else:
                    if len(value) <= 5:
                        print(" "*indent + f"  {key:30s} = {value}")
                    else:
                        print(" "*indent + f"  {key:30s} = {type(value).__name__}(length={len(value)})")
            elif isinstance(value, (int, float, complex, bool, str, type(None))):
                # Format numbers nicely
                if isinstance(value, float):
                    if abs(value) < 1e-3 or abs(value) > 1e4:
                        print(" "*indent + f"  {key:30s} = {value:.4e}")
                    else:
                        print(" "*indent + f"  {key:30s} = {value:.6f}")
                elif isinstance(value, complex):
                    print(" "*indent + f"  {key:30s} = {value:.4e}")
                else:
                    print(" "*indent + f"  {key:30s} = {value}")
            else:
                # Unknown type
                print(" "*indent + f"  {key:30s} = {value} (type: {type(value).__name__})")

    # Print all input_kwargs sections
    for key, value in input_kwargs.items():
        if isinstance(value, dict):
            # Add dt to grid_parameters if available
            if key == 'grid_parameters' and dt_computed is not None:
                # Create a copy with dt added
                value_with_dt = value.copy()
                value_with_dt['dt'] = dt_computed
                print_dict(value_with_dt, indent=2, section_name=key.replace('_', ' ').upper())
            else:
                print_dict(value, indent=2, section_name=key.replace('_', ' ').upper())
        else:
            # Top-level non-dict parameters
            if key not in ['system_parameters', 'grid_parameters', 'optimization_parameters',
                          'sigma_scatterings', 'field_params']:
                if isinstance(value, float):
                    if abs(value) < 1e-3 or abs(value) > 1e4:
                        print(f"  {key:30s} = {value:.4e}")
                    else:
                        print(f"  {key:30s} = {value:.6f}")
                else:
                    print(f"  {key:30s} = {value}")

    # Print derived quantities if requested
    if show_derived:
        print("\n" + "─"*80)
        print("║ DERIVED QUANTITIES")
        print("─"*80)

        state = save_data.get('final_state', None)

        if state is not None:
            # Compute dt
            if hasattr(state, 'dt') and state.dt is not None:
                dt = state.dt
            else:
                time_sampling = state.gr.data.shape[-1]
                dt = state.T_max / (time_sampling - 1)

            print(f"  {'Time step (dt)':30s} = {dt:.6e}")
            print(f"  {'State shape':30s} = {state.gr.data.shape}")
            print(f"  {'Number of timesteps':30s} = {state.gr.data.shape[-1]}")

            # BCS ratio
            T_c = input_kwargs['system_parameters']['critical_temperature']
            gaps = save_data.get('gaps', None)
            if gaps is not None:
                gap_initial = np.abs(gaps[0])
                gap_final = np.abs(gaps[-1])
                print(f"  {'Initial gap / T_c':30s} = {gap_initial / T_c:.6f}")
                print(f"  {'Final gap / T_c':30s} = {gap_final / T_c:.6f}")
                print(f"  {'Gap change':30s} = {(gap_final - gap_initial) / gap_initial * 100:.2f}%")

            # Currents
            currents = save_data.get('currents', None)
            if currents is not None and len(currents) > 0:
                current_initial = currents[0]
                current_final = currents[-1]
                print(f"  {'Initial current':30s} = {current_initial:.6e}")
                print(f"  {'Final current':30s} = {current_final:.6e}")
                if np.abs(current_initial) > 1e-12:
                    print(f"  {'Current change':30s} = {(current_final - current_initial) / current_initial * 100:.2f}%")

            # Vector potential
            vector_potentials = save_data.get('vector_potentials', None)
            if vector_potentials is not None and len(vector_potentials) > 0:
                A_max = np.max(np.abs(vector_potentials))
                print(f"  {'Max vector potential':30s} = {A_max:.6e}")

        # Evolution time info
        times = save_data.get('times', None)
        if times is not None:
            print(f"  {'Evolution duration':30s} = {times[-1] - times[0]:.6f}")
            print(f"  {'Number of evolution steps':30s} = {len(times)}")

        # Field type
        field_type = input_kwargs.get('field_type', None)
        if field_type is not None:
            print(f"  {'Field type':30s} = {field_type}")

    # Print footer
    print("\n" + "="*80)
    print(f"Parameter summary complete")
    print("="*80 + "\n")


#* ============================================================================
#* PHYSICS CHECKS: Normalization & Fluctuation-Dissipation Theorem
#* ============================================================================

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
    evolution.get_thermal_sum(system_params['temperature'])

    gr_errors, gr_totals = state.check_gr_normalization(t1_idx=-1)
    gk_errors, gk_totals, gk_components = state.check_keldysh_normalization(-1, evolution.thermal_dist, evolution.thermal_integral, thermal_sum_left=evolution.thermal_sum_left, thermal_sum_right=evolution.thermal_sum_right)

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
    evolution.get_thermal_sum(system_params['temperature'])

    gk_fdt_row, gk_actual_row, error_row, max_error = state.check_fdt(evolution.thermal_dist, evolution.thermal_integral, time_index=-1, thermal_sum_left=evolution.thermal_sum_left, thermal_sum_right=evolution.thermal_sum_right)

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


#* ============================================================================
#* MAIN PLOTTING FUNCTIONS: Gap, Current, Conductivity (Dimensionless units)
#* ============================================================================

def plot_gap(timestamp, job_index=None, save_plot=False, save_dir=None):
    """
    Plot gap magnitude vs time in dimensionless units.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index (default None plots all jobs in folder)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    field_type = None
    for job_idx in job_indices:
        # Load data with normalization constants
        data = load_simulation_data(timestamp, job_idx)

        # Extract normalized quantities
        times_norm = data['times'] * data['T_c']  # t * T_c (dimensionless)
        gap_norm = np.abs(data['gaps']) / data['T_c']  # |Δ| / T_c

        # Store field type from first job
        if field_type is None:
            field_type = data['field_type']

        label = f'Job {job_idx}' if plot_all else None
        ax.plot(times_norm, gap_norm, linewidth=2.5, label=label, alpha=0.8)

    # Configure plot
    ax.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    ax.set_ylabel(r'$|\Delta(t)| / T_c$', fontsize=13)

    # Title with field type
    title = f'{field_type} pulse simulation' if field_type else 'Gap evolution'
    ax.set_title(title, fontsize=14)

    if plot_all:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        job_str = f'job{job_index}' if job_index is not None else 'all_jobs'
        filename = os.path.join(save_dir, f'gap_vs_time_{timestamp}_{job_str}.png')

        # Save with metadata using extended_savefig
        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamp),
            plot_generating_method='plot_gap',
            additional_information=job_str,
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()


def _extract_param_value(input_kwargs, param_path):
    """Extract nested parameter value from input_kwargs using dot notation."""
    keys = param_path.split('.')
    value = input_kwargs
    for key in keys:
        value = value[key]
    return value


def _explore_all_params(d, prefix='', max_depth=4):
    """
    Recursively explore dictionary and return all parameter paths.

    Args:
        d: Dictionary to explore
        prefix: Current path prefix
        max_depth: Maximum recursion depth

    Returns:
        list: List of parameter paths (dot-notation strings)
    """
    if max_depth == 0 or not isinstance(d, dict):
        return []

    paths = []
    for key, value in d.items():
        current_path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            paths.extend(_explore_all_params(value, current_path, max_depth - 1))
        elif isinstance(value, (int, float, complex, str, bool, type(None))):
            paths.append(current_path)
        elif isinstance(value, np.ndarray) and value.size == 1:
            paths.append(current_path)

    return paths


def _find_varying_parameters(all_kwargs, max_params=3):
    """
    Identify parameters that vary across multiple jobs.

    Automatically searches through all nested parameter structures including
    field_params, system_parameters, grid_parameters, etc.

    Args:
        all_kwargs: List of input_kwargs dicts
        max_params: Maximum number of varying parameters to return

    Returns:
        list: List of (param_path, values) tuples for varying parameters
    """
    if len(all_kwargs) <= 1:
        return []

    # Get all possible parameter paths from first kwargs
    all_paths = _explore_all_params(all_kwargs[0])

    # Parameters to ignore (always vary across jobs)
    ignore_params = ['save_filename', 'filename', 'job_index', 'job_id', 'timestamp']

    varying_params = []

    # Check each parameter path across all jobs
    for param_path in all_paths:
        # Skip ignored parameters
        if any(ignore_param in param_path for ignore_param in ignore_params):
            continue
        try:
            values = [_extract_param_value(kwargs, param_path) for kwargs in all_kwargs]

            # Check if values differ (accounting for numerical precision)
            if isinstance(values[0], (int, float)):
                if not np.allclose(values, values[0], rtol=1e-10):
                    varying_params.append((param_path, values))
            else:
                if not all(v == values[0] for v in values):
                    varying_params.append((param_path, values))

        except (KeyError, TypeError, AttributeError):
            continue

    # Sort by priority: field_params first, then system_parameters, then others
    def get_priority(param_tuple):
        param_path, values = param_tuple
        if param_path.startswith('field_params'):
            return (0, param_path)
        elif param_path.startswith('system_parameters'):
            return (1, param_path)
        elif param_path.startswith('grid_parameters'):
            return (2, param_path)
        else:
            return (3, param_path)

    varying_params.sort(key=get_priority)
    return varying_params[:max_params]


def _create_label_from_params(job_idx, varying_params, values_for_job):
    """Create a label string from varying parameters."""
    if not varying_params:
        return f'Job {job_idx}'

    label_parts = []
    for (param_path, all_values), value in zip(varying_params, values_for_job):
        # Extract short name from path
        param_name = param_path.split('.')[-1]

        # Format value
        if isinstance(value, float):
            if 'temperature' in param_path.lower() or 'critical' in param_path.lower():
                label_parts.append(f'{param_name}={value:.3f}')
            else:
                label_parts.append(f'{param_name}={value:.4g}')
        else:
            label_parts.append(f'{param_name}={value}')

    return ', '.join(label_parts)


def plot_gap_evolution_multi_jobs(timestamp, job_indices=None, save_plot=False,
                                  save_dir=None, running_machine='laptop',
                                  xlim_time=None, xlim_freq=None,
                                  ylim_gap_t=None, ylim_gap_omega_re=None, ylim_gap_omega_im=None,
                                  ylim_A_t=None, ylim_A_omega_re=None, ylim_A_omega_im=None):
    """
    Plot gap and vector potential evolution with time and frequency domain representations.

    Creates a 2x3 plot showing:
        - Top row (Gap): Gap(t), Re[Gap(ω)], Im[Gap(ω)]
        - Bottom row (A): A(t), Re[A(ω)], Im[A(ω)]

    FFTs are phase-corrected for pulses centered at t_0 ≠ 0.

    Args:
        timestamp: Timestamp folder name
        job_indices: List of job indices to plot. If None, plots all jobs in timestamp.
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')
        xlim_time: Tuple (xmin, xmax) for time domain x-axis (both Gap(t) and A(t))
        xlim_freq: Tuple (xmin, xmax) for frequency domain x-axis (overrides automatic range)
        ylim_gap_t: Tuple (ymin, ymax) for Gap(t) y-axis
        ylim_gap_omega_re: Tuple (ymin, ymax) for Re[Gap(ω)] y-axis
        ylim_gap_omega_im: Tuple (ymin, ymax) for Im[Gap(ω)] y-axis
        ylim_A_t: Tuple (ymin, ymax) for A(t) y-axis
        ylim_A_omega_re: Tuple (ymin, ymax) for Re[A(ω)] y-axis
        ylim_A_omega_im: Tuple (ymin, ymax) for Im[A(ω)] y-axis

    Returns:
        A_data_list: List of vector potential arrays A(t), one per job
        gap_data_list: List of gap arrays Δ(t), one per job (complex)
        fig: matplotlib Figure object
        axes: 2x3 numpy array of matplotlib Axes objects

    Example:
        A_data, gap_data, fig, axes = plot_gap_evolution_multi_jobs(
            '20240615_143022',
            job_indices=[0, 1, 2],
            xlim_time=(-10, 10),
            ylim_gap_t=(0, 2)
        )
    """
    # Determine which jobs to plot
    if job_indices is None:
        lines = io.read_contents_readable_file(timestamp)
        job_no = io.recover_job_no(lines)
        job_indices = list(range(job_no))
    else:
        if not isinstance(job_indices, list):
            job_indices = list(job_indices)

    print(f"\n{'='*70}")
    print(f"Plotting gap evolution for {len(job_indices)} jobs from timestamp: {timestamp}")
    print(f"{'='*70}\n")

    # Load all jobs and collect kwargs
    all_data = []
    all_kwargs = []

    for job_idx in job_indices:
        print(f"Loading job {job_idx}...")
        data = load_simulation_data(timestamp, job_idx, running_machine=running_machine)
        all_data.append(data)
        all_kwargs.append(data['input_kwargs'])

    # Find parameters that vary across jobs
    varying_params = _find_varying_parameters(all_kwargs, max_params=3)

    if varying_params:
        print(f"\nDetected varying parameters:")
        for param_path, values in varying_params:
            print(f"  {param_path}: {values}")
    else:
        print("\nNo varying parameters detected - will use job index for labels")

    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_gap_t = axes[0, 0]
    ax_gap_omega_re = axes[0, 1]
    ax_gap_omega_im = axes[0, 2]
    ax_A_t = axes[1, 0]
    ax_A_omega_re = axes[1, 1]
    ax_A_omega_im = axes[1, 2]

    # Define colormap
    from matplotlib import cm
    colors = cm.get_cmap('tab10', len(job_indices))

    # Lists to collect data for return
    A_data_list = []
    gap_data_list = []

    # Plot each job
    field_type = None
    is_current_driven = False
    for idx, (job_idx, data) in enumerate(zip(job_indices, all_data)):
        # Extract data
        times = data['times']
        gaps = data['gaps']
        vector_potentials = data['vector_potentials']
        T_c = data['T_c']

        # Check for input pulse (for current-driven simulations)
        input_pulse = data['save_data'].get('input_pulse', None)

        # Store field type from first job
        if field_type is None:
            field_type = data['field_type']
            is_current_driven = field_type is not None and field_type.startswith('current_')

        # Create label from varying parameters
        if varying_params:
            values_for_job = [values[idx] for _, values in varying_params]
            label = _create_label_from_params(job_idx, varying_params, values_for_job)
        else:
            label = f'Job {job_idx}'

        # Store data for return (raw arrays)
        A_data_list.append(vector_potentials)
        gap_data_list.append(gaps)

        # Time domain normalization
        times_norm = times * T_c  # Dimensionless time
        gap_norm = np.abs(gaps) / T_c  # Dimensionless gap
        A_norm = vector_potentials  # Keep A in original units

        # Compute FFT parameters
        N_t = len(times)
        dt = times[1] - times[0]

        # Find pulse center (time of maximum |A|)
        t_center_idx = np.argmax(np.abs(vector_potentials))
        t_center = times[t_center_idx]

        # Frequency grid
        freq = np.fft.fftfreq(N_t, d=dt)
        omega = 2 * np.pi * freq
        omega_shifted = np.fft.fftshift(omega)

        # FFT of gap deviation (remove DC component to see dynamics)
        # Use absolute value of gap in time domain to avoid complex warnings
        gap_abs = np.abs(gaps)
        gap_deviation = gap_abs - gap_abs[0]  # δ|Δ|(t) = |Δ|(t) - |Δ|(0)
        gap_fft_raw = np.fft.fft(gap_deviation) * dt  # Include dt for proper normalization
        phase_correction = np.exp(-1j * omega * t_center)
        gap_fft_corrected = gap_fft_raw * phase_correction
        gap_fft_shifted = np.fft.fftshift(gap_fft_corrected)
        gap_omega_re_norm = np.real(gap_fft_shifted) / T_c
        gap_omega_im_norm = np.imag(gap_fft_shifted) / T_c

        # FFT of vector potential with phase correction
        A_fft_raw = np.fft.fft(vector_potentials) * dt
        A_fft_corrected = A_fft_raw * phase_correction
        A_fft_shifted = np.fft.fftshift(A_fft_corrected)
        A_omega_re_norm = np.real(A_fft_shifted)
        A_omega_im_norm = np.imag(A_fft_shifted)

        # Plot time domain - Gap(t)
        ax_gap_t.plot(times_norm, gap_norm, linewidth=2.5, label=label,
                      alpha=0.8, color=colors(idx))

        # Plot frequency domain - Re[Gap(ω)]
        ax_gap_omega_re.plot(omega_shifted / T_c, gap_omega_re_norm, linewidth=2.5, label=label,
                             alpha=0.8, color=colors(idx), marker='o', markersize=4)

        # Plot frequency domain - Im[Gap(ω)]
        ax_gap_omega_im.plot(omega_shifted / T_c, gap_omega_im_norm, linewidth=2.5, label=label,
                             alpha=0.8, color=colors(idx), marker='o', markersize=4)

        # Plot time domain - A(t)
        plot_label = label + ' (A)' if is_current_driven and input_pulse is not None else label
        ax_A_t.plot(times_norm, A_norm, linewidth=2.5, label=plot_label,
                    alpha=0.8, color=colors(idx), linestyle='-')

        # Plot input pulse if current-driven
        if is_current_driven and input_pulse is not None:
            ax_A_t.plot(times_norm, input_pulse, linewidth=2.0,
                       label=label + ' ($I_{in}$)',
                       alpha=0.6, color=colors(idx), linestyle='--')

        # Plot frequency domain - Re[A(ω)]
        ax_A_omega_re.plot(omega_shifted / T_c, A_omega_re_norm, linewidth=2.5, label=label,
                           alpha=0.8, color=colors(idx), marker='o', markersize=4)

        # Plot frequency domain - Im[A(ω)]
        ax_A_omega_im.plot(omega_shifted / T_c, A_omega_im_norm, linewidth=2.5, label=label,
                           alpha=0.8, color=colors(idx), marker='o', markersize=4)

    # Add vertical lines at ±2*Δ(0) on frequency plots
    # Use first job's equilibrium gap
    first_gap_0 = np.abs(all_data[0]['gaps'][0])
    first_T_c = all_data[0]['T_c']
    two_gap_norm = 2 * first_gap_0 / first_T_c

    ax_gap_omega_re.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_gap_omega_re.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_gap_omega_im.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_gap_omega_im.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_A_omega_re.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_A_omega_re.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_A_omega_im.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_A_omega_im.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Configure Gap(t) plot
    ax_gap_t.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=12)
    ax_gap_t.set_ylabel(r'$|\Delta(t)| / T_c$', fontsize=12)
    ax_gap_t.set_title(r'Gap: Time Domain', fontsize=13)
    if xlim_time is not None:
        ax_gap_t.set_xlim(xlim_time)
    if ylim_gap_t is not None:
        ax_gap_t.set_ylim(ylim_gap_t)
    ax_gap_t.legend(fontsize=9, loc='best')
    ax_gap_t.grid(True, alpha=0.3)

    # Configure Re[Gap(ω)] plot
    ax_gap_omega_re.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_gap_omega_re.set_ylabel(r'$\mathrm{Re}[\delta\tilde{\Delta}(\omega)] / T_c$', fontsize=12)
    ax_gap_omega_re.set_title(r'Gap: Re[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_gap_omega_re.set_xlim(xlim_freq)
    else:
        ax_gap_omega_re.set_xlim(-3 * two_gap_norm, 3 * two_gap_norm)
    if ylim_gap_omega_re is not None:
        ax_gap_omega_re.set_ylim(ylim_gap_omega_re)
    ax_gap_omega_re.legend(fontsize=9, loc='best')
    ax_gap_omega_re.grid(True, alpha=0.3)

    # Configure Im[Gap(ω)] plot
    ax_gap_omega_im.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_gap_omega_im.set_ylabel(r'$\mathrm{Im}[\delta\tilde{\Delta}(\omega)] / T_c$', fontsize=12)
    ax_gap_omega_im.set_title(r'Gap: Im[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_gap_omega_im.set_xlim(xlim_freq)
    else:
        ax_gap_omega_im.set_xlim(-3 * two_gap_norm, 3 * two_gap_norm)
    if ylim_gap_omega_im is not None:
        ax_gap_omega_im.set_ylim(ylim_gap_omega_im)
    ax_gap_omega_im.legend(fontsize=9, loc='best')
    ax_gap_omega_im.grid(True, alpha=0.3)

    # Configure A(t) plot
    ax_A_t.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=12)
    if is_current_driven:
        ax_A_t.set_ylabel(r'$A(t)$, $I_{in}(t)$', fontsize=12)
        ax_A_t.set_title(r'Vector Potential \& Input Current: Time Domain', fontsize=13)
    else:
        ax_A_t.set_ylabel(r'$A(t)$', fontsize=12)
        ax_A_t.set_title(r'Vector Potential: Time Domain', fontsize=13)
    if xlim_time is not None:
        ax_A_t.set_xlim(xlim_time)
    if ylim_A_t is not None:
        ax_A_t.set_ylim(ylim_A_t)
    ax_A_t.legend(fontsize=9, loc='best')
    ax_A_t.grid(True, alpha=0.3)

    # Configure Re[A(ω)] plot
    ax_A_omega_re.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_A_omega_re.set_ylabel(r'$\mathrm{Re}[\tilde{A}(\omega)]$', fontsize=12)
    ax_A_omega_re.set_title(r'Vector Potential: Re[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_A_omega_re.set_xlim(xlim_freq)
    else:
        ax_A_omega_re.set_xlim(-5 * two_gap_norm, 5 * two_gap_norm)
    if ylim_A_omega_re is not None:
        ax_A_omega_re.set_ylim(ylim_A_omega_re)
    ax_A_omega_re.legend(fontsize=9, loc='best')
    ax_A_omega_re.grid(True, alpha=0.3)

    # Configure Im[A(ω)] plot
    ax_A_omega_im.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_A_omega_im.set_ylabel(r'$\mathrm{Im}[\tilde{A}(\omega)]$', fontsize=12)
    ax_A_omega_im.set_title(r'Vector Potential: Im[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_A_omega_im.set_xlim(xlim_freq)
    else:
        ax_A_omega_im.set_xlim(-5 * two_gap_norm, 5 * two_gap_norm)
    if ylim_A_omega_im is not None:
        ax_A_omega_im.set_ylim(ylim_A_omega_im)
    ax_A_omega_im.legend(fontsize=9, loc='best')
    ax_A_omega_im.grid(True, alpha=0.3)

    # Overall title
    title = f'{field_type} pulse simulation' if field_type else 'Gap and vector potential evolution'
    fig.suptitle(title, fontsize=15, y=0.995)

    plt.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        job_str = f"jobs_{'_'.join(map(str, job_indices))}"
        filename = os.path.join(save_dir, f'gap_evolution_2x3_{timestamp}_{job_str}.png')

        # Save with metadata using extended_savefig
        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamp),
            plot_generating_method='plot_gap_evolution_multi_jobs',
            additional_information=job_str,
            dpi=150,
            bbox_inches='tight'
        )
        print(f"\nPlot saved to: {filename}")
        plt.close()
    else:
        plt.show()

    return A_data_list, gap_data_list, fig, axes


def plot_current_evolution_multi_jobs(timestamp, job_indices=None, save_plot=False,
                                      save_dir=None, running_machine='laptop',
                                      xlim_time=None, xlim_freq=None,
                                      ylim_current_t=None, ylim_current_omega_re=None, ylim_current_omega_im=None,
                                      ylim_A_t=None, ylim_A_omega_re=None, ylim_A_omega_im=None):
    """
    Plot current and vector potential evolution with time and frequency domain representations.

    Creates a 2x3 plot showing:
        - Top row (Current): J(t), Re[J(ω)], Im[J(ω)]
        - Bottom row (A): A(t), Re[A(ω)], Im[A(ω)]

    FFTs are phase-corrected for pulses centered at t_0 ≠ 0.

    Args:
        timestamp: Timestamp folder name
        job_indices: List of job indices to plot. If None, plots all jobs in timestamp.
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')
        xlim_time: Tuple (xmin, xmax) for time domain x-axis (both J(t) and A(t))
        xlim_freq: Tuple (xmin, xmax) for frequency domain x-axis (overrides automatic range)
        ylim_current_t: Tuple (ymin, ymax) for J(t) y-axis
        ylim_current_omega_re: Tuple (ymin, ymax) for Re[J(ω)] y-axis
        ylim_current_omega_im: Tuple (ymin, ymax) for Im[J(ω)] y-axis
        ylim_A_t: Tuple (ymin, ymax) for A(t) y-axis
        ylim_A_omega_re: Tuple (ymin, ymax) for Re[A(ω)] y-axis
        ylim_A_omega_im: Tuple (ymin, ymax) for Im[A(ω)] y-axis

    Returns:
        A_data_list: List of vector potential arrays A(t), one per job
        current_data_list: List of current arrays J(t), one per job
        fig: matplotlib Figure object
        axes: 2x3 numpy array of matplotlib Axes objects

    Example:
        A_data, current_data, fig, axes = plot_current_evolution_multi_jobs(
            '20240615_143022',
            job_indices=[0, 1, 2],
            xlim_time=(-10, 10),
            ylim_current_t=(-0.5, 0.5)
        )
    """
    # Determine which jobs to plot
    if job_indices is None:
        lines = io.read_contents_readable_file(timestamp)
        job_no = io.recover_job_no(lines)
        job_indices = list(range(job_no))
    else:
        if not isinstance(job_indices, list):
            job_indices = list(job_indices)

    print(f"\n{'='*70}")
    print(f"Plotting current evolution for {len(job_indices)} jobs from timestamp: {timestamp}")
    print(f"{'='*70}\n")

    # Load all jobs and collect kwargs
    all_data = []
    all_kwargs = []

    for job_idx in job_indices:
        print(f"Loading job {job_idx}...")
        data = load_simulation_data(timestamp, job_idx, running_machine=running_machine)
        all_data.append(data)
        all_kwargs.append(data['input_kwargs'])

    # Find parameters that vary across jobs
    varying_params = _find_varying_parameters(all_kwargs, max_params=3)

    if varying_params:
        print(f"\nDetected varying parameters:")
        for param_path, values in varying_params:
            print(f"  {param_path}: {values}")
    else:
        print("\nNo varying parameters detected - will use job index for labels")

    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_current_t = axes[0, 0]
    ax_current_omega_re = axes[0, 1]
    ax_current_omega_im = axes[0, 2]
    ax_A_t = axes[1, 0]
    ax_A_omega_re = axes[1, 1]
    ax_A_omega_im = axes[1, 2]

    # Define colormap
    from matplotlib import cm
    colors = cm.get_cmap('tab10', len(job_indices))

    # Lists to collect data for return
    A_data_list = []
    current_data_list = []

    # Plot each job
    field_type = None
    is_current_driven = False
    for idx, (job_idx, data) in enumerate(zip(job_indices, all_data)):
        # Extract data
        times = data['times']
        currents = data['currents']
        vector_potentials = data['vector_potentials']
        T_c = data['T_c']

        # Check for input pulse (for current-driven simulations)
        input_pulse = data['save_data'].get('input_pulse', None)

        # Store field type from first job
        if field_type is None:
            field_type = data['field_type']
            is_current_driven = field_type is not None and field_type.startswith('current_')

        # Create label from varying parameters
        if varying_params:
            values_for_job = [values[idx] for _, values in varying_params]
            label = _create_label_from_params(job_idx, varying_params, values_for_job)
        else:
            label = f'Job {job_idx}'

        # Store data for return (raw arrays)
        A_data_list.append(vector_potentials)
        current_data_list.append(currents)

        # Time domain normalization
        times_norm = times * T_c  # Dimensionless time
        current_norm = currents  # Keep current in original units
        A_norm = vector_potentials  # Keep A in original units

        # Compute FFT parameters
        N_t = len(times)
        dt = times[1] - times[0]

        # Find pulse center (time of maximum |A|)
        t_center_idx = np.argmax(np.abs(vector_potentials))
        t_center = times[t_center_idx]

        # Frequency grid
        freq = np.fft.fftfreq(N_t, d=dt)
        omega = 2 * np.pi * freq
        omega_shifted = np.fft.fftshift(omega)

        # FFT of current deviation (remove DC component to see dynamics)
        current_deviation = currents - currents[0]  # δJ(t) = J(t) - J(0)
        current_fft_raw = np.fft.fft(current_deviation) * dt  # Include dt for proper normalization
        phase_correction = np.exp(-1j * omega * t_center)
        current_fft_corrected = current_fft_raw * phase_correction
        current_fft_shifted = np.fft.fftshift(current_fft_corrected)
        current_omega_re_norm = np.real(current_fft_shifted)
        current_omega_im_norm = np.imag(current_fft_shifted)

        # FFT of vector potential with phase correction
        A_fft_raw = np.fft.fft(vector_potentials) * dt
        A_fft_corrected = A_fft_raw * phase_correction
        A_fft_shifted = np.fft.fftshift(A_fft_corrected)
        A_omega_re_norm = np.real(A_fft_shifted)
        A_omega_im_norm = np.imag(A_fft_shifted)

        # Plot time domain - Current(t)
        ax_current_t.plot(times_norm, current_norm, linewidth=2.5, label=label,
                          alpha=0.8, color=colors(idx))

        # Plot frequency domain - Re[Current(ω)]
        ax_current_omega_re.plot(omega_shifted / T_c, current_omega_re_norm, linewidth=2.5, label=label,
                                 alpha=0.8, color=colors(idx), marker='o', markersize=4)

        # Plot frequency domain - Im[Current(ω)]
        ax_current_omega_im.plot(omega_shifted / T_c, current_omega_im_norm, linewidth=2.5, label=label,
                                 alpha=0.8, color=colors(idx), marker='o', markersize=4)

        # Plot time domain - A(t)
        plot_label = label + ' (A)' if is_current_driven and input_pulse is not None else label
        ax_A_t.plot(times_norm, A_norm, linewidth=2.5, label=plot_label,
                    alpha=0.8, color=colors(idx), linestyle='-')

        # Plot input pulse if current-driven
        if is_current_driven and input_pulse is not None:
            ax_A_t.plot(times_norm, input_pulse, linewidth=2.0,
                       label=label + ' ($I_{in}$)',
                       alpha=0.6, color=colors(idx), linestyle='--')

        # Plot frequency domain - Re[A(ω)]
        ax_A_omega_re.plot(omega_shifted / T_c, A_omega_re_norm, linewidth=2.5, label=label,
                           alpha=0.8, color=colors(idx), marker='o', markersize=4)

        # Plot frequency domain - Im[A(ω)]
        ax_A_omega_im.plot(omega_shifted / T_c, A_omega_im_norm, linewidth=2.5, label=label,
                           alpha=0.8, color=colors(idx), marker='o', markersize=4)

    # Add vertical lines at ±2*Δ(0) on frequency plots
    # Use first job's equilibrium gap
    first_gap_0 = np.abs(all_data[0]['gaps'][0])
    first_T_c = all_data[0]['T_c']
    two_gap_norm = 2 * first_gap_0 / first_T_c

    ax_current_omega_re.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_current_omega_re.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_current_omega_im.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_current_omega_im.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_A_omega_re.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_A_omega_re.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_A_omega_im.axvline(two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\pm 2\Delta_0$')
    ax_A_omega_im.axvline(-two_gap_norm, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Configure Current(t) plot
    ax_current_t.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=12)
    ax_current_t.set_ylabel(r'$J(t)$', fontsize=12)
    ax_current_t.set_title(r'Current: Time Domain', fontsize=13)
    if xlim_time is not None:
        ax_current_t.set_xlim(xlim_time)
    if ylim_current_t is not None:
        ax_current_t.set_ylim(ylim_current_t)
    ax_current_t.legend(fontsize=9, loc='best')
    ax_current_t.grid(True, alpha=0.3)

    # Configure Re[Current(ω)] plot
    ax_current_omega_re.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_current_omega_re.set_ylabel(r'$\mathrm{Re}[\delta\tilde{J}(\omega)]$', fontsize=12)
    ax_current_omega_re.set_title(r'Current: Re[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_current_omega_re.set_xlim(xlim_freq)
    else:
        ax_current_omega_re.set_xlim(-3 * two_gap_norm, 3 * two_gap_norm)
    if ylim_current_omega_re is not None:
        ax_current_omega_re.set_ylim(ylim_current_omega_re)
    ax_current_omega_re.legend(fontsize=9, loc='best')
    ax_current_omega_re.grid(True, alpha=0.3)

    # Configure Im[Current(ω)] plot
    ax_current_omega_im.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_current_omega_im.set_ylabel(r'$\mathrm{Im}[\delta\tilde{J}(\omega)]$', fontsize=12)
    ax_current_omega_im.set_title(r'Current: Im[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_current_omega_im.set_xlim(xlim_freq)
    else:
        ax_current_omega_im.set_xlim(-3 * two_gap_norm, 3 * two_gap_norm)
    if ylim_current_omega_im is not None:
        ax_current_omega_im.set_ylim(ylim_current_omega_im)
    ax_current_omega_im.legend(fontsize=9, loc='best')
    ax_current_omega_im.grid(True, alpha=0.3)

    # Configure A(t) plot
    ax_A_t.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=12)
    if is_current_driven:
        ax_A_t.set_ylabel(r'$A(t)$, $I_{in}(t)$', fontsize=12)
        ax_A_t.set_title(r'Vector Potential \& Input Current: Time Domain', fontsize=13)
    else:
        ax_A_t.set_ylabel(r'$A(t)$', fontsize=12)
        ax_A_t.set_title(r'Vector Potential: Time Domain', fontsize=13)
    if xlim_time is not None:
        ax_A_t.set_xlim(xlim_time)
    if ylim_A_t is not None:
        ax_A_t.set_ylim(ylim_A_t)
    ax_A_t.legend(fontsize=9, loc='best')
    ax_A_t.grid(True, alpha=0.3)

    # Configure Re[A(ω)] plot
    ax_A_omega_re.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_A_omega_re.set_ylabel(r'$\mathrm{Re}[\tilde{A}(\omega)]$', fontsize=12)
    ax_A_omega_re.set_title(r'Vector Potential: Re[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_A_omega_re.set_xlim(xlim_freq)
    else:
        ax_A_omega_re.set_xlim(-5 * two_gap_norm, 5 * two_gap_norm)
    if ylim_A_omega_re is not None:
        ax_A_omega_re.set_ylim(ylim_A_omega_re)
    ax_A_omega_re.legend(fontsize=9, loc='best')
    ax_A_omega_re.grid(True, alpha=0.3)

    # Configure Im[A(ω)] plot
    ax_A_omega_im.set_xlabel(r'Frequency $\omega$ ($T_c$)', fontsize=12)
    ax_A_omega_im.set_ylabel(r'$\mathrm{Im}[\tilde{A}(\omega)]$', fontsize=12)
    ax_A_omega_im.set_title(r'Vector Potential: Im[FFT]', fontsize=13)
    if xlim_freq is not None:
        ax_A_omega_im.set_xlim(xlim_freq)
    else:
        ax_A_omega_im.set_xlim(-5 * two_gap_norm, 5 * two_gap_norm)
    if ylim_A_omega_im is not None:
        ax_A_omega_im.set_ylim(ylim_A_omega_im)
    ax_A_omega_im.legend(fontsize=9, loc='best')
    ax_A_omega_im.grid(True, alpha=0.3)

    # Overall title
    title = f'{field_type} pulse simulation' if field_type else 'Current and vector potential evolution'
    fig.suptitle(title, fontsize=15, y=0.995)

    plt.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        job_str = f"jobs_{'_'.join(map(str, job_indices))}"
        filename = os.path.join(save_dir, f'current_evolution_2x3_{timestamp}_{job_str}.png')

        # Save with metadata using extended_savefig
        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamp),
            plot_generating_method='plot_current_evolution_multi_jobs',
            additional_information=job_str,
            dpi=150,
            bbox_inches='tight'
        )
        print(f"\nPlot saved to: {filename}")
        plt.close()
    else:
        plt.show()

    return A_data_list, current_data_list, fig, axes


def compare_gap_across_timestamps(timestamps, sweep_parameter, label_parameter='dt',
                                  label_format=None, normalize_gap=True,
                                  gap_type='final', job_range=None,
                                  save_plot=False, save_dir=None, running_machine='laptop'):
    """
    Compare gap vs sweep parameter across multiple timestamps with customizable labels.

    Each timestamp should contain a parameter sweep (multiple jobs with varying parameter).
    Plots gap vs the sweep parameter for each timestamp, with curves labeled by
    a different parameter (e.g., dt, eta).

    Args:
        timestamps: List of timestamp folder names to compare (each should be a parameter sweep)
        sweep_parameter: Parameter being swept in each timestamp. Can be:
                        - String path like 'system_parameters.temperature' or 'field_params.amplitude'
                        - Callable that takes (input_kwargs, save_data) and returns sweep value
        label_parameter: Parameter to use for labeling curves. Can be:
                        - 'dt': Compute dt from grid parameters
                        - String path like 'system_parameters.eta'
                        - Callable that takes (input_kwargs, save_data) and returns label string
                        - None: use timestamp as label
        label_format: Format string for labels (e.g., 'dt = {:.4f}'). If None, uses default.
        normalize_gap: If True, normalize gap by T_c (default True)
        gap_type: Which gap to extract. Options:
                 - 'final': Final gap value (last timestep)
                 - 'equilibrium': Equilibrium gap (first timestep, for equilibrium sweeps)
                 - 'mean': Mean gap over all timesteps
                 - 'max': Maximum gap
                 - 'min': Minimum gap
        job_range: Range of jobs to load from each timestamp. Can be:
                  - None: load all jobs
                  - tuple (start, end): load jobs from start to end (exclusive)
                  - int: load first N jobs
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')

    Returns:
        dict: {
            'fig': matplotlib figure,
            'ax': matplotlib axis,
            'data': list of dicts with sweep data for each timestamp
        }

    Example:
        # Compare temperature sweeps with different dt values
        compare_gap_across_timestamps(
            timestamps=['20240615_143022', '20240615_150033', '20240615_152145'],
            sweep_parameter='system_parameters.temperature',
            label_parameter='dt',
            label_format='$\\Delta t = {:.4f}$'
        )

        # Compare vector potential sweeps labeled by eta
        compare_gap_across_timestamps(
            timestamps=['20240615_143022', '20240615_150033'],
            sweep_parameter='field_params.amplitude',
            label_parameter='system_parameters.eta',
            label_format='$\\eta = {:.3f}$',
            gap_type='equilibrium'
        )

        # Custom sweep parameter extraction
        def get_A_value(kwargs, data):
            # Extract vector potential from data
            return np.max(np.abs(data['vector_potentials']))

        compare_gap_across_timestamps(
            timestamps=['20240615_143022', '20240615_150033'],
            sweep_parameter=get_A_value,
            label_parameter='dt'
        )
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Store loaded data for return
    loaded_data = []

    # Define colormap for multiple curves
    from matplotlib import cm
    colors = cm.get_cmap('tab10', len(timestamps))

    print(f"\n{'='*70}")
    print(f"Comparing gap sweeps across {len(timestamps)} timestamps")
    print(f"{'='*70}")

    for idx, timestamp in enumerate(timestamps):
        print(f"\nLoading timestamp {idx+1}/{len(timestamps)}: {timestamp}")

        # Determine how many jobs to load
        lines = io.read_contents_readable_file(timestamp)
        total_jobs = io.recover_job_no(lines)

        if job_range is None:
            jobs_to_load = range(total_jobs)
        elif isinstance(job_range, tuple):
            jobs_to_load = range(job_range[0], min(job_range[1], total_jobs))
        elif isinstance(job_range, int):
            jobs_to_load = range(min(job_range, total_jobs))
        else:
            raise ValueError("job_range must be None, tuple, or int")

        print(f"  Loading {len(jobs_to_load)} jobs (total available: {total_jobs})")

        # Arrays to store sweep data
        sweep_values = []
        gap_values = []

        # Load all jobs in sweep
        for job_idx in jobs_to_load:
            input_kwargs, save_data = load_job_data(timestamp, job_idx, running_machine=running_machine)

            # Extract sweep parameter value
            if callable(sweep_parameter):
                sweep_val = sweep_parameter(input_kwargs, save_data)
            elif isinstance(sweep_parameter, str):
                keys = sweep_parameter.split('.')
                sweep_val = input_kwargs
                try:
                    for key in keys:
                        sweep_val = sweep_val[key]
                except (KeyError, TypeError) as e:
                    print(f"  Warning: Could not extract '{sweep_parameter}' from job {job_idx}")
                    print(f"  Error: {e}")
                    continue
            else:
                raise ValueError("sweep_parameter must be callable or string path")

            # Extract gap based on gap_type
            gaps = save_data['gaps']
            gap_magnitude = np.abs(gaps)

            if gap_type == 'final':
                gap_val = gap_magnitude[-1]
            elif gap_type == 'equilibrium':
                gap_val = gap_magnitude[0]
            elif gap_type == 'mean':
                gap_val = np.mean(gap_magnitude)
            elif gap_type == 'max':
                gap_val = np.max(gap_magnitude)
            elif gap_type == 'min':
                gap_val = np.min(gap_magnitude)
            else:
                raise ValueError(f"Unknown gap_type: '{gap_type}'")

            sweep_values.append(sweep_val)
            gap_values.append(gap_val)

        # Convert to arrays and sort by sweep parameter
        sweep_values = np.array(sweep_values)
        gap_values = np.array(gap_values)
        sort_idx = np.argsort(sweep_values)
        sweep_values = sweep_values[sort_idx]
        gap_values = gap_values[sort_idx]

        # Get normalization constant from first job
        first_kwargs, _ = load_job_data(timestamp, jobs_to_load[0], running_machine=running_machine)
        T_c = first_kwargs['system_parameters']['critical_temperature']

        # Normalize gap if requested
        if normalize_gap:
            gap_values = gap_values / T_c
            ylabel = r'$|\Delta| / T_c$'
        else:
            ylabel = r'$|\Delta|$'

        # Generate label for this curve
        if label_parameter is None:
            label = timestamp
        elif label_parameter == 'dt':
            # Compute dt from state
            _, first_save_data = load_job_data(timestamp, jobs_to_load[0], running_machine=running_machine)
            state = first_save_data['final_state']
            if hasattr(state, 'dt') and state.dt is not None:
                dt_val = state.dt
            else:
                # Compute from grid parameters
                time_sampling = state.gr.data.shape[-1]
                dt_val = state.T_max / (time_sampling - 1)

            if label_format is None:
                label = f'$\\Delta t = {dt_val:.4g}$'
            else:
                label = label_format.format(dt_val)

        elif callable(label_parameter):
            label = label_parameter(first_kwargs, first_save_data)
        elif isinstance(label_parameter, str):
            keys = label_parameter.split('.')
            value = first_kwargs
            try:
                for key in keys:
                    value = value[key]

                if label_format is None:
                    if isinstance(value, float):
                        label = f'{label_parameter} = {value:.4g}'
                    else:
                        label = f'{label_parameter} = {value}'
                else:
                    label = label_format.format(value)

            except (KeyError, TypeError) as e:
                print(f"  Warning: Could not extract '{label_parameter}'. Using timestamp as label.")
                print(f"  Error: {e}")
                label = timestamp
        else:
            raise ValueError("label_parameter must be None, 'dt', callable, or string path")

        print(f"  Label: {label}")
        print(f"  Sweep parameter range: [{np.min(sweep_values):.6f}, {np.max(sweep_values):.6f}]")
        print(f"  Gap range: [{np.min(gap_values):.6f}, {np.max(gap_values):.6f}]")

        # Store data
        loaded_data.append({
            'timestamp': timestamp,
            'sweep_values': sweep_values,
            'gap_values': gap_values,
            'label': label,
            'T_c': T_c
        })

        # Plot
        ax.plot(sweep_values, gap_values, 'o-', linewidth=2.5, markersize=6,
               label=label, alpha=0.8, color=colors(idx))

    # Configure plot
    # Determine xlabel from sweep_parameter
    if isinstance(sweep_parameter, str):
        if 'temperature' in sweep_parameter.lower():
            xlabel = r'Temperature ($T_c$)'
        elif 'amplitude' in sweep_parameter.lower() or 'vector' in sweep_parameter.lower():
            xlabel = r'Vector Potential $A$'
        else:
            # Use parameter name as label
            param_name = sweep_parameter.split('.')[-1]
            xlabel = param_name.replace('_', ' ').title()
    else:
        xlabel = 'Sweep Parameter'

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title('Gap comparison across parameter sweeps', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print(f"\n{'='*70}")
    print(f"Plot complete")
    print(f"{'='*70}\n")

    # Save if requested
    if save_plot:
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        timestamp_str = f'{timestamps[0]}_to_{timestamps[-1]}' if len(timestamps) > 1 else timestamps[0]
        filename = os.path.join(save_dir, f'gap_sweep_comparison_{timestamp_str}.png')

        extended_savefig(
            fig,
            filename,
            data_source_timestamp=', '.join(timestamps),
            plot_generating_method='compare_gap_across_timestamps',
            additional_information=f'{len(timestamps)} sweeps compared',
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()

    return {
        'fig': fig,
        'ax': ax,
        'data': loaded_data
    }


def compare_fdt_error_across_timestamps(timestamps, job_indices=None, x_parameter='time',
                                        label_parameter='dt', label_format=None,
                                        normalize_time=True, pauli_channels='max',
                                        plot_every=1, save_plot=False, save_dir=None,
                                        running_machine='laptop'):
    """
    Compare FDT error across multiple timestamps with customizable labels.

    Computes FDT error at each available timestep and plots maximum error magnitude
    vs a specified parameter (time, index, etc.) for multiple simulation runs.

    Args:
        timestamps: List of timestamp folder names to compare
        job_indices: Job indices for each timestamp. Can be:
                    - None: use job 0 for all timestamps
                    - int: use same job index for all timestamps
                    - list: one job index per timestamp (must match length of timestamps)
        x_parameter: What to plot on x-axis. Options:
                    - 'time': Time array from simulation
                    - 'index': Timestep index (0, 1, 2, ...)
                    - 'normalized_time': Time normalized by T_c
        label_parameter: Parameter from input_kwargs to use for labels. Can be:
                        - String path like 'dt' or 'system_parameters.eta'
                        - Callable that takes (input_kwargs, save_data) and returns label string
                        - None: use timestamp as label
        label_format: Format string for labels (e.g., 'dt = {:.4f}'). If None, uses default.
                     Only used if label_parameter is a string path.
        normalize_time: If True and x_parameter='time', normalize time by T_c (default True)
        pauli_channels: Which Pauli channels to plot. Options:
                       - 'max': Maximum error across all 4 channels (default)
                       - 'all': Plot all 4 channels separately (4 subplots)
                       - int (0-3): Specific Pauli channel to plot
                       - list of ints: Multiple specific channels
        plot_every: Plot every N-th time slice to reduce clutter (default 1)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')

    Returns:
        dict: {
            'fig': matplotlib figure,
            'ax': matplotlib axis or array of axes,
            'data': list of dicts with loaded data and FDT errors for each timestamp
        }

    Example:
        # Compare FDT errors with different dt values
        compare_fdt_error_across_timestamps(
            timestamps=['20240615_143022', '20240615_150033', '20240615_152145'],
            label_parameter='dt',
            label_format='$\\Delta t = {:.4f}$',
            pauli_channels='max'
        )

        # Plot all Pauli channels separately
        compare_fdt_error_across_timestamps(
            timestamps=['20240615_143022', '20240615_150033'],
            label_parameter='system_parameters.temperature',
            pauli_channels='all',
            plot_every=5
        )
    """
    # Handle job_indices argument
    if job_indices is None:
        job_indices_list = [0] * len(timestamps)
    elif isinstance(job_indices, int):
        job_indices_list = [job_indices] * len(timestamps)
    elif isinstance(job_indices, list):
        if len(job_indices) != len(timestamps):
            raise ValueError(f"job_indices list length ({len(job_indices)}) must match timestamps length ({len(timestamps)})")
        job_indices_list = job_indices
    else:
        raise ValueError("job_indices must be None, int, or list")

    # Parse pauli_channels argument
    if pauli_channels == 'max':
        plot_mode = 'max'
        n_subplots = 1
        channels_to_plot = None
    elif pauli_channels == 'all':
        plot_mode = 'all'
        n_subplots = 4
        channels_to_plot = [0, 1, 2, 3]
    elif isinstance(pauli_channels, int):
        plot_mode = 'specific'
        n_subplots = 1
        channels_to_plot = [pauli_channels]
    elif isinstance(pauli_channels, list):
        plot_mode = 'specific'
        n_subplots = len(pauli_channels)
        channels_to_plot = pauli_channels
    else:
        raise ValueError("pauli_channels must be 'max', 'all', int, or list of ints")

    # Create figure
    if n_subplots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

    # Store loaded data and errors for return
    loaded_data = []

    # Define colormap for multiple curves
    from matplotlib import cm
    colors = cm.get_cmap('tab10', len(timestamps))

    # Pauli labels
    pauli_labels = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']

    print(f"\n{'='*70}")
    print(f"Comparing FDT errors across {len(timestamps)} timestamps")
    print(f"{'='*70}")

    for idx, (timestamp, job_idx) in enumerate(zip(timestamps, job_indices_list)):
        print(f"\nLoading timestamp {idx+1}/{len(timestamps)}: {timestamp} (job {job_idx})")

        # Load data
        input_kwargs, save_data = load_job_data(timestamp, job_idx, running_machine=running_machine)
        state = save_data['final_state']
        system_params = _get_system_parameters(input_kwargs)
        grid_params = _get_grid_parameters(save_data, input_kwargs)

        # Get normalization constant
        T_c = system_params['critical_temperature']

        # Check if state has enough time history
        n_times = state.gr.data.shape[2]
        if n_times < 2:
            print(f"  Warning: State only has {n_times} timestep. Cannot compute FDT error sweep.")
            print(f"  This state was saved with save_full_state=False. Skipping timestamp.")
            continue

        # Create evolution object for thermal distributions
        evolution = UsadelKeldyshEvolution(grid_params, system_params)
        evolution.get_thermal_occupation(system_params['temperature'])
        evolution.get_thermal_integral(system_params['temperature'])
        evolution.get_thermal_sum(system_params['temperature'])

        # Compute FDT errors at all time indices
        time_indices = np.arange(0, n_times, plot_every)
        fdt_errors = []  # List of arrays with shape (4,) for each time index

        print(f"  Computing FDT errors at {len(time_indices)} time points...")

        for t_idx in time_indices:
            # Check FDT at this time index
            _, _, error_row, _ = state.check_fdt(
                evolution.thermal_dist,
                evolution.thermal_integral,
                time_index=t_idx,
                thermal_sum_left=evolution.thermal_sum_left,
                thermal_sum_right=evolution.thermal_sum_right
            )

            # Extract max error per Pauli channel
            errors_per_channel = np.zeros(4)
            for pauli_idx in range(4):
                error_pauli = error_row.trace(pauli_idx) / 2
                if error_pauli.ndim == 2:
                    error_pauli = error_pauli[0, :]
                errors_per_channel[pauli_idx] = np.max(np.abs(error_pauli))

            fdt_errors.append(errors_per_channel)

        fdt_errors = np.array(fdt_errors)  # Shape: (n_time_indices, 4)

        # Determine x-axis data
        if x_parameter == 'time':
            times = save_data.get('times', None)
            if times is None:
                raise ValueError(f"Timestamp {timestamp}: 'times' not found in save_data. Use x_parameter='index' instead.")
            times_subset = times[time_indices]
            if normalize_time:
                x_data = times_subset * T_c
                xlabel = r'Time ($T_c^{-1}$)'
            else:
                x_data = times_subset
                xlabel = r'Time'
        elif x_parameter == 'normalized_time':
            times = save_data.get('times', None)
            if times is None:
                raise ValueError(f"Timestamp {timestamp}: 'times' not found in save_data. Use x_parameter='index' instead.")
            x_data = times[time_indices] * T_c
            xlabel = r'Time ($T_c^{-1}$)'
        elif x_parameter == 'index':
            x_data = time_indices
            xlabel = 'Timestep index'
        else:
            raise ValueError(f"Unknown x_parameter: '{x_parameter}'. Use 'time', 'normalized_time', or 'index'.")

        # Generate label
        if label_parameter is None:
            label = timestamp
        elif callable(label_parameter):
            label = label_parameter(input_kwargs, save_data)
        elif isinstance(label_parameter, str):
            keys = label_parameter.split('.')
            value = input_kwargs
            try:
                for key in keys:
                    value = value[key]

                if label_format is None:
                    if isinstance(value, float):
                        label = f'{label_parameter} = {value:.4g}'
                    else:
                        label = f'{label_parameter} = {value}'
                else:
                    label = label_format.format(value)

            except (KeyError, TypeError) as e:
                print(f"  Warning: Could not extract '{label_parameter}' from input_kwargs. Using timestamp as label.")
                print(f"  Error: {e}")
                label = timestamp
        else:
            raise ValueError("label_parameter must be None, callable, or string path")

        print(f"  Label: {label}")
        print(f"  FDT error range (max across channels): [{np.min(np.max(fdt_errors, axis=1)):.4e}, {np.max(np.max(fdt_errors, axis=1)):.4e}]")

        # Store data
        loaded_data.append({
            'timestamp': timestamp,
            'job_index': job_idx,
            'input_kwargs': input_kwargs,
            'save_data': save_data,
            'fdt_errors': fdt_errors,
            'x_data': x_data,
            'label': label
        })

        # Plot based on mode
        if plot_mode == 'max':
            # Plot maximum error across all channels
            max_errors = np.max(fdt_errors, axis=1)
            axes[0].semilogy(x_data, max_errors, linewidth=2.5, label=label,
                           alpha=0.8, color=colors(idx), marker='o', markersize=4)

        elif plot_mode == 'all' or plot_mode == 'specific':
            # Plot each channel separately
            for subplot_idx, pauli_idx in enumerate(channels_to_plot):
                axes[subplot_idx].semilogy(x_data, fdt_errors[:, pauli_idx],
                                          linewidth=2.5, label=label, alpha=0.8,
                                          color=colors(idx), marker='o', markersize=4)

    # Configure plots
    if plot_mode == 'max':
        axes[0].set_xlabel(xlabel, fontsize=13)
        axes[0].set_ylabel(r'Max FDT Error (all $\tau$ channels)', fontsize=13)
        axes[0].set_title('Maximum FDT Error Comparison', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, which='both')

    else:
        for subplot_idx, pauli_idx in enumerate(channels_to_plot):
            axes[subplot_idx].set_xlabel(xlabel, fontsize=12)
            axes[subplot_idx].set_ylabel(f'FDT Error {pauli_labels[pauli_idx]}', fontsize=12)
            axes[subplot_idx].set_title(f'FDT Error: {pauli_labels[pauli_idx]}', fontsize=13)
            axes[subplot_idx].legend(fontsize=9)
            axes[subplot_idx].grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    print(f"\n{'='*70}")
    print(f"Plot complete")
    print(f"{'='*70}\n")

    # Save if requested
    if save_plot:
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        timestamp_str = f'{timestamps[0]}_to_{timestamps[-1]}' if len(timestamps) > 1 else timestamps[0]
        filename = os.path.join(save_dir, f'fdt_error_comparison_{timestamp_str}.png')

        extended_savefig(
            fig,
            filename,
            data_source_timestamp=', '.join(timestamps),
            plot_generating_method='compare_fdt_error_across_timestamps',
            additional_information=f'{len(timestamps)} timestamps compared',
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()

    return {
        'fig': fig,
        'ax': axes[0] if n_subplots == 1 else axes,
        'data': loaded_data
    }


def extract_conductivity(timestamp, job_index, threshold=1e-6, save_plot=False, save_dir=None,
                        omega_external=None, sigma_re_external=None, sigma_im_external=None,
                        labels_external=None):
    """
    Extract and plot optical conductivity σ(ω) = j(ω)/(iω A(ω)) from time-domain data.

    Computes the optical conductivity using σ(ω) = j(ω)/E(ω) where E(ω) = iω A(ω).
    Filters out frequencies where |A(ω)| or |ω| are below threshold to avoid
    numerical instabilities.

    Args:
        timestamp: Timestamp folder name
        job_index: Job index
        threshold: Minimum |A(ω)| value for valid conductivity (default 1e-6)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
        omega_external: Optional external frequency values for comparison. Can be:
                       - Single array: plots one external dataset
                       - List of arrays: plots multiple external datasets
        sigma_re_external: Optional external Re[σ(ω)] values. Can be:
                          - Single array: plots one external dataset
                          - List of arrays: plots multiple external datasets (must match omega_external length)
        sigma_im_external: Optional external Im[σ(ω)] values. Can be:
                          - Single array: plots one external dataset
                          - List of arrays: plots multiple external datasets (must match omega_external length)
        labels_external: Optional labels for external datasets. Can be:
                        - Single string: label for single external dataset
                        - List of strings: labels for multiple external datasets (must match omega_external length)
                        - If not provided, defaults to 'External data' or 'External 1', 'External 2', etc.

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
    max_A = np.max(np.abs(vector_potentials))
    if max_A < 1e-10:
        print(f"Vector potential is zero for job {job_index}. Max |A| = {max_A:.3e}. Conductivity extraction requires non-zero driving.")
        return None
    else:
        print(f"Max |A(t)| = {max_A:.3e}")

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

    # Check boundary values to diagnose FFT artifacts
    print(f"\nTime-domain boundary values:")
    print(f"  A(t_start) = {vector_potentials[0]:.3e}, A(t_end) = {vector_potentials[-1]:.3e}")
    print(f"  j(t_start) = {currents[0]:.3e}, j(t_end) = {currents[-1]:.3e}")
    print(f"  Max |A(t)| = {np.max(np.abs(vector_potentials)):.3e}")
    print(f"  Max |j(t)| = {np.max(np.abs(currents)):.3e}")

    # Construct frequency grid first (needed for phase correction)
    freq = np.fft.fftfreq(N_t, d=dt)
    omega_grid = 2 * np.pi * freq
    omega_grid = np.fft.fftshift(omega_grid)

    # Fourier transform with phase correction for centered pulse
    # If pulse is centered at t_center, FFT includes phase e^{-iω t_center}
    # Remove this to get clean spectrum
    t_center = (times[0] + times[-1]) / 2.0

    # Fourier transform current: j(ω)
    j_omega = np.fft.fft(currents)
    j_omega = np.fft.fftshift(j_omega)

    # Fourier transform vector potential: A(ω)
    A_omega = np.fft.fft(vector_potentials)
    A_omega = np.fft.fftshift(A_omega)

    # Remove phase from centered pulse: multiply by e^{+iω t_center}
    phase_correction = np.exp(1j * omega_grid * t_center)
    j_omega = j_omega * phase_correction
    A_omega = A_omega * phase_correction

    # Calculate frequency resolution
    T_total = times[-1] - times[0]
    d_omega = omega_grid[1] - omega_grid[0] if len(omega_grid) > 1 else 0

    # Extract equilibrium gap (same as plot_gap function)
    gaps = save_data['gaps']
    # Use first gap value as equilibrium (before driving)
    gap_eq = np.abs(gaps[0])

    # Compute conductivity: σ(ω) = -j(ω) / (iω A(ω))
    sigma = np.zeros_like(j_omega, dtype=complex)
    # Filter: need both |A(ω)| > threshold and |ω| > threshold (avoid ω=0)
    mask = (np.abs(A_omega) > threshold) & (np.abs(omega_grid) > threshold)
    sigma[mask] = j_omega[mask] / (-1j * omega_grid[mask] * A_omega[mask])

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
    print(f"  Time step (dt): {dt:.6f}")
    print(f"  Total duration (T_total): {T_total:.3f}")
    print(f"  Number of time points: {N_t}")
    print(f"  Frequency resolution (Δω): {d_omega:.6f}")
    print(f"  Nyquist frequency (ω_max): {np.max(np.abs(omega_grid)):.3f}")
    print(f"  Equilibrium gap (Δ₀): {gap_eq:.6f}")
    print(f"  Max |A(ω)|: {np.max(np.abs(A_omega)):.3e}")
    print(f"  Threshold: {threshold:.3e}")
    print(f"  Valid frequency range: [{omega_min:.3f}, {omega_max:.3f}]" if omega_min is not None else "  No valid frequencies")
    print(f"  Number of valid frequency points: {np.sum(mask)}/{N_t}")
    print(f"{'='*60}\n")

    # Plot conductivity and components in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute electric field E(ω) = iωA(ω)
    E_omega = 1j * omega_grid * A_omega

    # Top left: Re[σ(ω)] - only positive frequencies where |σ| > 0
    # Additional mask: only plot where absolute conductivity is non-zero
    sigma_abs = np.abs(sigma)
    sigma_threshold = 1e-12  # Threshold for considering conductivity as non-zero

    mask_plot_re = (omega_grid > threshold) & (sigma_abs > sigma_threshold)
    axes[0, 0].plot(omega_grid[mask_plot_re], np.real(sigma)[mask_plot_re], 'b-', linewidth=2, label=r'Simulation')
    axes[0, 0].set_xlabel(r'Frequency $\omega$', fontsize=12)
    axes[0, 0].set_ylabel(r'Re($\sigma(\omega)$)', fontsize=12)
    axes[0, 0].set_title(r'Real part of conductivity ($\omega > 0$)', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Im[σ(ω)] - only positive frequencies where |σ| > 0
    mask_plot_im = (omega_grid > 0) & (sigma_abs > sigma_threshold)
    axes[0, 1].plot(omega_grid[mask_plot_im], np.imag(sigma)[mask_plot_im], 'r-', linewidth=2, label=r'Simulation')
    axes[0, 1].set_xlabel(r'Frequency $\omega$', fontsize=12)
    axes[0, 1].set_ylabel(r'Im($\sigma(\omega)$)', fontsize=12)
    axes[0, 1].set_title(r'Imaginary part of conductivity ($\omega > 0$)', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot external data if provided
    if omega_external is not None:
        # Normalize external data to lists
        if not isinstance(omega_external, list):
            omega_list = [omega_external]
            sigma_re_list = [sigma_re_external] if sigma_re_external is not None else [None]
            sigma_im_list = [sigma_im_external] if sigma_im_external is not None else [None]
            if labels_external is None:
                label_list = ['External data']
            elif isinstance(labels_external, str):
                label_list = [labels_external]
            else:
                label_list = labels_external
        else:
            omega_list = omega_external
            sigma_re_list = sigma_re_external if sigma_re_external is not None else [None] * len(omega_list)
            sigma_im_list = sigma_im_external if sigma_im_external is not None else [None] * len(omega_list)
            if labels_external is None:
                label_list = [f'External {i+1}' for i in range(len(omega_list))]
            else:
                label_list = labels_external

        # Plot each external dataset
        for i, (omega_ext, sigma_re_ext, sigma_im_ext, label_ext) in enumerate(zip(omega_list, sigma_re_list, sigma_im_list, label_list)):
            color_idx = i + 1  # Offset to avoid blue/red used for simulation

            # Plot real part if provided
            if sigma_re_ext is not None:
                axes[0, 0].plot(omega_ext, sigma_re_ext, '--', linewidth=2,
                              label=label_ext, alpha=0.7, color=f'C{color_idx}')

            # Plot imaginary part if provided
            if sigma_im_ext is not None:
                axes[0, 1].plot(omega_ext, sigma_im_ext, '--', linewidth=2,
                              label=label_ext, alpha=0.7, color=f'C{color_idx}')

    axes[0, 0].legend(fontsize=10)
    axes[0, 1].legend(fontsize=10)

    # Bottom left: j(ω)
    axes[1, 0].plot(omega_grid, np.real(j_omega), 'b-', linewidth=2, label=r'Re($j(\omega)$)')
    axes[1, 0].plot(omega_grid, np.imag(j_omega), 'b--', linewidth=2, alpha=0.7, label=r'Im($j(\omega)$)')
    axes[1, 0].set_xlabel(r'Frequency $\omega$', fontsize=12)
    axes[1, 0].set_ylabel(r'$j(\omega)$', fontsize=12)
    axes[1, 0].set_title(r'Current density in frequency domain', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)

    # Bottom right: E(ω) = iωA(ω)
    axes[1, 1].plot(omega_grid, np.real(E_omega), 'r-', linewidth=2, label=r'Re($E(\omega)$)')
    axes[1, 1].plot(omega_grid, np.imag(E_omega), 'r--', linewidth=2, alpha=0.7, label=r'Im($E(\omega)$)')
    axes[1, 1].set_xlabel(r'Frequency $\omega$', fontsize=12)
    axes[1, 1].set_ylabel(r'$E(\omega) = i\omega A(\omega)$', fontsize=12)
    axes[1, 1].set_title(r'Electric field in frequency domain', fontsize=13)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)

    # Set x-axis limits to positive frequency range only with small margin
    if omega_min is not None and omega_max is not None:
        # Only use positive part of valid range
        omega_max_positive = max(omega_max, 0)
        margin = 0.1 * omega_max_positive  # 10% margin on right
        plot_min = 0
        plot_max = omega_max_positive + margin
    else:
        # Fallback if no valid range found
        plot_min = 0
        plot_max = 10 * gap_eq

    for ax in axes.flat:
        ax.set_xlim(plot_min, plot_max)

    # Mark valid frequency range and coherence peaks on all subplots
    for ax in axes.flat:
        if omega_min is not None:
            ax.axvline(omega_min, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
            ax.axvline(omega_max, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
        ax.axvline(2 * gap_eq, color='red', linestyle=':', linewidth=1, alpha=0.4)
        ax.axvline(-2 * gap_eq, color='red', linestyle=':', linewidth=1, alpha=0.4)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'conductivity_{timestamp}_job{job_index}.png')

        # Save with metadata using extended_savefig
        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamp),
            plot_generating_method='extract_conductivity',
            additional_information=f'job{job_index}',
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()

    return {'omega_grid': omega_grid, 'sigma': sigma}

def plot_current(timestamp, job_index=None, save_plot=False, save_dir=None):
    """
    Plot current and vector potential vs time in dimensionless units.

    Automatically detects current-driven simulations and creates appropriate layout:
    - Current-driven: 3 panels (Current | Vector Potential | Input Current)
    - Voltage-driven: 2 panels (Current | Vector Potential)

    Args:
        timestamp: Timestamp folder name
        job_index: Job index (default None plots all jobs in folder)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
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

    # Detect if current-driven by loading first job
    data_first = load_simulation_data(timestamp, job_indices[0])
    field_type = data_first['field_type']
    is_current_driven = field_type is not None and field_type.startswith('current_')
    has_input_pulse = 'input_pulse' in data_first['save_data']

    # Create figure with appropriate number of subplots
    if is_current_driven and has_input_pulse:
        # Current-driven: 3 panels (Current | Vector Potential | Input Current)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_current, ax_A, ax_input = axes
    else:
        # Voltage-driven: 2 panels (Current | Vector Potential)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_current, ax_A = axes
        ax_input = None
    for job_idx in job_indices:
        # Load data with normalization constants
        data = load_simulation_data(timestamp, job_idx)

        # Extract normalized quantities
        times_norm = data['times'] * data['T_c']  # t * T_c (dimensionless)
        current_norm = np.real(data['currents'])  # J / J_0

        # Vector potential (real part if complex)
        A_values = np.real(data['vector_potentials']) if np.iscomplexobj(data['vector_potentials']) else data['vector_potentials']
        A_norm = A_values  # A / A_0

        # Input pulse (for current-driven simulations)
        input_pulse = data['save_data'].get('input_pulse', None)

        # Create label
        label = f'Job {job_idx}' if plot_all else None

        # Plot current
        ax_current.plot(times_norm, current_norm, linewidth=2.5, label=label, alpha=0.8, color=f'C{job_idx}')

        # Plot vector potential
        ax_A.plot(times_norm, A_norm, linewidth=2.5, label=label, alpha=0.8, color=f'C{job_idx}')

        # Plot input pulse on separate panel if current-driven
        if is_current_driven and has_input_pulse and ax_input is not None:
            ax_input.plot(times_norm, input_pulse, linewidth=2.5, label=label, alpha=0.8, color=f'C{job_idx}')

    # Configure current axis
    ax_current.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    ax_current.set_ylabel(r'$J(t)$', fontsize=13)
    title = f'{field_type} - Supercurrent' if field_type else 'Supercurrent'
    ax_current.set_title(title, fontsize=14)
    if plot_all:
        ax_current.legend(fontsize=10)
    ax_current.grid(True, alpha=0.3)

    # Configure vector potential axis
    ax_A.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    ax_A.set_ylabel(r'$A(t)$', fontsize=13)
    title_A = f'{field_type} - Vector potential' if field_type else 'Vector potential'
    ax_A.set_title(title_A, fontsize=14)
    if plot_all:
        ax_A.legend(fontsize=10)
    ax_A.grid(True, alpha=0.3)

    # Configure input current axis if it exists
    if ax_input is not None:
        ax_input.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
        ax_input.set_ylabel(r'$I_{in}(t)$', fontsize=13)
        title_input = f'{field_type} - Input current' if field_type else 'Input current'
        ax_input.set_title(title_input, fontsize=14)
        if plot_all:
            ax_input.legend(fontsize=10)
        ax_input.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        job_str = f'job{job_index}' if job_index is not None else 'all_jobs'
        filename = os.path.join(save_dir, f'current_vs_time_{timestamp}_{job_str}.png')

        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamp),
            plot_generating_method='plot_current',
            additional_information=job_str,
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close(fig)
    else:
        plt.show()

def plot_current_response(timestamp, job_index=None, save_plot=False, save_dir=None,
                          running_machine='laptop'):
    """
    Plot current response analysis with linear response theory comparison.

    Creates a 3-panel figure showing:
    - Panel 1: Input current I_in(t) and supercurrent response J(t) in units of J_c
                Plus linear response prediction I_linear(t) as dashed line
    - Panel 2: Vector potential A(t) vs time
    - Panel 3: Gap Δ(t) vs time

    Linear response computed using:
        Z(ω) = 1/(iω/L + 1) where L = Δ(0) * tanh(βΔ(0)/2)
        I_linear(ω) = I_in(ω) * 2*Z_T / (2*Z_T + Z(ω))

    Args:
        timestamp: Timestamp folder name
        job_index: Job index to plot (default None plots first job)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')

    Returns:
        dict: Contains times, currents, input_pulse, vector_potentials, gaps, linear_current
    """
    # Determine job index
    if job_index is None:
        job_index = 0

    # Load data
    data = load_simulation_data(timestamp, job_index, running_machine=running_machine)

    # Check if current-driven
    field_type = data['field_type']
    is_current_driven = field_type is not None and field_type.startswith('current_')

    if not is_current_driven:
        print("Warning: This simulation is not current-driven. plot_current_response is designed for current-driven simulations.")

    # Extract data
    times = data['times']
    gaps = data['gaps']
    currents = data['currents']
    vector_potentials = data['vector_potentials']
    T_c = data['T_c']
    input_pulse = data['save_data'].get('input_pulse', None)

    # Normalize to dimensionless units
    times_norm = times * T_c  # t * T_c (dimensionless)

    # Normalize currents to J_c (critical current)
    # For BCS: J_c ∝ Δ(0), use initial gap as reference
    gap_initial = np.abs(gaps[0])
    J_c = gap_initial  # Simple normalization: J_c ~ Δ(0)

    currents_norm = np.real(currents) / J_c
    input_pulse_norm = input_pulse / J_c if input_pulse is not None else None

    # Normalize gap to T_c
    gaps_norm = np.abs(gaps) / T_c

    # Vector potential (keep in original units for now)
    A_norm = np.real(vector_potentials) if np.iscomplexobj(vector_potentials) else vector_potentials

    # Compute linear response for current
    linear_current_norm = None
    if input_pulse is not None:
        # FFT parameters
        N_t = len(times)
        dt = times[1] - times[0]

        # Compute FFT of input pulse
        I_in_fft = np.fft.fft(input_pulse)
        freq = np.fft.fftfreq(N_t, d=dt)
        omega = 2 * np.pi * freq

        # Get temperature and compute inverse temperature β
        temperature = data['input_kwargs']['system_parameters']['temperature']
        beta = 1.0 / temperature

        # Compute inductance: L = Δ(0) * tanh(βΔ(0)/2)
        gap_0 = gap_initial
        L = gap_0 * np.tanh(beta * gap_0 / 2.0)

        # Compute impedance: Z(ω) = 1/(iω/L + 1)
        Z_omega = 1.0 / (1j * omega / L + 1.0)

        # Get circuit impedance Z_T from circuit_params if available
        circuit_params = data['input_kwargs'].get('circuit_params', None)
        if circuit_params is not None and 'Z_T' in circuit_params:
            Z_T = circuit_params['Z_T']
        else:
            # Default Z_T if not available (assume typical value)
            Z_T = 1.0
            print(f"Warning: Z_T not found in circuit_params, using default Z_T = {Z_T}")

        # Apply linear response transfer function: 2*Z_T / (2*Z_T + Z(ω))
        transfer_function = (2.0 * Z_T) / (2.0 * Z_T + Z_omega)

        # Multiply input current by transfer function
        I_linear_fft = I_in_fft * transfer_function

        # Inverse FFT to get linear response in time domain
        I_linear_t = np.fft.ifft(I_linear_fft)

        # Normalize to J_c
        linear_current_norm = np.real(I_linear_t) / J_c

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_current, ax_A, ax_gap = axes

    # Panel 1: Current response
    if input_pulse_norm is not None:
        ax_current.plot(times_norm, input_pulse_norm, linewidth=2.5,
                       label=r'$I_{in}(t)$', alpha=0.8, color='C0', linestyle='-')

    ax_current.plot(times_norm, currents_norm, linewidth=2.5,
                   label=r'$J(t)$ (response)', alpha=0.8, color='C1', linestyle='-')

    if linear_current_norm is not None:
        ax_current.plot(times_norm, linear_current_norm, linewidth=2.0,
                       label=r'$J_{linear}(t)$', alpha=0.6, color='C2', linestyle='--')

    ax_current.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    ax_current.set_ylabel(r'Current ($J_c$)', fontsize=13)
    ax_current.set_title(r'Current Response', fontsize=14)
    ax_current.legend(fontsize=10)
    ax_current.grid(True, alpha=0.3)

    # Panel 2: Vector potential
    ax_A.plot(times_norm, A_norm, linewidth=2.5, label=r'$A(t)$',
             alpha=0.8, color='C3')

    ax_A.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    ax_A.set_ylabel(r'$A(t)$', fontsize=13)
    ax_A.set_title(r'Vector Potential', fontsize=14)
    ax_A.legend(fontsize=10)
    ax_A.grid(True, alpha=0.3)

    # Panel 3: Gap evolution
    ax_gap.plot(times_norm, gaps_norm, linewidth=2.5, label=r'$|\Delta(t)|$',
               alpha=0.8, color='C4')

    ax_gap.set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    ax_gap.set_ylabel(r'$|\Delta(t)| / T_c$', fontsize=13)
    ax_gap.set_title(r'Gap Evolution', fontsize=14)
    ax_gap.legend(fontsize=10)
    ax_gap.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'current_response_{timestamp}_job{job_index}.png')

        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamp),
            plot_generating_method='plot_current_response',
            additional_information=f'job{job_index}',
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close(fig)
    else:
        plt.show()

    # Return data for further analysis
    return {
        'times': times,
        'times_norm': times_norm,
        'currents': currents_norm,
        'input_pulse': input_pulse_norm,
        'vector_potentials': A_norm,
        'gaps': gaps_norm,
        'linear_current': linear_current_norm,
        'J_c': J_c,
        'T_c': T_c
    }

def plot_gap_current_combined(timestamp, job_index=None, equilibrium_timestamp=None,
                              n_average=100, save_plot=False, save_dir=None,
                              times_external=None, gaps_external=None,
                              currents_external=None, A_external=None,
                              labels_external=None, align_pulses=True):
    """
    Plot gap, current, and vector potential vs time in a combined 3-panel figure.

    Optionally overlay equilibrium gap(Q) and current(Q) interpolated at instantaneous Q(t).
    Can also overlay external data for comparison.

    Args:
        timestamp: Timestamp for time evolution data
        job_index: Job index (default None plots all jobs in folder)
        equilibrium_timestamp: Optional timestamp for equilibrium sweep data
        n_average: Number of timesteps to average for equilibrium data (default 100)
        save_plot: Whether to save plot (default False)
        save_dir: Directory for plots (default None uses Figures/ in project folder)
        times_external: Optional external time values for comparison. Can be:
                       - Single array: plots one external dataset
                       - List of arrays: plots multiple external datasets
        gaps_external: Optional external gap values. Can be:
                      - Single array: plots one external dataset
                      - List of arrays: plots multiple external datasets (must match times_external length)
        currents_external: Optional external current values. Can be:
                          - Single array: plots one external dataset
                          - List of arrays: plots multiple external datasets (must match times_external length)
        A_external: Optional external vector potential values. Can be:
                   - Single array: plots one external dataset
                   - List of arrays: plots multiple external datasets (must match times_external length)
        labels_external: Optional labels for external datasets. Can be:
                        - Single string: label for single external dataset
                        - List of strings: labels for multiple external datasets (must match times_external length)
                        - If not provided, defaults to 'External data' or 'External 1', 'External 2', etc.
        align_pulses: If True, shift external time axes to align pulse maxima (default True)
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

    # Create figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Load equilibrium data if provided and validate parameters
    if equilibrium_timestamp is not None:
        # Load parameters from both timestamps for validation
        input_kwargs_evolution, _ = load_job_data(timestamp, job_indices[0])
        input_kwargs_eq, _ = load_job_data(equilibrium_timestamp, 0)

        # Extract parameters for comparison
        params_evolution = input_kwargs_evolution['system_parameters']
        params_eq = input_kwargs_eq['system_parameters']

        # Check and warn if parameters differ
        params_to_check = [
            ('eta', 'eta'),
            ('temperature', 'temperature'),
            ('critical_temperature', 'critical_temperature')
        ]

        print(f"\n{'='*60}")
        print(f"Parameter validation (evolution vs equilibrium):")
        print(f"{'='*60}")

        for param_name, key in params_to_check:
            val_evolution = params_evolution.get(key, None)
            val_eq = params_eq.get(key, None)

            if val_evolution != val_eq:
                print(f"WARNING: {param_name} differs!")
                print(f"  Evolution: {val_evolution}")
                print(f"  Equilibrium: {val_eq}")
            else:
                print(f"  {param_name}: {val_evolution} ✓")

        # Check time parameters from evolution kwargs
        if 'dt' in input_kwargs_evolution:
            dt_evolution = input_kwargs_evolution['dt']
            dt_eq = input_kwargs_eq.get('dt', None)
            if dt_evolution != dt_eq:
                print(f"WARNING: dt (time sampling) differs!")
                print(f"  Evolution: {dt_evolution}")
                print(f"  Equilibrium: {dt_eq}")
            else:
                print(f"  dt (time sampling): {dt_evolution} ✓")

        if 'num_timesteps' in input_kwargs_evolution:
            nt_evolution = input_kwargs_evolution['num_timesteps']
            nt_eq = input_kwargs_eq.get('num_timesteps', None)
            dt_evolution = input_kwargs_evolution.get('dt', 0)
            dt_eq = input_kwargs_eq.get('dt', 0)
            T_evolution = nt_evolution * dt_evolution
            T_eq = nt_eq * dt_eq if nt_eq is not None else 0

            if abs(T_evolution - T_eq) > 1e-10:
                print(f"WARNING: Total time duration differs!")
                print(f"  Evolution: {T_evolution:.6f} ({nt_evolution} steps × {dt_evolution})")
                print(f"  Equilibrium: {T_eq:.6f} ({nt_eq} steps × {dt_eq})")
            else:
                print(f"  Total time duration: {T_evolution:.6f} ✓")

        print(f"{'='*60}\n")

        # Load equilibrium sweep data directly without creating plots
        print("Loading equilibrium sweep data...")

        # Manually load and process equilibrium data
        lines = io.read_contents_readable_file(equilibrium_timestamp)
        job_no_eq = io.recover_job_no(lines)

        # Lists to store equilibrium data
        A_eq_list = []
        gap_eq_list = []
        current_eq_list = []

        # Load all jobs from equilibrium sweep
        for eq_job_idx in range(job_no_eq):
            eq_input_kwargs, eq_save_data = load_job_data(equilibrium_timestamp, eq_job_idx)

            # Extract vector potential parameter value
            eq_vector_potentials = eq_save_data['vector_potentials']
            A_param = np.max(np.abs(eq_vector_potentials))

            # Extract equilibrated gap and current (average last n_average timesteps)
            eq_gaps = eq_save_data['gaps']
            eq_currents = eq_save_data['currents']

            gap_eq_mean = np.mean(eq_gaps[-n_average:])
            current_eq_mean = np.mean(eq_currents[-n_average:])

            A_eq_list.append(A_param)
            gap_eq_list.append(gap_eq_mean)
            current_eq_list.append(current_eq_mean)

        # Convert to arrays and sort by A
        sort_indices = np.argsort(A_eq_list)
        A_eq = np.array(A_eq_list)[sort_indices]
        gap_eq_means = np.array(gap_eq_list)[sort_indices]
        current_eq_means = np.array(current_eq_list)[sort_indices]

        print(f"Loaded equilibrium data: {len(A_eq)} points from A={A_eq.min():.3f} to {A_eq.max():.3f}")
        print()

    field_type = None
    is_current_driven = False
    for job_idx in job_indices:
        # Load time evolution data with normalization constants
        data = load_simulation_data(timestamp, job_idx)

        # Extract normalized quantities
        times_norm = data['times']  # data['T_c']
        gap_norm = np.abs(data['gaps']) #/ data['T_c']
        current_norm = np.real(data['currents']) #/ data['J_0']

        # Vector potential (real part if complex)
        A_values = np.real(data['vector_potentials'])

        # Check for input pulse (for current-driven simulations)
        input_pulse = data['save_data'].get('input_pulse', None)

        # Store field type from first job
        if field_type is None:
            field_type = data['field_type']
            is_current_driven = field_type is not None and field_type.startswith('current_')

        # Debug output
        print(f"Job {job_idx}: Plotting {len(times_norm)} points")
        print(f"  Gap range: [{gap_norm.min():.3f}, {gap_norm.max():.3f}]")
        print(f"  Current range: [{current_norm.min():.3f}, {current_norm.max():.3f}]")
        print(f"  A range: [{A_values.min():.3e}, {A_values.max():.3e}]")
        print(f"  Time range: [{times_norm.min():.3f}, {times_norm.max():.3f}]")

        label = f'Job {job_idx}' if plot_all else None

        # Use consistent colors: blue for single job, color cycle for multiple jobs
        if plot_all:
            plot_color = f'C{job_idx}'
        else:
            plot_color = 'C0'  # Always blue for single job

        # Plot time evolution on each subplot
        axes[0].plot(times_norm, gap_norm, linewidth=2.5, label=label, alpha=0.8, color=plot_color)
        axes[1].plot(times_norm, current_norm, linewidth=2.5, label=label, alpha=0.8, color=plot_color)

        # Plot vector potential with label adjustment for current-driven
        A_label = label + ' (A)' if is_current_driven and input_pulse is not None and label is not None else label
        axes[2].plot(times_norm, A_values, linewidth=2.5, label=A_label, alpha=0.8, color=plot_color, linestyle='-')

        # Plot input pulse if current-driven
        if is_current_driven and input_pulse is not None:
            I_label = label + ' ($I_{in}$)' if label is not None else '$I_{in}$'
            axes[2].plot(times_norm, input_pulse, linewidth=2.0, label=I_label,
                        alpha=0.6, color=plot_color, linestyle='--')

        # Compute and plot equilibrium reference if available
        if equilibrium_timestamp is not None:
            # Normalize equilibrium data using time evolution normalization constants
            T_c = data['T_c']
            J_0 = data['J_0']

            gap_eq_norm = np.abs(gap_eq_means) #/ T_c
            current_eq_norm = np.abs(current_eq_means) #/ J_0

            # Get instantaneous |Q(t)|
            Q_mag = np.abs(data['vector_potentials'])

            # Interpolate equilibrium values at each Q(t) using linear interpolation
            # np.interp returns boundary values for points outside the range
            gap_eq_interp = np.interp(Q_mag, A_eq, gap_eq_norm)

            # For current: -sign(Q) * interpolated(|Q|)
            Q_sign = np.sign(np.real(data['vector_potentials']))
            current_eq_interp = -Q_sign * np.interp(Q_mag, A_eq, current_eq_norm)

            # Plot equilibrium reference curves
            eq_label = 'Equilibrium' if job_idx == job_indices[0] else None
            axes[0].plot(times_norm, gap_eq_interp, '--', linewidth=2,
                        label=eq_label, alpha=0.7, color='red')
            axes[1].plot(times_norm, current_eq_interp, '--', linewidth=2,
                        label=eq_label, alpha=0.7, color='red')

    # Plot external data if provided
    if times_external is not None:
        # Normalize external data to lists
        if not isinstance(times_external, list):
            times_list = [times_external]
            gaps_list = [gaps_external] if gaps_external is not None else [None]
            currents_list = [currents_external] if currents_external is not None else [None]
            A_list = [A_external] if A_external is not None else [None]
            if labels_external is None:
                label_list = ['External data']
            elif isinstance(labels_external, str):
                label_list = [labels_external]
            else:
                label_list = labels_external
        else:
            times_list = times_external
            gaps_list = gaps_external if gaps_external is not None else [None] * len(times_list)
            currents_list = currents_external if currents_external is not None else [None] * len(times_list)
            A_list = A_external if A_external is not None else [None] * len(times_list)
            if labels_external is None:
                label_list = [f'External {i+1}' for i in range(len(times_list))]
            else:
                label_list = labels_external

        # Get reference pulse maximum time from first job for alignment
        if align_pulses:
            ref_data = load_simulation_data(timestamp, job_indices[0])
            ref_A = np.abs(ref_data['vector_potentials'])
            ref_t_max = ref_data['times'][np.argmax(ref_A)]
            print(f"\nPulse alignment enabled:")
            print(f"  Reference pulse maximum at t = {ref_t_max:.3f}")

        # Plot each external dataset
        for i, (times_ext, gaps_ext, currents_ext, A_ext, label_ext) in enumerate(zip(times_list, gaps_list, currents_list, A_list, label_list)):
            color_idx = i + 2  # Offset to avoid colors used for main data and equilibrium

            # Compute time shift for pulse alignment
            time_shift = 0.0
            if align_pulses and A_ext is not None:
                # Find time of maximum in external A pulse
                ext_A_abs = np.abs(A_ext)
                ext_t_max = times_ext[np.argmax(ext_A_abs)]
                time_shift = ref_t_max - ext_t_max
                print(f"  External dataset '{label_ext}': t_max = {ext_t_max:.3f}, shift = {time_shift:+.3f}")

            # Apply time shift
            times_ext_shifted = times_ext + time_shift

            # Plot gap if provided
            if gaps_ext is not None:
                axes[0].plot(times_ext_shifted, gaps_ext, '--', linewidth=2,
                           label=label_ext, alpha=0.7, color=f'C{color_idx}')

            # Plot current if provided
            if currents_ext is not None:
                axes[1].plot(times_ext_shifted, currents_ext, '--', linewidth=2,
                           label=label_ext, alpha=0.7, color=f'C{color_idx}')

            # Plot vector potential if provided
            if A_ext is not None:
                axes[2].plot(times_ext_shifted, A_ext, '--', linewidth=2,
                           label=label_ext, alpha=0.7, color=f'C{color_idx}')

    # Configure gap subplot
    axes[0].set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    axes[0].set_ylabel(r'$|\Delta(t)| / T_c$', fontsize=13)
    axes[0].set_title('Gap evolution', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    if plot_all or equilibrium_timestamp is not None or times_external is not None:
        axes[0].legend(fontsize=10)

    # Configure current subplot
    axes[1].set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    axes[1].set_ylabel(r'$J(t) / J_0$', fontsize=13)
    axes[1].set_title('Current evolution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    if plot_all or equilibrium_timestamp is not None or times_external is not None:
        axes[1].legend(fontsize=10)

    # Configure vector potential subplot
    axes[2].set_xlabel(r'Time ($T_c^{-1}$)', fontsize=13)
    if is_current_driven:
        axes[2].set_ylabel(r'$A(t)$, $I_{in}(t)$', fontsize=13)
        title_A = f'{field_type} pulse' if field_type else 'Vector potential & input current'
    else:
        axes[2].set_ylabel(r'$A(t)$', fontsize=13)
        title_A = f'{field_type} pulse' if field_type else 'Vector potential'
    axes[2].set_title(title_A, fontsize=14)
    axes[2].grid(True, alpha=0.3)
    if plot_all or times_external is not None:
        axes[2].legend(fontsize=10)

    plt.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        job_str = f'job{job_index}' if job_index is not None else 'all_jobs'
        eq_str = f'_eq{equilibrium_timestamp}' if equilibrium_timestamp else ''
        filename = os.path.join(save_dir, f'gap_current_combined_{timestamp}_{job_str}{eq_str}.png')

        # Save with metadata using extended_savefig
        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamp),
            plot_generating_method='plot_gap_current_combined',
            additional_information=f'{job_str}{eq_str}',
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()


def plot_equilibrated_gap_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir=None, combine_timestamps=False, x_external=None, y_external=None, labels_external=None, add_point_labels=False):
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
        save_dir: Directory for plots (default None uses Figures/ in project folder)
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
    # Set labels based on parameter type
    if parameter == 'temperature':
        ax.set_xlabel(r'$T / T_c$', fontsize=13)
        ax.set_ylabel(r'$|\Delta| / T_c$', fontsize=13)
        ax.set_title(f'Equilibrated Gap vs Temperature', fontsize=14)
    elif parameter == 'vector_potential':
        ax.set_xlabel(r'$A / A_0$', fontsize=13)
        ax.set_ylabel(r'$|\Delta| / T_c$', fontsize=13)
        ax.set_title(f'Equilibrated Gap vs Vector Potential', fontsize=14)
    else:
        ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=13)
        ax.set_ylabel(r'$|\Delta| / T_c$', fontsize=13)
        ax.set_title(f'Equilibrated Gap vs {parameter.replace("_", " ").title()}', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0,2.0])
    plt.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'gap_vs_{parameter}.png')

        # Save with metadata using extended_savefig
        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamps[0]) if timestamps else 'multiple',
            plot_generating_method='plot_equilibrated_gap_vs_parameter',
            additional_information=f'parameter={parameter}',
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()

    return {'parameter_values': all_parameter_values, 'gap_means': all_gap_means,
            'gap_errors': all_gap_errors, 'max_gap': max_gap, 'max_parameter': max_param,
            'timestamp_data': timestamp_data}


def plot_equilibrated_current_vs_parameter(timestamps, parameter='temperature', n_average=100, save_plot=False, save_dir=None, combine_timestamps=False, x_external=None, y_external=None, labels_external=None, add_point_labels=False):
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
        save_dir: Directory for plots (default None uses Figures/ in project folder)
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
    # Set labels based on parameter type
    if parameter == 'temperature':
        ax.set_xlabel(r'$T / T_c$', fontsize=13)
        ax.set_ylabel(r'$|J| / J_0$', fontsize=13)
        ax.set_title(f'Equilibrated Current vs Temperature', fontsize=14)
    elif parameter == 'vector_potential':
        ax.set_xlabel(r'$A / A_0$', fontsize=13)
        ax.set_ylabel(r'$|J| / J_0$', fontsize=13)
        ax.set_title(f'Equilibrated Current vs Vector Potential', fontsize=14)
    else:
        ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=13)
        ax.set_ylabel(r'$|J| / J_0$', fontsize=13)
        ax.set_title(f'Equilibrated Current vs {parameter.replace("_", " ").title()}', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Use Figures folder in project directory if no save_dir specified
        if save_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_root, 'Figures')
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'current_vs_{parameter}.png')

        # Save with metadata using extended_savefig
        extended_savefig(
            fig,
            filename,
            data_source_timestamp=str(timestamps[0]) if timestamps else 'multiple',
            plot_generating_method='plot_equilibrated_current_vs_parameter',
            additional_information=f'parameter={parameter}',
            dpi=150,
            bbox_inches='tight'
        )
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()

    return {'parameter_values': all_parameter_values, 'current_means': all_current_means,
            'current_errors': all_current_errors, 'max_current': max_current, 'max_parameter': max_param,
            'timestamp_data': timestamp_data}


#* ============================================================================
#* INTERNAL PLOTTING HELPERS (called by main functions)
#* ============================================================================

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

    # Compute equilibrium Green's functions using Usadel methods
    import sys
    import os
    equilibrium_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equilibrium_python')
    if equilibrium_path not in sys.path:
        sys.path.insert(0, equilibrium_path)

    from Usadel_methods import UsadelEvolution
    from nambu_class import NambuTensor

    # Extract parameters
    system_params = input_kwargs['system_parameters']
    eta = system_params['eta']
    temperature = system_params['temperature']

    # Get gap from gap history at t=0
    gaps = save_data['gaps']
    gap = np.abs(gaps[0])

    # Grid parameters for UsadelEvolution
    grid_parameters = {
        'omega_grid': energy_grid,
        'energy_cutoff': np.max(np.abs(energy_grid)),
        'eta': eta
    }

    # System parameters for UsadelEvolution
    system_parameters_usadel = {
        'critical_temperature': system_params['critical_temperature'],
        'temperature': temperature,
        'current_maximum': 1.0
    }

    # Create UsadelEvolution object
    usadel = UsadelEvolution(grid_parameters, system_parameters_usadel)

    # Compute equilibrium Green's functions using Usadel methods with Q=0
    import jax.numpy as jnp

    # Thermal occupation function
    f_thermal = NambuTensor(usadel._thermal_occupation_numbers(temperature), 0)

    # Get h_r and convert to g_r
    hr_eq = usadel._get_hr(Q=0.0, delta=gap, sigma_r=None)
    gr_eq = usadel._hr2gr(hr_eq)

    # Compute gk from gr and f: gk = gr @ f + f @ gr_involution
    gk_eq = gr_eq @ f_thermal + f_thermal @ gr_eq._involution()

    # For 'f', just use the thermal occupation directly
    f_eq = f_thermal

    # Extract Pauli components (convert jax arrays to numpy)
    def extract_pauli_components_from_nambu(nambu_tensor, pauli_indices):
        """Extract Pauli components from NambuTensor."""
        components = []
        for idx in pauli_indices:
            component = nambu_tensor._trace(pauli_index=idx) / 2.0
            components.append(np.array(component))
        return components

    # Create tau_3 Pauli matrix for subtractions (broadcast to energy dimension)
    import jax.numpy as jnp
    energy_shape = gr_eq.data.shape[2:]  # Extract energy dimension shape
    tau_3 = NambuTensor(jnp.ones(energy_shape), pauli_channel=3)

    # Subtract asymptotic/thermal components before comparison
    # This isolates the non-trivial energy dependence
    if green_function_type == 'gr':
        # Subtract τ₃ from g^R (asymptotic unit matrix in particle-hole space)
        gr_eq_subtracted = gr_eq - tau_3
        eq_nambu = gr_eq_subtracted
    elif green_function_type == 'gk':
        # Subtract 2 * f_thermal * τ₃ from g^K (equilibrium FDT contribution)
        # Work around NambuTensor multiplication bug by accessing .data directly
        thermal_term_data = f_thermal.data * tau_3.data * 2.0
        thermal_term = NambuTensor(thermal_term_data, None)
        gk_eq_subtracted = gk_eq - thermal_term
        eq_nambu = gk_eq_subtracted
    elif green_function_type == 'f':
        # Subtract f_thermal from f (equilibrium thermal distribution)
        f_eq_subtracted = f_eq - f_thermal
        eq_nambu = f_eq_subtracted

    # Extract only the Pauli components that will be plotted
    eq_pauli_components = extract_pauli_components_from_nambu(eq_nambu, pauli_components)

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

        # Plot equilibrium Green's function (dashed lines)
        eq_component_idx = pauli_components.index(pauli_idx)
        eq_component = eq_pauli_components[eq_component_idx]
        eq_real = np.real(eq_component)
        eq_imag = np.imag(eq_component)

        ax.plot(energy_grid, eq_real, '--', color='black', linewidth=2,
               alpha=0.9, label='Equilibrium (Re)')
        ax.plot(energy_grid, eq_imag, '--', color='gray', linewidth=2,
               alpha=0.9, label='Equilibrium (Im)')

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
