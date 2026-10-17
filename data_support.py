"""
Support functions for data analysis in Keldysh-Usadel simulations.

Helper functions for extracting observables, comparing runs, and processing results.
"""

import numpy as np
import os, sys, pickle
from demler_tools.file_manager import io, path_management

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


#* ============================================================================
#* DATA LOADING
#* ============================================================================

def load_job_data(timestamp, job_index, running_machine='laptop'):
    """Load raw simulation data (input_kwargs, save_data) using demler_tools."""

    if not hasattr(path_management, 'project_data'):
        path_management.initialize(project_name='psBQP-keldysh')

    simulation_data_path = path_management.default_simulation_data_path(running_machine=running_machine)
    kwargs_file = os.path.join(simulation_data_path, str(timestamp), path_management.args_input_filetag())
    result_file = os.path.join(simulation_data_path, str(timestamp), f"{path_management.individual_result_file_tag()}{job_index}")

    #* Check archive path for cluster runs if not found
    if not os.path.exists(kwargs_file) or not os.path.exists(result_file):
        if running_machine == 'cluster_euler':
            archive_path = path_management.default_autoarchive_path(running_machine=running_machine)
            if not os.path.exists(kwargs_file):
                kwargs_file = os.path.join(archive_path, str(timestamp), path_management.args_input_filetag())
            if not os.path.exists(result_file):
                result_file = os.path.join(archive_path, str(timestamp), f"{path_management.individual_result_file_tag()}{job_index}")

    if not os.path.exists(kwargs_file):
        raise FileNotFoundError(f"Kwargs file not found for timestamp {timestamp}")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found for timestamp {timestamp}, job {job_index}")

    kwargs_array = io.recover_full_calculation_arguments(full_path=kwargs_file)
    input_kwargs = kwargs_array[job_index]

    with open(result_file, 'rb') as f:
        save_data = pickle.load(f)

    return input_kwargs, save_data


#* ============================================================================
#* CRITICAL CURRENT EXTRACTION
#* ============================================================================

def extract_critical_current(job_run_timestamp, critical_current_timestamp, n_average=100, running_machine='laptop'):
    """
    Extract critical current from a vector potential sweep and compare with reference run.

    Loads data from two timestamps:
    1. job_run_timestamp: Reference run with specific parameters
    2. critical_current_timestamp: Vector potential sweep to extract critical current

    The function verifies that both timestamps have matching parameters (except for
    vector potential which is expected to be swept in critical_current_timestamp).
    It then finds the maximum current from the sweep (critical current) and assesses
    the quality of this determination by comparing with the second-highest value.

    Args:
        job_run_timestamp: Timestamp of reference job run
        critical_current_timestamp: Timestamp of vector potential sweep
        n_average: Number of final timesteps to average for equilibrated current (default 100)
        running_machine: Machine type for data loading ('laptop' or 'cluster_euler')

    Returns:
        float: Critical current value (maximum equilibrated current from sweep)

    Prints:
        - All run parameters from critical_current_timestamp
        - Critical current value (maximum current)
        - Second highest current value
        - Relative error estimate: (I_max - I_second) / I_max
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING CRITICAL CURRENT")
    print(f"{'='*70}\n")
    print(f"Job run timestamp:           {job_run_timestamp}")
    print(f"Critical current timestamp:  {critical_current_timestamp}")
    print(f"\n{'-'*70}")
    print(f"LOADING DATA")
    print(f"{'-'*70}\n")

    # Load reference job (job 0 from job_run_timestamp)
    ref_kwargs, ref_save_data = load_job_data(job_run_timestamp, job_index=0, running_machine=running_machine)

    # Load all jobs from critical current sweep
    lines = io.read_contents_readable_file(critical_current_timestamp)
    n_jobs_sweep = io.recover_job_no(lines)
    print(f"Number of jobs in critical current sweep: {n_jobs_sweep}\n")

    # Collect current vs vector potential data
    vector_potentials = []
    equilibrated_currents = []
    all_sweep_kwargs = []

    for job_idx in range(n_jobs_sweep):
        sweep_kwargs, sweep_save_data = load_job_data(critical_current_timestamp, job_idx, running_machine=running_machine)
        all_sweep_kwargs.append(sweep_kwargs)

        # Extract equilibrated current (average over last n_average timesteps)
        currents = sweep_save_data['currents']
        current_eq = np.mean(currents[-n_average:])
        equilibrated_currents.append(current_eq)

        # Extract vector potential (maximum absolute value)
        vp = sweep_save_data['vector_potentials']
        vp_max = np.max(np.abs(vp))
        vector_potentials.append(vp_max)

    vector_potentials = np.array(vector_potentials)
    equilibrated_currents = np.array(equilibrated_currents)

    print(f"{'-'*70}")
    print(f"PARAMETER COMPARISON")
    print(f"{'-'*70}\n")

    # Compare parameters between reference and sweep (using first sweep job)
    sweep_kwargs_first = all_sweep_kwargs[0]

    # Helper function to recursively compare nested dicts
    def compare_params(ref_dict, sweep_dict, path='', exclude_keys=None):
        """Compare two parameter dictionaries recursively, excluding specified keys."""
        if exclude_keys is None:
            exclude_keys = set()

        mismatches = []
        for key in ref_dict:
            if key in exclude_keys:
                continue

            full_path = f"{path}.{key}" if path else key

            if key not in sweep_dict:
                mismatches.append(f"  Missing in sweep: {full_path}")
                continue

            ref_val = ref_dict[key]
            sweep_val = sweep_dict[key]

            if isinstance(ref_val, dict) and isinstance(sweep_val, dict):
                # Recursively compare nested dicts
                mismatches.extend(compare_params(ref_val, sweep_val, full_path, exclude_keys))
            else:
                # Compare values
                try:
                    if isinstance(ref_val, (int, float)) and isinstance(sweep_val, (int, float)):
                        if not np.isclose(ref_val, sweep_val, rtol=1e-9):
                            mismatches.append(f"  {full_path}: {ref_val} != {sweep_val}")
                    elif ref_val != sweep_val:
                        mismatches.append(f"  {full_path}: {ref_val} != {sweep_val}")
                except (TypeError, ValueError):
                    if ref_val != sweep_val:
                        mismatches.append(f"  {full_path}: {ref_val} != {sweep_val}")
        return mismatches

    # Compare, excluding field_params which typically contains vector potential
    exclude_from_comparison = {'field_params', 'num_timesteps'}
    mismatches = compare_params(ref_kwargs, sweep_kwargs_first, exclude_keys=exclude_from_comparison)

    if mismatches:
        print("WARNING: Parameter mismatches detected:")
        for mismatch in mismatches:
            print(mismatch)
        print()
    else:
        print("✓ All parameters match (excluding field_params and num_timesteps)\n")

    print(f"{'-'*70}")
    print(f"RUN PARAMETERS (from critical_current_timestamp)")
    print(f"{'-'*70}\n")

    # Print all parameters from sweep
    def print_params_recursive(d, indent=0):
        """Recursively print parameter dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                print_params_recursive(value, indent + 1)
            else:
                print(f"{'  ' * indent}{key}: {value}")

    print_params_recursive(sweep_kwargs_first)

    print(f"\n{'-'*70}")
    print(f"CRITICAL CURRENT ANALYSIS")
    print(f"{'-'*70}\n")

    # Find maximum and second maximum currents
    abs_currents = np.abs(equilibrated_currents)
    sorted_indices = np.argsort(abs_currents)[::-1]  # Sort in descending order

    max_idx = sorted_indices[0]
    second_max_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]

    critical_current = abs_currents[max_idx]
    critical_vp = vector_potentials[max_idx]
    second_current = abs_currents[second_max_idx]
    second_vp = vector_potentials[second_max_idx]

    # Compute relative error
    relative_error = (critical_current - second_current) / critical_current if critical_current > 0 else 0.0

    print(f"Critical current (maximum):      {critical_current:.6e}")
    print(f"  at vector potential:           {critical_vp:.6e}")
    print(f"\nSecond highest current:          {second_current:.6e}")
    print(f"  at vector potential:           {second_vp:.6e}")
    print(f"\nRelative error estimate:")
    print(f"  (I_max - I_second) / I_max =   {relative_error:.6e}")
    print(f"  Percentage:                    {relative_error * 100:.4f}%")

    print(f"\n{'='*70}\n")

    return critical_current


#* ============================================================================
#* CRITICAL CURRENT EXTRACTION FROM POST-PROCESSED DATA
#* ============================================================================

def load_postprocessed_file(timestamp, directory_job_index=0, running_machine='laptop'):
    """Load one result file (index directory_job_index) of a post-processed timestamp folder."""

    if not hasattr(path_management, 'project_data'):
        path_management.initialize(project_name='psBQP-keldysh')

    simulation_data_path = path_management.default_simulation_data_path(running_machine=running_machine)
    result_file = os.path.join(simulation_data_path, str(timestamp), f"{path_management.individual_result_file_tag()}{directory_job_index}")

    #* Check archive path for cluster runs if not found
    if not os.path.exists(result_file) and running_machine == 'cluster_euler':
        archive_path = path_management.default_autoarchive_path(running_machine=running_machine)
        result_file = os.path.join(archive_path, str(timestamp), f"{path_management.individual_result_file_tag()}{directory_job_index}")

    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Post-processed file not found for timestamp {timestamp}, directory job index {directory_job_index}")

    with open(result_file, 'rb') as f:
        return pickle.load(f)


def _compare_parameter_dicts(reference, other, path=''):
    """Mismatches (list of strings) between two parameter dicts, compared recursively; None counts as an empty dict."""
    reference = reference or {}
    other = other or {}
    mismatches = []
    for key, reference_value in reference.items():
        full_path = f"{path}.{key}" if path else key
        if key not in other:
            mismatches.append(f"  Missing: {full_path}")
            continue
        other_value = other[key]
        if isinstance(reference_value, dict) and isinstance(other_value, dict):
            mismatches.extend(_compare_parameter_dicts(reference_value, other_value, full_path))
        elif isinstance(reference_value, (int, float, np.integer, np.floating)) and isinstance(other_value, (int, float, np.integer, np.floating)):
            if not np.isclose(reference_value, other_value, rtol=1e-9):
                mismatches.append(f"  {full_path}: {reference_value} != {other_value}")
        elif reference_value != other_value:
            mismatches.append(f"  {full_path}: {reference_value} != {other_value}")
    return mismatches


def extract_critical_current_postprocessed(critical_current_timestamp, directory_job_index=0, reference_timestamp=None, reference_directory_job_index=0,
                                           reference_job_index=0, running_machine='laptop'):
    """
    Extract the critical current from a vector potential sweep that has been post-processed with the 'static' keyword.

    The 'static' post-processing stores, for every inner job of a result file, the current and the vector potential averaged over the last 50 timesteps
    ('averaged_currents', 'averaged_vector_potentials'). The critical current is the maximum of |averaged current| over the sweep; its quality is judged
    from the second-highest value, as in extract_critical_current.

    Args:
        critical_current_timestamp: post-processed timestamp of the vector potential sweep
        directory_job_index: result file index (post-processing job) of that timestamp, or a list of indices whose inner jobs are combined (default 0)
        reference_timestamp: optional post-processed timestamp of the reference run, the system and grid parameters of its job reference_job_index in
                             result file reference_directory_job_index must match those of the sweep; if None the sweep is only checked for internal consistency
        running_machine: 'laptop' or 'cluster_euler'

    Returns:
        float: critical current (maximum of |averaged current|)

    Prints:
        parameters of the sweep, parameter comparison, the critical current and its vector potential, the second highest current and the relative error
        estimate (I_max - I_second) / I_max; a warning if the maximum lies at the largest vector potential of the sweep (the sweep does not pass the maximum)
    """
    file_indices = [int(directory_job_index)] if np.isscalar(directory_job_index) else [int(i) for i in directory_job_index]

    print(f"\n{'='*70}")
    print(f"EXTRACTING CRITICAL CURRENT FROM POST-PROCESSED DATA")
    print(f"{'='*70}\n")
    print(f"Critical current timestamp:  {critical_current_timestamp}, result file(s) {file_indices}")

    averaged_currents, averaged_vector_potentials, system_parameters, grid_parameters, field_parameters = [], [], [], [], []
    #* result file and inner job index of every sweep point, parameters of every file for the printout
    point_files, point_jobs, file_info = [], [], []
    for file_index in file_indices:
        data = load_postprocessed_file(critical_current_timestamp, file_index, running_machine)
        if 'averaged_currents' not in data:
            raise ValueError(f"Timestamp {critical_current_timestamp}, file {file_index} has no 'averaged_currents'; post-process it with compute_keywords=['static'] "
                             f"(keys present: {sorted(data.keys())})")
        n_jobs = len(data['averaged_currents'])
        averaged_currents += list(np.abs(np.asarray(data['averaged_currents'])))
        averaged_vector_potentials += list(np.abs(np.asarray(data['averaged_vector_potentials'])))
        system_parameters += list(data['system_parameters'])
        grid_parameters += list(data.get('grid_parameters', [None] * n_jobs))
        field_parameters += list(data.get('field_params', [None] * n_jobs))
        point_files += [file_index] * n_jobs
        point_jobs += list(range(n_jobs))
        file_info.append((file_index, n_jobs, data['system_parameters'][0], (data.get('grid_parameters') or [None])[0], list(data.get('field_params', [None] * n_jobs))))
    averaged_currents = np.array(averaged_currents)
    averaged_vector_potentials = np.array(averaged_vector_potentials)
    print(f"Number of jobs in the sweep: {len(averaged_currents)}\n")

    print(f"{'-'*70}")
    print(f"PARAMETER COMPARISON")
    print(f"{'-'*70}\n")

    if reference_timestamp is not None:
        reference_data = load_postprocessed_file(reference_timestamp, reference_directory_job_index, running_machine)
        reference_system = reference_data['system_parameters'][reference_job_index]
        reference_grid = reference_data.get('grid_parameters', [None] * len(reference_data['system_parameters']))[reference_job_index]
        label = f"reference run {reference_timestamp} (file {reference_directory_job_index}, job {reference_job_index})"
    else:
        reference_system, reference_grid = system_parameters[0], grid_parameters[0]
        label = "the first job of the sweep"
    mismatches = []
    for job, (system, grid) in enumerate(zip(system_parameters, grid_parameters)):
        mismatches += [f"job {job}, system_parameters" + line for line in _compare_parameter_dicts(reference_system, system)]
        mismatches += [f"job {job}, grid_parameters" + line for line in _compare_parameter_dicts(reference_grid, grid)]
    if mismatches:
        print(f"WARNING: parameter mismatches with {label}:")
        for mismatch in mismatches:
            print(mismatch)
        print()
    else:
        print(f"✓ system and grid parameters of all jobs match {label}\n")

    print(f"{'-'*70}")
    print(f"RUN PARAMETERS (per result file; the jobs of a file differ only in field_params)")
    print(f"{'-'*70}\n")
    for file_index, n_jobs, first_system, first_grid, file_fields in file_info:
        print(f"result file {file_index}: {n_jobs} jobs (inner job indices 0 .. {n_jobs - 1})")
        print(f"  system_parameters: {first_system}")
        print(f"  grid_parameters:   {first_grid}")
        for key in sorted({k for fields in file_fields if fields for k in fields}):
            values = [fields[key] for fields in file_fields if fields and key in fields]
            numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in values)
            if numeric and np.ptp(np.asarray(values, dtype=float)) > 0:
                print(f"  field_params['{key}']: varies from {min(values):.6g} to {max(values):.6g} over {len(values)} jobs")
            else:
                print(f"  field_params['{key}']: {values[0]} (constant)")
        print()
    print(f"averaged vector potential range of the sweep: [{averaged_vector_potentials.min():.6e}, {averaged_vector_potentials.max():.6e}]")

    print(f"\n{'-'*70}")
    print(f"CRITICAL CURRENT ANALYSIS")
    print(f"{'-'*70}\n")

    sorted_indices = np.argsort(averaged_currents)[::-1]
    max_idx = sorted_indices[0]
    second_max_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]

    critical_current = averaged_currents[max_idx]
    second_current = averaged_currents[second_max_idx]
    relative_error = (critical_current - second_current) / critical_current if critical_current > 0 else 0.0

    print(f"Critical current (maximum):      {critical_current:.6e}")
    print(f"  at vector potential:           {averaged_vector_potentials[max_idx]:.6e}")
    print(f"  taken from:                    timestamp {critical_current_timestamp}, result file (directory job index) {point_files[max_idx]}, inner job {point_jobs[max_idx]}")
    print(f"    system_parameters:           {system_parameters[max_idx]}")
    print(f"    grid_parameters:             {grid_parameters[max_idx]}")
    print(f"    field_params:                {field_parameters[max_idx]}")
    print(f"\nSecond highest current:          {second_current:.6e}")
    print(f"  at vector potential:           {averaged_vector_potentials[second_max_idx]:.6e}  (result file {point_files[second_max_idx]}, inner job {point_jobs[second_max_idx]})")
    print(f"\nRelative error estimate:")
    print(f"  (I_max - I_second) / I_max =   {relative_error:.6e}")
    print(f"  Percentage:                    {relative_error * 100:.4f}%")
    if max_idx == np.argmax(averaged_vector_potentials):
        print(f"\nWARNING: the maximum current is at the largest vector potential of the sweep, the sweep may not reach the maximum")

    print(f"\n{'='*70}\n")

    return float(critical_current)
