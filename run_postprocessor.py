"""
Postprocessing module for Keldysh evolution data.

Extracts and saves observables from completed simulations without plotting.
"""

import os
import numpy as np
import pickle
import sys

from demler_tools.file_manager import path_management, io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_job_data(timestamp, job_index, running_machine='laptop'):
    """Load raw simulation data (input_kwargs, save_data) using demler_tools."""

    if not hasattr(path_management, 'project_data'):
        path_management.initialize(project_name='psBQP-keldysh')

    simulation_data_path = path_management.default_simulation_data_path(running_machine=running_machine)
    kwargs_file = os.path.join(simulation_data_path, str(timestamp), 'inputs', path_management.args_input_filetag())
    result_file = os.path.join(simulation_data_path, str(timestamp), path_management.relative_path_raw_result_file().format(job_index))

    # Check archive path for cluster runs if not found
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


def postprocess_keldysh_data(save_filename, timestamp, compute_keywords):
    """
    Postprocess Keldysh simulation data and extract observables.

    Processes all jobs in a timestamp folder and extracts requested observables
    based on keyword arguments. Results are saved to a pickle file for later analysis.

    Args:
        save_filename: Output pickle file path for results
        timestamp: Timestamp folder identifier (processes all jobs in folder)
        compute_keywords: List of strings specifying what to compute
                          Supported keywords:
                          - 'transmitivity': Compute gap, current, and transmitivity analysis

    Returns:
        0 on success

    Saved data format for 'transmitivity':
        Dictionary with keys:
            - 'system_parameters': List of system parameter dicts (one per job)
            - 'incoming_current_pulses': List of input current arrays I_in(t)
            - 'gap_evolution': List of gap evolution arrays Δ(t)
            - 'current_evolution': List of current evolution arrays J(t)
            - 'vector_potential_evolution': List of vector potential arrays A(t)
            - 'times': List of time arrays
            - 'incoming_current_amplitudes': Array of I_in_max values
            - 'transmitted_current_amplitudes': Array of J_max values
            - 'transmitivity': Array of transmitivity values (J_max/I_in_max)
    """
    # Detect which machine we're running on
    if 'SLURM_JOB_ID' in os.environ:
        running_machine = 'cluster_euler'
        path_management.initialize_minimalistic(project_name='psBQP-keldysh', crt_controlling_machine='cluster_euler')
    else:
        running_machine = 'laptop'
        path_management.initialize(project_name='psBQP-keldysh')

    # Check if timestamp exists in simulation data path or archive path
    simulation_data_path = path_management.default_simulation_data_path(running_machine=running_machine)
    timestamp_dir = os.path.join(simulation_data_path, str(timestamp))

    if not os.path.exists(timestamp_dir):
        archive_path = path_management.default_autoarchive_path(running_machine=running_machine)
        timestamp_dir = os.path.join(archive_path, str(timestamp))

        if not os.path.exists(timestamp_dir):
            raise FileNotFoundError(f"Timestamp directory {timestamp} not found in simulation data or archive paths")

    # Determine job count from timestamp folder
    # Pass full path to readable file instead of just timestamp
    readable_file_path = os.path.join(timestamp_dir, 'readable')
    lines = io.read_contents_readable_file(full_path=readable_file_path)
    job_no = io.recover_job_no(lines)
    job_indices = range(job_no)

    # Process based on keywords
    if 'transmitivity' in compute_keywords:
        # Initialize storage lists
        system_parameters_list = []
        field_params_list = []
        incoming_current_pulses = []
        gap_evolution_list = []
        current_evolution_list = []
        vector_potential_evolution_list = []
        times_list = []

        I_in_max_list = []
        J_max_list = []
        transmitivity_list = []

        # Loop over all jobs
        for job_idx in job_indices:
            # Load data
            input_kwargs, save_data = load_job_data(timestamp, job_idx, running_machine)

            # Extract system parameters and field parameters
            system_parameters = input_kwargs['system_parameters']
            field_params = input_kwargs.get('field_params', {})
            system_parameters_list.append(system_parameters)
            field_params_list.append(field_params)

            # Extract time series data from save_data
            gaps = save_data['gaps']
            currents = save_data['currents']
            vector_potentials = save_data['vector_potentials']
            times = save_data.get('times', None)
            input_pulse_raw = save_data.get('input_pulse', None)

            # Handle missing input pulse
            if input_pulse_raw is None:
                continue

            # Store time series
            gap_evolution_list.append(gaps)
            current_evolution_list.append(np.real(currents))
            vector_potential_evolution_list.append(vector_potentials)
            times_list.append(times)
            incoming_current_pulses.append(np.real(input_pulse_raw))

            # Compute transmitivity quantities
            I_in_max = np.max(np.abs(input_pulse_raw))
            J_max = np.max(np.abs(currents))
            transmitivity = J_max / I_in_max if I_in_max != 0 else 0

            I_in_max_list.append(I_in_max)
            J_max_list.append(J_max)
            transmitivity_list.append(transmitivity)

        # Convert to arrays
        incoming_current_amplitudes = np.array(I_in_max_list)
        transmitted_current_amplitudes = np.array(J_max_list)
        transmitivity = np.array(transmitivity_list)

        # Package results
        result_data = {
            'system_parameters': system_parameters_list,
            'field_params': field_params_list,
            'incoming_current_pulses': incoming_current_pulses,
            'gap_evolution': gap_evolution_list,
            'current_evolution': current_evolution_list,
            'vector_potential_evolution': vector_potential_evolution_list,
            'times': times_list,
            'incoming_current_amplitudes': incoming_current_amplitudes,
            'transmitted_current_amplitudes': transmitted_current_amplitudes,
            'transmitivity': transmitivity
        }

        # Save results
        with open(save_filename, 'wb') as f:
            pickle.dump(result_data, f)

    else:
        raise ValueError(f"Unknown compute keyword(s): {compute_keywords}")

    return 0
