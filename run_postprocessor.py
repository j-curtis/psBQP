"""
Postprocessing module for Keldysh evolution data.

Extracts and saves observables from completed simulations without plotting.
"""

import os
import numpy as np
import pickle
import sys

if 'SLURM_JOB_ID' in os.environ:
    sys.path.append('/cluster/work/demler/tools_library_source_code/v2.3.0-beta')
from demler_tools.file_manager import path_management, io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from usadel_keldysh_evolution import UsadelKeldyshEvolution


def load_job_data(timestamp, job_index, running_machine='laptop'):
    """Load raw simulation data (input_kwargs, save_data) using demler_tools."""

    if not hasattr(path_management, 'project_data'):
        path_management.initialize(project_name='psBQP-keldysh')

    simulation_data_path = path_management.default_simulation_data_path(running_machine=running_machine)
    kwargs_file = os.path.join(simulation_data_path, str(timestamp), path_management.args_input_filetag())
    result_file = os.path.join(simulation_data_path, str(timestamp), f"{path_management.individual_result_file_tag()}{job_index}")

    # Check archive path for cluster runs if not found
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


def compute_state_rows(final_state, thermal, time_duration=None):
    """
    Rows at t = -1 versus t' of a final state: g^R, g^K and the g^R normalization, Keldysh normalization and FDT error rows.

    Args:
        final_state: StateObject
        thermal: ThermalConvolutions object (evolution.thermal) for the temperature of the run
        time_duration: T_max, only used if the state has no time grid

    Returns:
        dict with 't_prime' (N_t,) and 'gr_row', 'gk_row', 'gr_normalization_row', 'gk_normalization_row', 'fdt_row' ((4, N_t) complex,
        Pauli components 0..3), same content as the 'rows' entries saved by the 'static' postprocessing
    """
    def pauli_row(tensor_row):
        return np.array([tensor_row.trace(p) / 2 for p in range(4)]).reshape(4, -1)

    n_t = final_state.gr.data.shape[2]
    t_prime = final_state.time_grid if final_state.time_grid is not None else np.linspace(-time_duration, 0, n_t)

    _, gr_totals = final_state.check_gr_normalization(t1_idx=-1)
    _, gk_totals, _ = final_state.check_keldysh_normalization(-1, thermal)
    _, _, error_row, _ = final_state.check_fdt(thermal, time_index=-1)

    return {'t_prime': np.array(t_prime),
            'gr_row': pauli_row(final_state.gr[-1:, :]),
            'gk_row': pauli_row(final_state.gk[-1:, :]),
            'gr_normalization_row': np.array(gr_totals),
            'gk_normalization_row': np.array(gk_totals),
            'fdt_row': pauli_row(error_row)}


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

    Saved data format for 'static': system/field/grid parameters per job, averaged gap/current/vector potential
    (mean of the last 50 timesteps), 'gap_std' (standard deviation of |gap| over the same 50 timesteps) and the full
    'gap_evolution' / 'times' (list with one array per job),
    (N_jobs, 4) gr/gk normalization and FDT errors at (t=-1,t'=-1), and 'rows': list (one per job, None on failure) of dicts with
    't_prime', 'gr_row', 'gk_row', 'gr_normalization_row', 'gk_normalization_row', 'fdt_row' ((4, N_t) complex) at t = -1.
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

    if not os.path.exists(timestamp_dir) and running_machine == 'cluster_euler':
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

    elif 'state_evolution' in compute_keywords:
        # Initialize storage lists
        system_parameters_list = []
        field_params_list = []
        gap_evolution_list = []
        current_evolution_list = []
        vector_potential_evolution_list = []
        times_list = []

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

            # Store time series
            gap_evolution_list.append(gaps)
            current_evolution_list.append(np.real(currents))
            vector_potential_evolution_list.append(vector_potentials)
            times_list.append(times)

        # Package results
        result_data = {
            'system_parameters': system_parameters_list,
            'field_params': field_params_list,
            'gap_evolution': gap_evolution_list,
            'current_evolution': current_evolution_list,
            'vector_potential_evolution': vector_potential_evolution_list,
            'times': times_list
        }

        # Save results
        with open(save_filename, 'wb') as f:
            pickle.dump(result_data, f)

    elif 'spectral_analysis' in compute_keywords:
        # Initialize storage lists
        system_parameters_list = []
        field_params_list = []
        gap_evolution_list = []
        current_evolution_list = []
        vector_potential_evolution_list = []
        times_list = []
        gap_fft_list = []
        current_fft_list = []
        vpot_fft_list = []
        omega_list = []
        T_c_list = []

        # Loop over all jobs
        for job_idx in job_indices:
            # Load data
            input_kwargs, save_data = load_job_data(timestamp, job_idx, running_machine)

            # Extract system parameters and field parameters
            system_parameters = input_kwargs['system_parameters']
            field_params = input_kwargs.get('field_params', {})
            T_c = system_parameters['critical_temperature']

            system_parameters_list.append(system_parameters)
            field_params_list.append(field_params)
            T_c_list.append(T_c)

            # Extract time series data from save_data
            gaps = save_data['gaps']
            currents = save_data['currents']
            vector_potentials = save_data['vector_potentials']
            times = save_data.get('times', None)

            # Store time series
            gap_evolution_list.append(gaps)
            current_evolution_list.append(np.real(currents))
            vector_potential_evolution_list.append(vector_potentials)
            times_list.append(times)

            # Compute FFT (following plot_gap_evolution_multi_jobs conventions)
            dt = times[1] - times[0]
            N_t = len(times)

            # Frequency grid construction
            freq = np.fft.fftfreq(N_t, d=dt)
            omega = 2 * np.pi * freq
            omega_shifted = np.fft.fftshift(omega)
            omega_list.append(omega_shifted)

            # Phase correction (center pulse at t=0)
            t_center_idx = np.argmax(np.abs(vector_potentials))
            t_center = times[t_center_idx]
            phase_correction = np.exp(-1j * omega * t_center)

            # Gap FFT (operates on gap deviation - DC removal)
            gap_abs = np.abs(gaps)
            gap_deviation = gap_abs - gap_abs[0]
            gap_fft_raw = np.fft.fft(gap_deviation) * dt
            gap_fft_corrected = gap_fft_raw * phase_correction
            gap_fft_shifted = np.fft.fftshift(gap_fft_corrected)
            gap_fft_list.append(gap_fft_shifted)

            # Current FFT (use full time series - no DC removal)
            current_fft_raw = np.fft.fft(np.real(currents)) * dt
            current_fft_corrected = current_fft_raw * phase_correction
            current_fft_shifted = np.fft.fftshift(current_fft_corrected)
            current_fft_list.append(current_fft_shifted)

            # Vector potential FFT (use full time series)
            A_fft_raw = np.fft.fft(vector_potentials) * dt
            A_fft_corrected = A_fft_raw * phase_correction
            A_fft_shifted = np.fft.fftshift(A_fft_corrected)
            vpot_fft_list.append(A_fft_shifted)

        # Package results
        result_data = {
            'system_parameters': system_parameters_list,
            'field_params': field_params_list,
            'gap_evolution': gap_evolution_list,
            'current_evolution': current_evolution_list,
            'vector_potential_evolution': vector_potential_evolution_list,
            'times': times_list,
            'gap_fft': gap_fft_list,
            'current_fft': current_fft_list,
            'vpot_fft': vpot_fft_list,
            'omega': omega_list,
            'T_c': T_c_list
        }

        # Save results
        with open(save_filename, 'wb') as f:
            pickle.dump(result_data, f)

    elif 'static' in compute_keywords:
        # Initialize storage lists
        system_parameters_list = []
        field_params_list = []
        grid_parameters_list = []
        averaged_gaps = []
        gap_stds = []
        #* full gap time series and time axis of every job (saved by default, not only the average)
        gap_evolution_list = []
        times_list = []
        averaged_currents = []
        averaged_vector_potentials = []

        # FDT and normalization check storage (4 Pauli components per job)
        gr_normalization_errors_list = []  # List of arrays shape (4,)
        gk_normalization_errors_list = []  # List of arrays shape (4,)
        fdt_errors_list = []  # List of arrays shape (4,)

        #* rows at t = -1 versus t' (Pauli components, complex) for every job
        rows_list = []

        # Loop over all jobs
        for job_idx in job_indices:
            # Load data
            input_kwargs, save_data = load_job_data(timestamp, job_idx, running_machine)

            # Extract system parameters and field parameters
            system_parameters = input_kwargs['system_parameters']
            field_params = input_kwargs.get('field_params', {})
            grid_parameters = input_kwargs.get('grid_parameters', {})
            system_parameters_list.append(system_parameters)
            field_params_list.append(field_params)
            grid_parameters_list.append(grid_parameters)

            # Extract time series data from save_data
            gaps = save_data['gaps']
            currents = save_data['currents']
            vector_potentials = save_data['vector_potentials']

            gap_evolution_list.append(np.asarray(gaps))
            times_list.append(save_data.get('times', None))

            # Compute averages over last 50 timesteps
            avg_gap = np.mean(gaps[-50:])
            avg_current = np.mean(currents[-50:])
            avg_vpot = np.mean(vector_potentials[-50:])
            #* standard deviation of |gap| over the same 50 timesteps (error bar of the averaged gap)
            gap_std = np.std(np.abs(gaps[-50:]))

            # Store averaged values
            averaged_gaps.append(avg_gap)
            gap_stds.append(gap_std)
            averaged_currents.append(avg_current)
            averaged_vector_potentials.append(avg_vpot)

            #* rows computed in code_run before the state was reduced (save_full_state=False), they replace the checks below
            saved_rows = save_data.get('state_rows', None)
            if saved_rows is not None:
                gr_normalization_errors_list.append(saved_rows['gr_normalization_row'][:, -1])
                gk_normalization_errors_list.append(saved_rows['gk_normalization_row'][:, -1])
                fdt_errors_list.append(np.abs(saved_rows['fdt_row'][:, -1]))
                rows_list.append(saved_rows)
                continue

            # Compute FDT and normalization checks for last row [-1, :]
            try:
                # Load state object
                final_state = save_data.get('final_state', None)

                if final_state is not None:
                    # Create evolution object to get thermal distributions
                    #~ evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
                    #* self_energy_dictionary=None is not handled by the constructor; only the thermal kernels are needed here
                    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters, self_energy_dictionary={})
                    #~ evolution.get_thermal_occupation(system_parameters['temperature'])
                    #~ evolution.get_thermal_integral(system_parameters['temperature'])
                    #~ evolution.get_thermal_sum(system_parameters['temperature'])
                    #* thermal quantities are in evolution.thermal (numerical_support.ThermalConvolutions)

                    # Check g^R normalization at last row
                    gr_errors, gr_totals = final_state.check_gr_normalization(t1_idx=-1)
                    # gr_totals has shape (4, N_t), extract last element for each Pauli component
                    gr_errors_last = gr_totals[:, -1]  # Shape (4,)

                    # Check Keldysh normalization at last row
                    #~ gk_errors, gk_totals, gk_components = final_state.check_keldysh_normalization(
                        #~ -1, evolution.thermal_dist, evolution.thermal_integral,
                        #~ thermal_sum_left=evolution.thermal_sum_left,
                        #~ thermal_sum_right=evolution.thermal_sum_right
                    #~ )
                    gk_errors, gk_totals, gk_components = final_state.check_keldysh_normalization(-1, evolution.thermal)
                    # gk_totals has shape (4, N_t), extract last element for each Pauli component
                    gk_errors_last = gk_totals[:, -1]  # Shape (4,)

                    # Check FDT relation at last row
                    #~ gk_fdt_row, gk_actual_row, error_row, max_error = final_state.check_fdt(
                        #~ evolution.thermal_dist, evolution.thermal_integral, time_index=-1,
                        #~ thermal_sum_left=evolution.thermal_sum_left,
                        #~ thermal_sum_right=evolution.thermal_sum_right
                    #~ )
                    gk_fdt_row, gk_actual_row, error_row, max_error = final_state.check_fdt(evolution.thermal, time_index=-1)
                    # Extract FDT error for each Pauli component at last element
                    fdt_errors_pauli = np.zeros(4)
                    for pauli_idx in range(4):
                        error_pauli = error_row.trace(pauli_idx) / 2  # Extract Pauli component
                        if error_pauli.ndim == 2:
                            error_pauli = error_pauli[0, :]  # shape (N_t,)
                        fdt_errors_pauli[pauli_idx] = np.abs(error_pauli[-1])  # Value at last element

                    #* rows at t = -1: g^R, g^K and the three error rows, Pauli components 0..3 as (4, N_t) complex
                    #* (built before anything is appended so a failure leaves all lists consistent)
                    def pauli_row(tensor_row):
                        components = np.array([tensor_row.trace(p) / 2 for p in range(4)])
                        return components.reshape(4, -1)

                    t_prime = final_state.time_grid if final_state.time_grid is not None else np.linspace(-grid_parameters['time_duration'], 0, final_state.gr.data.shape[2])
                    row_entry = {
                        't_prime': np.array(t_prime),
                        'gr_row': pauli_row(final_state.gr[-1:, :]),
                        'gk_row': pauli_row(final_state.gk[-1:, :]),
                        'gr_normalization_row': np.array(gr_totals),
                        'gk_normalization_row': np.array(gk_totals),
                        'fdt_row': pauli_row(error_row)
                    }

                    gr_normalization_errors_list.append(gr_errors_last)
                    gk_normalization_errors_list.append(gk_errors_last)
                    fdt_errors_list.append(fdt_errors_pauli)
                    rows_list.append(row_entry)
                else:
                    # If state not available, store NaN
                    gr_normalization_errors_list.append(np.full(4, np.nan))
                    gk_normalization_errors_list.append(np.full(4, np.nan))
                    fdt_errors_list.append(np.full(4, np.nan))
                    rows_list.append(None)
            except Exception as e:
                print(f"Warning: Could not compute consistency checks for job {job_idx}: {e}")
                gr_normalization_errors_list.append(np.full(4, np.nan))
                gk_normalization_errors_list.append(np.full(4, np.nan))
                fdt_errors_list.append(np.full(4, np.nan))
                rows_list.append(None)

        # Convert to arrays
        averaged_gaps = np.array(averaged_gaps)
        gap_stds = np.array(gap_stds)
        averaged_currents = np.array(averaged_currents)
        averaged_vector_potentials = np.array(averaged_vector_potentials)
        gr_normalization_errors = np.array(gr_normalization_errors_list)  # Shape (N_jobs, 4)
        gk_normalization_errors = np.array(gk_normalization_errors_list)  # Shape (N_jobs, 4)
        fdt_errors = np.array(fdt_errors_list)  # Shape (N_jobs, 4)

        # Package results
        result_data = {
            'system_parameters': system_parameters_list,
            'field_params': field_params_list,
            'grid_parameters': grid_parameters_list,
            'averaged_gaps': averaged_gaps,
            'gap_std': gap_stds,
            'gap_evolution': gap_evolution_list,
            'times': times_list,
            'averaged_currents': averaged_currents,
            'averaged_vector_potentials': averaged_vector_potentials,
            'gr_normalization_errors': gr_normalization_errors,
            'gk_normalization_errors': gk_normalization_errors,
            'fdt_errors': fdt_errors,
            'rows': rows_list
        }

        # Save results
        with open(save_filename, 'wb') as f:
            pickle.dump(result_data, f)

    else:
        raise ValueError(f"Unknown compute keyword(s): {compute_keywords}")

    return 0
