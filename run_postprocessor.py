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
from numerical_support import ThermalConvolutions


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


def _pauli_components(matrix_data):
    """(2, 2, ...) array of Nambu matrices -> (4, ...) Pauli components c_p of M = sum_p c_p tau_p."""
    m00, m01, m10, m11 = matrix_data[0, 0], matrix_data[0, 1], matrix_data[1, 0], matrix_data[1, 1]
    return np.array([(m00 + m11) / 2, (m01 + m10) / 2, 1j * (m01 - m10) / 2, (m00 - m11) / 2])


def wigner_energy_transform(antidiagonal_values, dt):
    """
    Same transform as StateObject.energy_time_representation, for values along the anti-diagonal already in the flipped order:
    antidiagonal_values[..., m] = g(t_m, t_(N-1-m)), i.e. the relative time s = t_m - t_(N-1-m) increases with m.
    Returns g(eps) = integral ds exp(i eps s) g(t_c + s/2, t_c - s/2) on the fftshifted energy grid eps = omega/2 of the energy representation.
    """
    n = antidiagonal_values.shape[-1]
    g_fft = np.fft.ifft(np.fft.ifftshift(antidiagonal_values, axes=-1), axis=-1) * n * dt
    return np.fft.fftshift(g_fft, axes=-1) * 2


def compute_energy_representation(save_data, system_parameters, times, gaps, energy_cutoff=10.0):
    """
    Energy-time representations tracked by code_run (track_every_n) turned into the physical functions.

    code_run stores the transforms of the evolved objects: g'^R (without the tau_3 delta(t - t') of g^R), g'' (g^K without 2 tau_3 f and without the
    thermal gap term T = -calF(t1 - t2) [Delta(t1) + Delta(t2)]) and the occupation n (without the thermal f). All are evaluated at the center time of the
    window, t_c = t_newest - T_max/2 (anti-diagonal of the two-time window). Here the full functions are rebuilt:
        g^R(eps)  = g'^R(eps) + tau_3
        g^K(eps)  = g''(eps) + 2 tau_3 tanh(eps / 2T) + T(eps, t_c),   T(eps, t_c) = Wigner transform of the thermal gap term, built from the gap history of the
                    window (before the start of the run the first gap value is used) and the thermal integral of ThermalConvolutions
        distribution: energy mode f_L = tanh(eps / 2T) + n_0, charge mode f_T = n_3 (n = n_0 tau_0 + n_3 tau_3)
    The s = 0 sample of the log-singular calF (odd N_t only) is replaced by its cell average -(i/pi)(ln(pi T dt/2) - 1).

    energy_cutoff: only the energies |eps| <= energy_cutoff * T_c are kept (the energy grid of the transform reaches pi/(2 dt)); None keeps the whole grid.
    The cut is applied to every array as soon as it is built, the thermal gap term is transformed on the full grid and cut afterwards.

    Returns:
        dict with 'energy_grid' (N_eps,), 'step_indices', 'step_times' (newest time of the window), 'center_times', 'temperature', 'thermal' (tanh),
        'gr', 'gk', 'occupation' ((n, 4, N_eps) complex Pauli components tau_0..tau_3; gr and gk are the full functions, occupation is n without the thermal part).
        The distribution functions follow as f_L = thermal + occupation[:, 0] and f_T = occupation[:, 3]
    """
    indices = np.asarray(save_data['energy_time_indices'])
    #~ energy_grid = np.asarray(save_data['gk_energy_time'][0]['energy_grid'])
    energy_grid_full = np.asarray(save_data['gk_energy_time'][0]['energy_grid'])
    n_t = len(energy_grid_full)
    #* energy window that is saved
    if energy_cutoff is None:
        keep = np.ones(n_t, dtype=bool)
    else:
        keep = np.abs(energy_grid_full) <= energy_cutoff * system_parameters.get('critical_temperature', 1.0)
    energy_grid = energy_grid_full[keep]
    times = np.asarray(times)
    dt = times[1] - times[0]
    window_duration = (n_t - 1) * dt
    temperature = system_parameters['temperature']

    def components(entries):
        #~ return np.array([_pauli_components(np.asarray(entry['g_energy'].data)) for entry in entries])
        return np.array([_pauli_components(np.asarray(entry['g_energy'].data))[..., keep] for entry in entries])

    gr_raw = components(save_data['gr_energy_time'])
    gk_raw = components(save_data['gk_energy_time'])
    occupation = components(save_data['f_energy_time'])

    #* thermal gap term along the anti-diagonal in the flipped order, T(t_m, t_(N-1-m)) = -calF(s_m) [Delta(t_m) + Delta(t_(N-1-m))]
    thermal = ThermalConvolutions(temperature, {'time_sampling': n_t, 'time_duration': window_duration})
    m = np.arange(n_t)
    calF = thermal.thermal_integral.data[0, 0][m, n_t - 1 - m].copy()
    if n_t % 2 == 1:
        calF[(n_t - 1) // 2] = -(1j / np.pi) * (np.log(np.pi * temperature * dt / 2) - 1.0)

    gap_series = np.asarray(gaps)
    gap_term = np.zeros_like(gk_raw)
    
    for position, step in enumerate(indices):
        gap_window = gap_series[np.clip(np.arange(step - n_t + 1, step + 1), 0, None)]
        gap_pair = gap_window[m] + gap_window[n_t - 1 - m]
        values = np.zeros((4, n_t), dtype=complex)
        values[2] = -calF * np.real(gap_pair)
        values[1] = -calF * np.imag(gap_pair)
        #~ gap_term[position] = wigner_energy_transform(values, dt)
        gap_term[position] = wigner_energy_transform(values, dt)[:, keep]

    thermal_part = np.tanh(energy_grid / (2 * temperature))
    #~ gr_full = gr_raw.copy()
    #~ gr_full[:, 3] += 1.0
    #~ gk_full = gk_raw + gap_term
    #~ gk_full[:, 3] += 2 * thermal_part
    #* the raw objects, the gap term and the distributions are not saved (they follow from the saved arrays and the gap history), the full functions are built in place
    gr_raw[:, 3] += 1.0
    gk_raw += gap_term
    gk_raw[:, 3] += 2 * thermal_part

    step_times = times[indices]
    #~ return {'energy_grid': energy_grid, 'step_indices': indices, 'step_times': step_times, 'center_times': step_times - window_duration / 2,
    #~         'temperature': temperature, 'thermal': thermal_part, 'gr_raw': gr_raw, 'gr': gr_full, 'gk_raw': gk_raw, 'gk': gk_full, 'gap_term': gap_term,
    #~         'occupation': occupation, 'distribution_L': thermal_part[None, :] + occupation[:, 0], 'distribution_T': occupation[:, 3]}
    return {'energy_grid': energy_grid, 'step_indices': indices, 'step_times': step_times, 'center_times': step_times - window_duration / 2,
            'temperature': temperature, 'thermal': thermal_part, 'gr': gr_raw, 'gk': gk_raw, 'occupation': occupation}


def compute_spectra(gaps, currents, vector_potentials, times):
    """
    Fourier transforms of the gap deviation, the current and the vector potential, with the time origin at the pulse (maximum of |A|).

    X(omega) = dt * sum_k x_k exp(-i omega (t_k - t_c)), t_c = times[argmax |A|] and t_k = times[k] (numpy's fft uses t_k = k dt, so the
    result of np.fft.fft is multiplied by exp(-i omega (times[0] - t_c))); for a symmetric pulse A(omega) is then real.
    The same factor multiplies all three transforms, so it drops out of ratios such as the conductivity J/(i omega A).

    Returns:
        dict with 'omega', 'gap_fft', 'current_fft', 'vpot_fft' (fftshifted, same length as times)
    """
    times = np.asarray(times)
    dt = times[1] - times[0]
    omega = 2 * np.pi * np.fft.fftfreq(len(times), d=dt)
    phase_correction = np.exp(-1j * omega * (times[0] - times[np.argmax(np.abs(vector_potentials))]))

    gap_abs = np.abs(gaps)
    return {'omega': np.fft.fftshift(omega),
            'gap_fft': np.fft.fftshift(np.fft.fft(gap_abs - gap_abs[0]) * dt * phase_correction),
            'current_fft': np.fft.fftshift(np.fft.fft(np.real(currents)) * dt * phase_correction),
            'vpot_fft': np.fft.fftshift(np.fft.fft(vector_potentials) * dt * phase_correction)}


def tracked_state_entries(save_data, system_parameters, times, gaps, energy_cutoff, job_idx):
    """
    Entries of one job that exist only for runs with track_state / track_every_n.

    Returns:
        (normalization, energy_representation): normalization is a dict with 'times', 'gr_normalization', 'gk_normalization' (None without track_state),
        energy_representation is the output of compute_energy_representation (None without track_every_n or on failure)
    """
    normalization = None
    if 'gr_normalization' in save_data:
        normalization = {'times': save_data['normalization_times'], 'gr_normalization': save_data['gr_normalization'],
                         'gk_normalization': save_data['gk_normalization']}

    energy_representation = None
    if 'gk_energy_time' in save_data and times is not None:
        try:
            energy_representation = compute_energy_representation(save_data, system_parameters, times, gaps, energy_cutoff)
        except Exception as e:
            print(f"Warning: Could not compute the energy representation for job {job_idx}: {type(e).__name__}: {e}")

    return normalization, energy_representation


def postprocess_keldysh_data(save_filename, timestamp, compute_keywords, energy_cutoff=10.0):
    """
    Postprocess Keldysh simulation data and extract observables.

    Processes all jobs in a timestamp folder and extracts requested observables
    based on keyword arguments. Results are saved to a pickle file for later analysis.

    Args:
        save_filename: Output pickle file path for results
        timestamp: Timestamp folder identifier (processes all jobs in folder)
        compute_keywords: List of strings specifying what to compute
                          Supported keywords:
                          - 'transmitivity': Compute gap, current, and transmitivity analysis; also the g^R / g^K normalization history ('normalization', needs
                            track_state=True), the energy-time representations ('energy_representation', needs track_every_n) and the circuit parameters
                          - 'state_evolution': gap, current, vector potential and times per job, their Fourier transforms (as in
                            'spectral_analysis'), if the run was made with track_state=True the g^R / g^K normalization errors
                            versus time ('normalization'), and if it was made with track_every_n the energy-time representations of g^R, g^K and the
                            occupation with the thermal parts and the thermal gap term added back ('energy_representation', see compute_energy_representation)
                          - 'spectral_analysis': gap, current and vector potential with their Fourier transforms
        energy_cutoff: energy window of the 'energy_representation' in units of T_c, only -energy_cutoff < eps/T_c < energy_cutoff is saved
                       (default 10, None = the whole energy grid, which reaches pi/(2 dt))

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
            - 'circuit_params': List of circuit parameter dicts (None for runs without a circuit)
            - 'normalization', 'energy_representation': Lists with one entry per job (None where not tracked), present only if at least one job has them
            Entries are None (arrays NaN) for jobs without a saved input pulse, so list positions equal the raw job indices.

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
        circuit_params_list = []
        normalization_list = []
        energy_list = []

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

            #~ # Handle missing input pulse
            #~ if input_pulse_raw is None:
            #~     continue

            # Store time series
            gap_evolution_list.append(gaps)
            current_evolution_list.append(np.real(currents))
            vector_potential_evolution_list.append(vector_potentials)
            times_list.append(times)
            #* jobs without a saved input pulse stay in the lists (NaN amplitudes) so that list positions equal the raw job indices
            incoming_current_pulses.append(None if input_pulse_raw is None else np.real(input_pulse_raw))
            circuit_params_list.append(input_kwargs.get('circuit_params', None))

            normalization, energy_representation = tracked_state_entries(save_data, system_parameters, times, gaps, energy_cutoff, job_idx)
            normalization_list.append(normalization)
            energy_list.append(energy_representation)

            # Compute transmitivity quantities
            if input_pulse_raw is None:
                I_in_max, J_max, transmitivity = np.nan, np.nan, np.nan
            else:
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
            'transmitivity': transmitivity,
            'circuit_params': circuit_params_list
        }
        #* only present if at least one job was run with track_state / track_every_n
        if any(entry is not None for entry in normalization_list):
            result_data['normalization'] = normalization_list
        if any(entry is not None for entry in energy_list):
            result_data['energy_representation'] = energy_list

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
        gap_fft_list, current_fft_list, vpot_fft_list, omega_list, T_c_list = [], [], [], [], []
        #* normalization versus time saved by code_run (track_state=True), None for runs without it
        normalization_list = []
        #* energy-time representations (code_run track_every_n), None for runs without them
        energy_list = []

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

            spectra = compute_spectra(gaps, currents, vector_potentials, times)
            gap_fft_list.append(spectra['gap_fft'])
            current_fft_list.append(spectra['current_fft'])
            vpot_fft_list.append(spectra['vpot_fft'])
            omega_list.append(spectra['omega'])
            T_c_list.append(system_parameters['critical_temperature'])

            #~ if 'gr_normalization' in save_data:
            #~     normalization_list.append({'times': save_data['normalization_times'], 'gr_normalization': save_data['gr_normalization'],
            #~                                'gk_normalization': save_data['gk_normalization']})
            #~ else:
            #~     normalization_list.append(None)
            #~
            #~ if 'gk_energy_time' in save_data and times is not None:
            #~     try:
            #~         energy_list.append(compute_energy_representation(save_data, system_parameters, times, gaps, energy_cutoff))
            #~     except Exception as e:
            #~         print(f"Warning: Could not compute the energy representation for job {job_idx}: {type(e).__name__}: {e}")
            #~         energy_list.append(None)
            #~ else:
            #~     energy_list.append(None)
            normalization, energy_representation = tracked_state_entries(save_data, system_parameters, times, gaps, energy_cutoff, job_idx)
            normalization_list.append(normalization)
            energy_list.append(energy_representation)

        # Package results
        result_data = {
            'system_parameters': system_parameters_list,
            'field_params': field_params_list,
            'gap_evolution': gap_evolution_list,
            'current_evolution': current_evolution_list,
            'vector_potential_evolution': vector_potential_evolution_list,
            'times': times_list
        }
        result_data.update({'gap_fft': gap_fft_list, 'current_fft': current_fft_list, 'vpot_fft': vpot_fft_list, 'omega': omega_list, 'T_c': T_c_list})
        #* only present if at least one job was run with track_state=True
        if any(entry is not None for entry in normalization_list):
            result_data['normalization'] = normalization_list
        #* only present if at least one job was run with track_every_n
        if any(entry is not None for entry in energy_list):
            result_data['energy_representation'] = energy_list

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

            #* Fourier transforms with the pulse at t = 0 (see compute_spectra)
            spectra = compute_spectra(gaps, currents, vector_potentials, times)
            omega_list.append(spectra['omega'])
            gap_fft_list.append(spectra['gap_fft'])
            current_fft_list.append(spectra['current_fft'])
            vpot_fft_list.append(spectra['vpot_fft'])

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
