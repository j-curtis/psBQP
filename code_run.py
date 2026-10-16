"""
Real-time Keldysh evolution worker function for demler_tools framework.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pickle
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if 'SLURM_JOB_ID' in os.environ:
    sys.path.append('/cluster/work/demler/tools_library_source_code/v2.3.0-beta')
from demler_tools.file_manager import path_management, io

from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution
from numerical_support import ThermalConvolutions, interpolate_two_time_state
from driving_field_utils import compute_driving_field
from run_postprocessor import compute_state_rows


def evolve_keldysh_state(save_filename, num_timesteps, system_parameters, initial_state_timestamp=None, initial_state_index=0, grid_parameters=None, field_type=None, field_params=None, circuit_params=None, save_full_state=True, track_every_n=None, track_state=False, track_state_every_n=1):
    """
    Evolve Keldysh Green's functions in real time.

    Args:
        save_filename: Output file path (injected by demler_tools)
        num_timesteps: Number of time steps to evolve
        initial_state_timestamp: Optional timestamp to load initial state from
        initial_state_index: Index of result file (default 0)
        grid_parameters: Dict with time_sampling, time_duration, eta. If an initial state is loaded and
                         (time_sampling, time_duration) differ from the loaded state, the loaded gr, gk
                         (and occupation_function) are interpolated onto this grid. If None, the loaded grid is used.
        system_parameters: Dict with critical_temperature, temperature, eta
        driving_field: Optional time-dependent driving field. Can be:
                       - None: No driving (zero field)
                       - Scalar: Constant driving field applied at each step
                       - Array (length num_timesteps): Time-dependent field values
        field_type: Optional field type string ('constant', 'gaussian', 'oscillatory', 'DC', 'step')
                    If provided, field_params must also be given
                    'gaussian' can be pure Gaussian (frequency=0) or modulated (frequency>0)
                    For current-driven mode, use prefix 'current_' (e.g., 'current_gaussian')
        field_params: Optional dict with field-specific parameters
                      See driving_field_utils.compute_driving_field() for parameter details
        circuit_params: Optional dict with circuit parameters for current-driven evolution
                        Required fields: Z_T, R_n, J_DP_prime, I_DP
                        If provided, enables Crank-Nicolson vector potential update
        save_full_state: If True, save full time history. If False, save only final timestep (default: True)
        track_every_n: If not None, compute and save energy-time representations (gr, gk, f) every
                       track_every_n timesteps to reduce memory usage. If None, energy-time
                       representations are not computed (default: None)
        track_state: If True, the g^R and g^K normalization errors of the newest row (t = -1, t' = t) are computed during the
                     evolution and saved (default: False)
        track_state_every_n: Stride in timesteps of the normalization tracking (default 1)
    Returns:
        0 on success
    Saved data format:
        Dictionary with keys:
            - 'final_state': Final (or full) StateObject
            - 'gaps': Gap evolution Δ(t) (length num_timesteps)
            - 'currents': Supercurrent response J(t) (length num_timesteps)
            - 'vector_potentials': Vector potential A(t) (length num_timesteps)
            - 'times': Time array (length num_timesteps)
            - 'input_pulse': Applied driving field (A(t) or I_in(t) for current-driven)
            - 'gr_normalization', 'gk_normalization', 'normalization_times': (if track_state) (4, n) complex Pauli components of the
              normalization errors at (t, t') = (t_n, t_n) and the times t_n
            - 'gr_energy_time', 'gk_energy_time', 'f_energy_time': (if track_every_n is set)
    """
    #* Detect which machine we're running on
    if 'SLURM_JOB_ID' in os.environ:
        running_machine = 'cluster_euler'
        path_management.initialize_minimalistic(project_name='psBQP-keldysh', crt_controlling_machine='cluster_euler')
    else:
        running_machine = 'laptop'
        path_management.initialize(project_name='psBQP-keldysh')

    #* self-energies entering the evolution: EM (scattering rate 1.0) only when a vector potential is present
    self_energy_dictionary = {}
    if field_type is not None or circuit_params is not None:
        self_energy_dictionary['em'] = 1.0

    #* Load or generate initial state
    if initial_state_timestamp is not None:
        # Construct paths for kwargs and state files
        initial_state_dir = path_management.default_simulation_data_path(running_machine=running_machine)
        kwargs_file = os.path.join(initial_state_dir, str(initial_state_timestamp), path_management.args_input_filetag())
        state_file = os.path.join(initial_state_dir, str(initial_state_timestamp), f"{path_management.individual_result_file_tag()}{initial_state_index}")

        # If files not found, try autoarchive location (only for cluster)
        if not os.path.exists(kwargs_file) or not os.path.exists(state_file):
            if running_machine == 'cluster_euler':
                archive_dir = path_management.default_autoarchive_path(running_machine=running_machine)
                if not os.path.exists(kwargs_file):
                    kwargs_file = os.path.join(archive_dir, str(initial_state_timestamp), path_management.args_input_filetag())
                if not os.path.exists(state_file):
                    state_file = os.path.join(archive_dir, str(initial_state_timestamp), f"{path_management.individual_result_file_tag()}{initial_state_index}")

        # Load input kwargs from the initial state job using io
        kwargs_array = io.recover_full_calculation_arguments(full_path=kwargs_file)
        initial_state_kwargs = kwargs_array[initial_state_index]

        # Load the result data
        with open(state_file, 'rb') as f:
            initial_data = pickle.load(f)

        initial_state = initial_data['final_state']

        # Ensure occupation_function attribute exists (for backward compatibility with old saved states)
        if not hasattr(initial_state, 'occupation_function'):
            initial_state.occupation_function = None

        loaded_ntpoints = initial_state.gr.data.shape[2]
        loaded_T_max = initial_state.T_max
        # check if there is a grid missmatch, if so interpolate 
        grid_mismatch = (grid_parameters is not None and (grid_parameters['time_sampling'] != loaded_ntpoints or not np.isclose(grid_parameters['time_duration'], loaded_T_max)))

        if grid_mismatch:
            if loaded_ntpoints == 1 or initial_state.gr.data.shape[3] != loaded_ntpoints:
                raise ValueError("Loaded state holds a single time row (save_full_state=False) and cannot be interpolated onto a new grid")

            new_ntpoints = grid_parameters['time_sampling']
            new_T_max = grid_parameters['time_duration']
            
            #print(f"Interpolating loaded state: (N_t, T_max) = ({loaded_ntpoints}, {loaded_T_max}) -> ({new_ntpoints}, {new_T_max})")

            initial_state.gr = interpolate_two_time_state(initial_state.gr, loaded_T_max, new_T_max, new_ntpoints, hermitian=False)
            initial_state.gk = interpolate_two_time_state(initial_state.gk, loaded_T_max, new_T_max, new_ntpoints, hermitian=True)

            if initial_state.occupation_function is not None:
                initial_state.occupation_function = interpolate_two_time_state(initial_state.occupation_function, loaded_T_max, new_T_max, new_ntpoints, hermitian=True)

            grid_parameters = {'time_sampling': new_ntpoints, 'time_duration': new_T_max, 'eta': system_parameters['eta']}
            initial_state.T_max = new_T_max
            initial_state.dt = new_T_max / (new_ntpoints - 1)
        else:
            grid_parameters = {'time_sampling': loaded_ntpoints, 'time_duration': loaded_T_max, 'eta': system_parameters['eta']}

        # Apply temperature correction to gk if temperature has changed
        # Get old temperature from initial_state_kwargs instead of initial_state object
        old_temperature = initial_state_kwargs['system_parameters']['temperature']
        new_temperature = system_parameters['temperature']

        bcs_coupling_constant = initial_state.bcs_coupling_constant

        if not np.isclose(old_temperature, new_temperature):

            #~ thermal_dist_old = ThermalConvolutions(old_temperature, grid_parameters).thermal_dist
            #~ thermal_dist_new = ThermalConvolutions(new_temperature, grid_parameters).thermal_dist
            thermal_old = ThermalConvolutions(old_temperature, grid_parameters)
            thermal_new = ThermalConvolutions(new_temperature, grid_parameters)
            thermal_dist_old = thermal_old.thermal_dist
            thermal_dist_new = thermal_new.thermal_dist

            tau_3 = NambuKeldyshTensor(1.0, pauli_channel=3)

            #* the stored gk is g'' = g^K - 2 tau_3 F - T with T = -(Delta*Int_F + Int_F*Delta): the physical g^K is kept and g'' is re-expressed with the kernels of the new temperature
            #* gap history is read before gk is changed (with the old coupling), the gap is a function of time since the loaded state can be out of equilibrium
            gap_history = initial_state.get_gap_history()
            gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

            # Apply correction: gk_new = gk_old + 2 * tau_3 * (thermal_old - thermal_new) + (T_old - T_new)
            thermal_diff = thermal_dist_old - thermal_dist_new
            thermal_gap_diff = thermal_old.thermal_gap_term(gap_tensor) - thermal_new.thermal_gap_term(gap_tensor)
            #~ correction = 2.0 * tau_3 * thermal_diff
            correction = 2.0 * tau_3 * thermal_diff + thermal_gap_diff

            #* thermal_integral is set to zero on the coincident points, but its regularized value there is -(i/pi) ln T + (T independent), this is where the ln(Tc/T) of the coupling comes from.
            #* so the diagonal of T = -(Delta*Int_F + Int_F*Delta) changes by (2i/pi) ln(T_old/T_new) Delta(t), which changes Tr[tau_- gk(t,t)] by (2/pi) ln(T_old/T_new) Delta(t)
            diagonal_correction_data = np.zeros_like(initial_state.gk.data)
            diagonal_index = np.arange(initial_state.gk.data.shape[2])
            diagonal_correction_data[:, :, diagonal_index, diagonal_index] = (2j/np.pi) * np.log(old_temperature/new_temperature) * gap_tensor.data
            correction = correction + NambuKeldyshTensor(diagonal_correction_data)

            initial_state.gk = initial_state.gk + correction

            #* coupling of the new temperature, same expression as in generate_initial_state, such that the gap read from the shifted gk is the loaded one
            bcs_coupling_constant = -1 / (1/2/np.pi * np.log(system_parameters['critical_temperature']/new_temperature))

            # Determine if we need to track occupations
            track_occupations = (track_every_n is not None)

            if track_occupations:
                if initial_state.occupation_function is None:
                    initial_state.occupation_function = thermal_diff
                else:
                    initial_state.occupation_function = initial_state.occupation_function + thermal_diff


        dt = initial_state.dt
        # Determine if we need to track occupations
        track_occupations = (track_every_n is not None)

        # Save occupation_function before recreating StateObject
        saved_occupation = initial_state.occupation_function

        #~ initial_state = StateObject(gr=initial_state.gr, gk=initial_state.gk, track_occupation=track_occupations, bcs_coupling_constant= initial_state.bcs_coupling_constant, temperature=system_parameters['temperature'], grid_params=grid_parameters)
        initial_state = StateObject(gr=initial_state.gr, gk=initial_state.gk, track_occupation=track_occupations, bcs_coupling_constant=bcs_coupling_constant, temperature=system_parameters['temperature'], grid_params=grid_parameters)

        # Restore occupation_function after StateObject creation
        if saved_occupation is not None:
            initial_state.occupation_function = saved_occupation

    else:
        #* generating initial state using equilibrium code
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters, self_energy_dictionary=self_energy_dictionary)
        #~ initial_state, _, _, equilibrium_current = evolution.generate_initial_state(Q=0.0)
        #* equilibrium state at the vector potential the evolution starts with, A(t=0) = driving_field[0]
        if field_type is not None and field_params is not None:
            initial_vector_potential = float(compute_driving_field(field_type, field_params, np.zeros(1))[0])
        else:
            initial_vector_potential = 0.0
        initial_state, _, _, equilibrium_current = evolution.generate_initial_state(Q=initial_vector_potential)
        dt = grid_parameters['time_duration'] / (grid_parameters['time_sampling'] - 1)


    # Prepare driving field

    times = np.arange(num_timesteps) * dt

    if field_type is not None and field_params is not None:
        prepared_driving_field = compute_driving_field(field_type, field_params, times)
    else:
        prepared_driving_field = 0 * times

    # Run evolution
    if 'evolution' not in locals():
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters, self_energy_dictionary=self_energy_dictionary)

    # Determine if we should track occupations
    track_occupations = (track_every_n is not None)

    result = evolution.real_time_evolution(initial_state=initial_state, num_timesteps=num_timesteps, driving_field=prepared_driving_field, circuit_params=circuit_params, track_occupations=track_occupations, track_state=track_state, track_state_every_n=track_state_every_n)
    final_state = result['final_state']
    gaps = result['gaps']
    currents = result['currents']
    vector_potentials = result['vector_potentials']

    #* normalization and FDT rows need the full state, so they are computed before the state is reduced and saved with the result
    state_rows = None
    if not save_full_state:
        try:
            state_rows = compute_state_rows(final_state, evolution.thermal, grid_parameters['time_duration'])
        except Exception as error:
            print(f"WARNING: could not compute the state rows before reducing the state: {type(error).__name__}: {error}")

    # Reduce state to last timestep if requested (memory optimization)
    if not save_full_state:

        # Extract only the last time slice from gr and gk
        # Data shape is typically [..., N_omega, N_t] where N_t is time axis
        gr_last = NambuKeldyshTensor(final_state.gr[-1:,:])
        gk_last = NambuKeldyshTensor(final_state.gk.data[-1:,:])

        # Create reduced state with only last timestep
        # Preserve original grid parameters (time_sampling from grid_parameters, not 1)
        reduced_state = StateObject( gr=gr_last, gk=gk_last, bcs_coupling_constant=final_state.bcs_coupling_constant, temperature=system_parameters['temperature'], grid_params={'time_duration': final_state.T_max, 'time_sampling': grid_parameters['time_sampling'], 'dt': final_state.dt})
        final_state = reduced_state

    #! understand what this means
    #* circuit mode: gaps, currents and vector_potentials are all evaluated at t_(n+1), so the saved times are shifted by dt
    saved_times = times + dt if circuit_params is not None else times
    result_data = { 'final_state': final_state, 'gaps': gaps, 'currents': currents, 'vector_potentials': vector_potentials,'times': saved_times, 'input_pulse': prepared_driving_field, 'state_rows': state_rows}

    if track_state:
        result_data['gr_normalization'] = result['gr_normalization']
        result_data['gk_normalization'] = result['gk_normalization']
        result_data['normalization_times'] = saved_times[result['normalization_indices']]

    # Add energy-time data if tracking was enabled, filtered by track_every_n
    if track_occupations:
        # Filter to save only every track_every_n timestep
        gr_energy_time_full = result['gr_energy_time']
        gk_energy_time_full = result['gk_energy_time']
        f_energy_time_full = result['f_energy_time']

        # Select every track_every_n entry
        result_data['gr_energy_time'] = gr_energy_time_full[::track_every_n]
        result_data['gk_energy_time'] = gk_energy_time_full[::track_every_n]
        result_data['f_energy_time'] = f_energy_time_full[::track_every_n]
        # Also save the indices for reference
        result_data['energy_time_indices'] = np.arange(0, num_timesteps, track_every_n)

    with open(save_filename, 'wb') as f:
        pickle.dump(result_data, f)

    return 0

# ===============================================================================================
# Cluster backend interface (for demler_tools)
# ===============================================================================================

def run_1_point(input_directory, point_index):
    """Run a single simulation point from both cluster and local execution."""
    args_file = os.path.join(input_directory, path_management.args_input_filetag())

    # Load calculation arguments
    with open(args_file, 'rb') as f:
        args_array, kwargs_array = pickle.load(f)

    # Get current kwargs
    crt_kwargs = kwargs_array[point_index] if kwargs_array else {}

    # Run the solver
    return evolve_keldysh_state(**crt_kwargs)


if __name__ == "__main__":
    """Entry point for cluster execution."""
    import sys

    # Parse command-line arguments
    calculation_type = sys.argv[1]

    if calculation_type == 'main_simulation':
        input_directory = sys.argv[2]
        point_index = int(sys.argv[3])

        # Run the simulation
        run_1_point(input_directory=input_directory, point_index=point_index)
