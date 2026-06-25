"""
Real-time Keldysh evolution worker function for demler_tools framework.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pickle
import sys

from demler_tools.file_manager import path_management, io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution
from driving_field_utils import compute_driving_field


def evolve_keldysh_state(save_filename, num_timesteps, system_parameters, initial_state_timestamp=None, initial_state_index=0, grid_parameters=None, field_type=None, field_params=None, circuit_params=None, save_full_state=True, track_every_n=None):
    """
    Evolve Keldysh Green's functions in real time.

    Args:
        save_filename: Output file path (injected by demler_tools)
        num_timesteps: Number of time steps to evolve
        initial_state_timestamp: Optional timestamp to load initial state from
        initial_state_index: Index of result file (default 0)
        grid_parameters: Dict with time_sampling, time_duration, eta
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
            - 'gr_energy_time', 'gk_energy_time', 'f_energy_time': (if track_every_n is set)
    """
    #* Detect which machine we're running on
    if 'SLURM_JOB_ID' in os.environ:
        running_machine = 'cluster_euler'
        path_management.initialize_minimalistic(project_name='psBQP-keldysh', crt_controlling_machine='cluster_euler')
    else:
        running_machine = 'laptop'
        path_management.initialize(project_name='psBQP-keldysh')

    #* Load or generate initial state
    if initial_state_timestamp is not None:
        # Construct paths for kwargs and state files
        initial_state_dir = path_management.default_simulation_data_path(running_machine=running_machine)
        kwargs_file = os.path.join(initial_state_dir, str(initial_state_timestamp), 'inputs', path_management.args_input_filetag())
        state_file = os.path.join(initial_state_dir, str(initial_state_timestamp), path_management.relative_path_raw_result_file().format(initial_state_index))

        # If files not found, try autoarchive location (only for cluster)
        if not os.path.exists(kwargs_file) or not os.path.exists(state_file):
            if running_machine == 'cluster_euler':
                archive_dir = path_management.default_autoarchive_path(running_machine=running_machine)
                if not os.path.exists(kwargs_file):
                    kwargs_file = os.path.join(archive_dir, str(initial_state_timestamp), 'inputs', path_management.args_input_filetag())
                if not os.path.exists(state_file):
                    state_file = os.path.join(archive_dir, str(initial_state_timestamp), path_management.relative_path_raw_result_file().format(initial_state_index))

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

        grid_parameters = {'time_sampling': initial_state.gr.data.shape[2], 'time_duration': initial_state.T_max, 'eta': system_parameters['eta']}

        # Apply temperature correction to gk if temperature has changed
        # Get old temperature from initial_state_kwargs instead of initial_state object
        old_temperature = initial_state_kwargs['system_parameters']['temperature']
        new_temperature = system_parameters['temperature']

        if not np.isclose(old_temperature, new_temperature):
            # Create temporary evolution object to generate thermal distributions
            temp_evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)

            # Generate thermal distributions for both temperatures
            temp_evolution.get_thermal_occupation(old_temperature)
            thermal_dist_old = temp_evolution.thermal_dist

            temp_evolution.get_thermal_occupation(new_temperature)
            thermal_dist_new = temp_evolution.thermal_dist

            # Construct tau_3
            tau_3 = NambuKeldyshTensor(1.0, pauli_channel=3)

            # Apply correction: gk_new = gk_old + 2 * tau_3 * (thermal_old - thermal_new)
            thermal_diff = thermal_dist_old - thermal_dist_new
            correction = 2.0 * tau_3 * thermal_diff
            initial_state.gk = initial_state.gk + correction

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
        initial_state = StateObject(gr=initial_state.gr, gk=initial_state.gk, track_occupation=track_occupations, bcs_coupling_constant= initial_state.bcs_coupling_constant, grid_params=grid_parameters)

    else:
        #* generating initial state using equilibrium code
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
        initial_state, _, _, equilibrium_current = evolution.generate_initial_state(Q=0.0)
        dt = grid_parameters['time_duration'] / (grid_parameters['time_sampling'] - 1)


    # Prepare driving field
    # Note: driving_field should have length num_timesteps (one value per evolution step)
    # NOT N_t (which is the history window size)
    
    times = np.arange(num_timesteps) * dt

    # Priority: field_type/field_params > driving_field
    if field_type is not None and field_params is not None:
        prepared_driving_field = compute_driving_field(field_type, field_params, times)
    else:
        prepared_driving_field = 0 * times

    # Run evolution
    if 'evolution' not in locals():
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)

    # Determine if we should track occupations
    track_occupations = (track_every_n is not None)

    result = evolution.real_time_evolution(initial_state=initial_state, num_timesteps=num_timesteps, driving_field=prepared_driving_field, circuit_params=circuit_params, track_occupations=track_occupations)
    final_state = result['final_state']
    gaps = result['gaps']
    currents = result['currents']
    vector_potentials = result['vector_potentials']

    # Reduce state to last timestep if requested (memory optimization)
    if not save_full_state:

        # Extract only the last time slice from gr and gk
        # Data shape is typically [..., N_omega, N_t] where N_t is time axis
        gr_last = NambuKeldyshTensor(final_state.gr[-1:,:])
        gk_last = NambuKeldyshTensor(final_state.gk.data[-1:,:])

        # Create reduced state with only last timestep
        # Preserve original grid parameters (time_sampling from grid_parameters, not 1)
        reduced_state = StateObject( gr=gr_last, gk=gk_last, bcs_coupling_constant=final_state.bcs_coupling_constant, grid_params={'time_duration': final_state.T_max, 'time_sampling': grid_parameters['time_sampling'], 'dt': final_state.dt})

        final_state = reduced_state

    # Save results
    result_data = { 'final_state': final_state, 'gaps': gaps, 'currents': currents, 'vector_potentials': vector_potentials,'times': times, 'input_pulse': prepared_driving_field}

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
    # Try to load calculation arguments from new location (with /inputs/) first
    args_file = os.path.join(input_directory, 'inputs', 'calculation_arguments')

    # Fallback to old location (without /inputs/) for compatibility with local backend
    if not os.path.exists(args_file):
        args_file = os.path.join(input_directory, 'calculation_arguments')

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
