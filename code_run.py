"""
Real-time Keldysh evolution worker function for demler_tools framework.
"""

import numpy as np
import pickle
import os
import sys

from demler_tools.file_manager import path_management

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution


def evolve_keldysh_state(save_filename, num_timesteps, system_parameters, initial_state_timestamp=None, initial_state_index=0, grid_parameters=None, driving_field=None):
    """
    Evolve Keldysh Green's functions in real time.

    Args:
        save_filename: Output file path (injected by demler_tools)
        num_timesteps: Number of time steps to evolve
        initial_state_timestamp: Optional timestamp to load initial state from
        initial_state_index: Index of result file (default 0)
        grid_parameters: Dict with time_sampling, time_duration, eta
        system_parameters: Dict with critical_temperature, temperature, eta
        driving_field: Optional time-dependent driving field (length num_timesteps)
                       - None: No driving (zero field)
                       - Scalar: Constant driving field applied at each step
                       - Array (length num_timesteps): Time-dependent field values

    Returns:
        0 on success
    """
    # Load or generate initial state
    if initial_state_timestamp is not None:
        path_management.initialize(project_name='psBQP-keldysh')
        initial_state_dir = path_management.default_simulation_data_path(running_machine='laptop')
        state_file = os.path.join(initial_state_dir, str(initial_state_timestamp), f'sr_{initial_state_index}')

        with open(state_file, 'rb') as f:
            initial_data = pickle.load(f)

        initial_state = initial_data['final_state']
        grid_parameters = {'time_sampling': initial_state.gr.data.shape[2], 'time_duration': initial_state.T_max, 'eta': system_parameters['eta']}

        # Apply temperature correction to gk if temperature has changed
        if hasattr(initial_state, 'temperature'):
            old_temperature = initial_state.temperature
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

    else:
        #* generating initial state will be different
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
        Q_0 = 0.0
        if driving_field is not None:
            Q_0 = driving_field[0]           
        initial_state, _, _ = evolution.generate_initial_state(Q=Q_0)

    # Prepare driving field
    # Note: driving_field should have length num_timesteps (one value per evolution step)
    # NOT N_t (which is the history window size)
    if driving_field is None:
        prepared_driving_field = None
    elif isinstance(driving_field, (int, float, complex)):
        # Constant driving: create array with same value repeated
        prepared_driving_field = np.ones(num_timesteps, dtype=complex) * driving_field
    else:
        # Time-dependent driving: ensure correct length and dtype
        prepared_driving_field = np.asarray(driving_field, dtype=complex)
        if len(prepared_driving_field) != num_timesteps:
            raise ValueError(f"driving_field length ({len(prepared_driving_field)}) must equal num_timesteps ({num_timesteps})")

    # Generate time array for output
    # Times are relative, starting from 0
    # dt comes from the initial_state
    if hasattr(initial_state, 'dt'):
        dt = initial_state.dt
    else:
        # Fallback: compute from grid parameters
        dt = grid_parameters['time_duration'] / (grid_parameters['time_sampling'] - 1)

    times = np.arange(num_timesteps) * dt

    # Run evolution
    if 'evolution' not in locals():
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)

    final_state, gaps, currents, vector_potentials = evolution.real_time_evolution(
        initial_state=initial_state,
        num_timesteps=num_timesteps,
        driving_field=prepared_driving_field
    )

    # Save results
    result_data = {
        'final_state': final_state,
        'gaps': gaps,
        'currents': currents,
        'vector_potentials': vector_potentials,
        'times': times
    }

    with open(save_filename, 'wb') as f:
        pickle.dump(result_data, f)

    return 0
