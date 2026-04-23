import jax
import jax.numpy as jnp

# import custom classes
from Usadel_methods import UsadelEvolution
from nambu_class import NambuTensor
from system_state import SupercondctingState

from tqdm import tqdm
# to disable jax add this to all the files
#jax.config.update("jax_disable_jit", True)


def calculate_equilibrium(temp_list, grid_parameters, system_parameters, optimization_parameters = None, sigma_scatterings = None,  Q_list = None):

    usadel_object = UsadelEvolution(grid_parameters, system_parameters, optimization_parameters, sigma_scatterings = sigma_scatterings)

    gr0 = None

    if Q_list is None:
        Q_list = jnp.array([0])
    else:
        Q_val = Q_list[0]

    self_consistent_states = []
    
    @jax.jit
    def temperature_run(temp, crt_Q, gr):
        gr, gap, current = usadel_object._run_temperature_computation(Q = crt_Q, T = temp, gr0 = gr)
        return gr,gap, current

    gaps = jnp.zeros((len(temp_list),len(Q_list)),dtype = jnp.float32)

    #vectorized_temperature_run = jax.vmap(temperature_run)
    crt_gaps = []
    crt_currents = []
    for crt_Q_index in tqdm(range(len(Q_list))):
        for crt_temp_index in range(len(temp_list)):
            gr0, gap, current = temperature_run(temp_list[crt_temp_index], Q_list[crt_Q_index], gr0)
            crt_gaps += [jnp.abs(gap)]
            crt_currents += [current]

    return jnp.reshape(jnp.array(crt_gaps), (len(temp_list),len(Q_list))), jnp.reshape(jnp.array(crt_currents), (len(temp_list),len(Q_list)))

def calculate_phonon_scattering_scan(temp_list, scattering_list, grid_parameters, system_parameters, optimization_parameters = None, sigma_scatterings = None):

    gr0 = None 

    gaps = jnp.zeros((len(temp_list),len(scattering_list)),dtype = jnp.float32)

    crt_gaps = []
    crt_gs = []
    for scattering_rate in tqdm(scattering_list):
        for temperature in temp_list:
            sigma_scatterings['Phonon'] = scattering_rate
            system_parameters['temperature'] = temperature
            crt_usadel_object = UsadelEvolution(grid_parameters, system_parameters, optimization_parameters, sigma_scatterings = sigma_scatterings)
            gr0, gap,current = crt_usadel_object._run_temperature_computation(Q = 0.0, T = temperature, gr0 = gr0)
            crt_gaps += [jnp.abs(gap)]
            print(jnp.abs(gap))
            crt_gs += [gr0]

    return jnp.reshape(jnp.array(crt_gaps), (len(temp_list),len(scattering_list))), crt_gs



def real_time_evolve(temperature, current_function, grid_parameters, system_parameters, optimization_parameters = None, sigma_scatterings = None):

    usadel_object = UsadelEvolution(grid_parameters, system_parameters, optimization_parameters, sigma_scatterings = sigma_scatterings)

    gr0 = None

    timestamps = jnp.linspace(-grid_parameters['time_duration'] * 2, grid_parameters['time_duration'], grid_parameters['time_sampling'])
    ts_indices = jnp.arange(grid_parameters['time_sampling']-1)
    current_pulse = current_function(timestamps)

    current_evolution = []
    gap_evolution = []
    Q_list = []
    
    @jax.jit
    def time_run(crt_Q, gr):
        gr, gap, current = usadel_object._run_temperature_computation(Q = crt_Q, T = temperature, gr0 = gr)
        return gr,gap, current

    inital_state, inital_gap, inital_current = time_run(0.0, gr0)

    current_evolution += [inital_current]
    gap_evolution += [jnp.abs(inital_gap)]
    Q_list += [0.0]
    #TODO These pre-factors need to be figured out 
    max_current = system_parameters['current_maximum']  
    prefactor = 1
    if 'R_c/z_t' in system_parameters:
        #TODO: check! 
        contact_resistance_prefactor = 1 + 1* system_parameters['R_c/z_t']/2
    else:
        contact_resistance_prefactor = 1

    resistance_factor = 1/2/system_parameters['z_t/R_n'] #* removed 2/pi/system_parameters['z_t/R_n'] and made it current version -- to be checked

    for crt_time_index in tqdm(ts_indices):
        gr0, gap, current = time_run(Q_list[crt_time_index], gr0)
        gap_evolution += [jnp.abs(gap)]
        usadel_prefactor = usadel_object._calc_dtQ_prefactor_new(gr0,temperature) 
        #usadel_prefactor = 0.0 
        dQ = ((current_pulse[crt_time_index] - current * contact_resistance_prefactor) * max_current) * (timestamps[crt_time_index+1] - timestamps[crt_time_index])/(resistance_factor + usadel_prefactor * contact_resistance_prefactor) 
        #print('dQ/dt', dQ/ (timestamps[crt_time_index+1] - timestamps[crt_time_index]) )
        Q_list += [Q_list[crt_time_index] + dQ]
        current_evolution += [current + dQ/ (timestamps[crt_time_index+1] - timestamps[crt_time_index]) * usadel_prefactor/max_current]


    return timestamps, current_pulse, jnp.array(current_evolution), jnp.array(gap_evolution), jnp.array(Q_list)


def calculate_equilibrium_full(temp_list, grid_parameters, system_parameters, optimization_parameters = None, sigma_scatterings = None,  Q_list = None):

    usadel_object = UsadelEvolution(grid_parameters, system_parameters, optimization_parameters, sigma_scatterings = sigma_scatterings)

    gr0 = None

    if Q_list is None:
        Q_list = jnp.array([0])
    else:
        Q_val = Q_list[0]

    self_consistent_states = []
    
    @jax.jit
    def temperature_run(temp, crt_Q, gr):
        gr, gap, current = usadel_object._run_temperature_computation(Q = crt_Q, T = temp, gr0 = gr)
        return gr,gap, current

    gaps = jnp.zeros((len(temp_list),len(Q_list)),dtype = jnp.float32)

    #vectorized_temperature_run = jax.vmap(temperature_run)
    crt_gaps = []
    crt_currents = []
    grs = []
    for crt_Q_index in tqdm(range(len(Q_list))):
        for crt_temp_index in range(len(temp_list)):
            gr0, gap, current = temperature_run(temp_list[crt_temp_index], Q_list[crt_Q_index], gr0)
            crt_gaps += [jnp.abs(gap)]
            crt_currents += [current]
            grs+= [gr0]

    return jnp.reshape(jnp.array(crt_gaps), (len(temp_list),len(Q_list))), jnp.reshape(jnp.array(crt_currents), (len(temp_list),len(Q_list))), grs
