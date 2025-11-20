import jax
import jax.numpy as jnp

# import custom classes
from Eilenberger_methods import EilenbergerEvolution
from nambu_class import NambuTensor
from system_state import SupercondctingState

from tqdm import tqdm
# to disable jax add this to all the files
#jax.config.update("jax_disable_jit", True)


def calculate_equilibrium(temp_list, grid_parameters, system_parameters, optimization_parameters = None, sigma_scatterings = None,  Q_list = None):

    # generate the mesh and initialize static system parameters
    eilenberger_object = EilenbergerEvolution(grid_parameters, system_parameters, optimization_parameters, sigma_scatterings = sigma_scatterings)

    gr0 = None

    if Q_list is None:
        Q_val = 0 
    else:
        Q_val = Q_list[0]

    self_consistent_states = []
    gaps = []

    @jax.jit
    def temperature_run(temp):
        self_consistent_state = eilenberger_object._run_temperature_computation(Q = Q_val, T = temp)
        gap = eilenberger_object._calc_gap(self_consistent_state)
        return gap

    vectorized_temperature_run = jax.vmap(temperature_run)
    return vectorized_temperature_run(temp_list)

"""

def equilibrium_sweep(temperatures: jnp.ndarray, params_dict: dict, meshgrid: dict, gr0 = None) -> tuple:
    
    #TODO: This can even be optimized using jax loops or just vectorization
    #TODO: see how to do this with g0 as input
    #TODO: remove Q from the params dict for later jitting
    
    # define a function which sweeps the temperature with explicility constant parameters and meshgrid 
    # this is crucial to be able to jit the function \
    def temperature_sweep_function(temperature,gr0 = None):
        return calc_equlibrium(temperature = temperature, params_dict = params_dict, meshgrid = meshgrid,gr0 = gr0)

    # jit the  function
    jitted_run = jax.jit(temperature_sweep_function)

    # compile the jitted function once
    gap,gr,sigma_r = jitted_run(temperatures[0],gr0 = gr0)
    print('first_gap', gap)
    gaps = []
    grs = []
    sigmas = []

    # run the for loop for all the temperatures 
    for temp in tqdm(temperatures):
        gap, gr, sigma_r = jitted_run(temp,gr0 = gr)
        gaps += [gap]
        grs += [gr]
        sigmas += [sigma_r]

    # this line here is to force compilation
    print('The gaps are',  gaps)

    return gaps, grs, sigmas 
"""