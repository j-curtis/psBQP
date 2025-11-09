import jax
import jax.numpy as jnp

import nambu_support
import custom_optimizer 
import parameter_generator 
import mesh_generator

import Eilenberger_compute

from tqdm import tqdm

""" A module with functions which run the code in equilibrium 
"""

def calc_equlibrium(temperature: float, params_dict: dict, meshgrid: dict,gr0 = None)->tuple:
    """A function which computes the equilibrium gap and Green's function given parameters and meshgrid dictionaries as well as initial g0 guess

    Args:
        temperature (float): System temperature
        params_dict (dict): Dictionary containing all the system parameters 
        meshgrid (dict): Dictionary containing all the meshgrid parameters
        gr0 (jnp.ndarray, optional): Initial guess for Green's function. Defaults to None.

    Returns:
        tuple: Collection of equilibrium gap, Green's function, and self energy
    """
    ### This computes the equilibrium gap and Green's function (optionally) given initial guesses to pass to the solver 
    params_dict['temperature'] = temperature
    nambu_dict = mesh_generator.get_nambu_dict(meshgrid = meshgrid)
    f_tensor = parameter_generator.get_initial_occupation_numbers(nambu_identity = nambu_dict['nambu_matrices'][0], omega_grid = meshgrid['nambu_omega_grid'], temperature = params_dict['temperature'])
    gr = Eilenberger_compute._calc_gr(f = f_tensor,Q = params_dict['vector_potential'],nambu_dict= nambu_dict,eta = meshgrid['eta'], gap_function= params_dict['gap_symmetry'], params_dict = params_dict, meshgrid= meshgrid,self_energy_dict = params_dict['self_energy_coefficients'],gr0 = gr0) 

    gap = Eilenberger_compute._calc_gap(gr = gr, f = f_tensor, nambu_dict = nambu_dict, gap_function = params_dict['gap_symmetry'], meshgrid = meshgrid, critical_temperature = params_dict['critical_temperature'])
    sigma_r = Eilenberger_compute.compactify_sigma(Eilenberger_compute._sigma_r(gr = gr,nambu_dict = nambu_dict,self_energy_coefficients = params_dict['self_energy_coefficients']))

    return gap, gr, sigma_r
    


def equilibrium_sweep(temperatures: jnp.ndarray, params_dict: dict, meshgrid: dict, gr0 = None) -> tuple:
    """A function which computes the equilibrium for multiple values of temperature

    Args:
        temperatures (jnp.ndarray): Temperature list
        params_dict (dict): parameters dictionary
        meshgrid (dict): meshgrid dictionary
        gr0 (jnp.ndarray, optional): Initial guess for Green's function. Defaults to None.


    Returns:
        tuple: Collection of equilibrium gap, Green's function, and self energy for all the temperatures
    """

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