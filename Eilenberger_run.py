import jax
import jax.numpy as jnp

import nambu_support
import custom_optimizer 
import parameter_generator 
import mesh_generator

import Eilenberger_compute

from tqdm import tqdm

def calc_equlibrium(temperature, params_dict, meshgrid,gr0 = None):
    ### This computes the equilibrium gap and Green's function (optionally) given initial guesses to pass to the solver 
    params_dict['temperature'] = temperature
    nambu_dict = mesh_generator.get_nambu_dict(meshgrid = meshgrid)
    f_tensor = parameter_generator.get_initial_occupation_numbers(nambu_identity = nambu_dict['nambu_matrices'][0], omega_grid = meshgrid['nambu_omega_grid'], temperature = params_dict['temperature'])
    gr = Eilenberger_compute._calc_gr(f = f_tensor,Q = params_dict['vector_potential'],nambu_dict= nambu_dict,eta = meshgrid['eta'], gap_function= params_dict['gap_symmetry'], params_dict = params_dict, meshgrid= meshgrid,self_energy_dict = params_dict['self_energy_coefficients'],gr0 = gr0) 

    gap = Eilenberger_compute._calc_gap(gr = gr, f = f_tensor, nambu_dict = nambu_dict, gap_function = params_dict['gap_symmetry'], meshgrid = meshgrid, critical_temperature = params_dict['critical_temperature'])
    sigma_r = Eilenberger_compute.compactify_sigma(Eilenberger_compute._sigma_r(gr = gr,nambu_dict = nambu_dict,self_energy_coefficients = params_dict['self_energy_coefficients']))

    return gap, gr, sigma_r
    


def equilibrium_sweep(temperatures, params_dict, meshgrid, gr0 = None):

    def temperature_sweep_function(temperature,gr0 = None):
        return calc_equlibrium(temperature = temperature, params_dict = params_dict, meshgrid = meshgrid,gr0 = gr0)

    jitted_run = jax.jit(temperature_sweep_function)

    gap,gr,sigma_r = jitted_run(temperatures[0],gr0 = gr0)
    print('first_gap', gap)
    gaps = []
    grs = []
    sigmas = []
    for temp in tqdm(temperatures):
        gap, gr, sigma_r = jitted_run(temp,gr0 = gr)
        gaps += [gap]
        grs += [gr]
        sigmas += [sigma_r]
    print(gaps)
    return gaps, grs, sigmas