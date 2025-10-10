import jax
import jax.numpy as jnp

import nambu_support

def get_system_parameters(temperature,mesh_dictionary,critical_temperature = 1, self_energy_coefficients = {},gap_symmetry = 's',vector_potential_params = None):
    param_dict = {}

    param_dict['temperature'] = temperature
    param_dict['critical_temperature'] = critical_temperature
 
    param_dict['vector_potential'] = get_vector_potential(vector_potential_params = vector_potential_params)   
    param_dict['gap_symmetry'] = generate_gap_symmetry_function(gap_symmetry = gap_symmetry, theta_grid= mesh_dictionary['nambu_theta_grid'])
    param_dict['bcs_coupling'] = get_BCS_coupling(cutoff = mesh_dictionary['cutoff'],critical_temperature = critical_temperature)

    param_dict['self_energy_coefficients'] = self_energy_coefficients

    #occupation_numbers = get_initial_occupation_numbers(nambu_identity = mesh_dictionary['nambu_matrices_on_mesh'][0], omega_grid = mesh_dictionary['nambu_omega_grid'], temperature = temperature)
    
    return param_dict

def generate_gap_symmetry_function(gap_symmetry,theta_grid):
    if gap_symmetry == 's':
        return jnp.ones_like(theta_grid)
    elif gap_symmetry == 'd-nodal':
        return jnp.sqrt(2.)*jnp.sin(2.*theta_grid)
    elif gap_symmetry == 'd-antinodal':
        return jnp.sqrt(2.)*jnp.cos(2.*theta_grid)
    else:
        raise ValueError("Invalid gap symmetry")


def get_BCS_coupling(cutoff,critical_temperature):
    ### This is a useful function which gives the relation between BCS lambda and Tc for a fixed cutoff in the case of clean s-wave BCS equation 
    return 1./jnp.log(nambu_support.get_bcs_gap_constant()*cutoff/critical_temperature) 

def get_vector_potential(vector_potential_params = None):
    # has to be generalized, since function should not be passed to jax jitted function
    # in general, given timestamps it should return the vector potential
    return 0

def get_initial_occupation_numbers(nambu_identity, omega_grid, temperature):
    return nambu_identity*jnp.tanh(0.5*omega_grid/temperature)
