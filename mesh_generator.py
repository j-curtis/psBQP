import jax 
import jax.numpy as jnp

import nambu_support 

def get_computation_grid(omega_sampling, theta_sampling, cutoff, fine_omega_sampling = None, fine_cutoff = None):
    mesh_dict = {}

    mesh_dict['omega_size'] = omega_sampling if omega_sampling % 2 == 0 else omega_sampling + 1
    mesh_dict['theta_size'] = theta_sampling
    mesh_dict['cutoff'] = cutoff
    mesh_dict['eta'] = 2 * cutoff/omega_sampling # default value for broadening
    mesh_dict['fine_omega_size'] = fine_omega_sampling if fine_omega_sampling % 2 == 0 else fine_omega_sampling + 1
    mesh_dict['fine_cutoff'] = fine_cutoff

    if fine_omega_sampling is not None and fine_cutoff is not None:
        mesh_dict['eta'] = 2 * fine_cutoff/fine_omega_sampling

    omega_array, theta_array, omega_grid, theta_grid, grid_shape = generate_mesh_grid(omega_sampling = omega_sampling, theta_sampling = theta_sampling, cutoff = cutoff, fine_omega_sampling = fine_omega_sampling, fine_cutoff = fine_cutoff)

    mesh_dict['omega_grid'] = omega_grid # used to be self.w_grid
    mesh_dict['theta_grid'] = theta_grid # used to be self.theta_grid 
    mesh_dict['omega_array'] = omega_array #used to be self.w_arr
    mesh_dict['theta_array'] = theta_array # used to be self.theta
    mesh_dict['grid_shape'] = grid_shape # used to be self.grid_shape 

    nambu_matrices_on_mesh, nambu_grid_shape, energy_array, angle_array = generate_nambu_space_grid(omega_grid,theta_grid)
    mesh_dict['nambu_matrices'] = nambu_matrices_on_mesh #used to be self.Nambu_matrices
    mesh_dict['nambu_matrix_shape'] = nambu_grid_shape #used to be self.Nambu_shape 

    mesh_dict['nambu_omega_grid'] = energy_array # used to be self.w
    mesh_dict['nambu_theta_grid'] = angle_array # used to be self.theta 

    return mesh_dict


def generate_mesh_grid(omega_sampling, theta_sampling, cutoff, fine_omega_sampling = None, fine_cutoff = None):
    
    omega_array = jnp.linspace(-cutoff, cutoff, omega_sampling)
    theta_array = jnp.linspace(0., 2. * jnp.pi, theta_sampling, endpoint=False)

    fine_omega_array = None    
    if fine_omega_sampling is not None and fine_cutoff is not None:
        fine_omega_array = jnp.linspace(-fine_cutoff, fine_cutoff, fine_omega_sampling)
    
    omega_array = jnp.concatenate((omega_array, fine_omega_array))
    omega_array = jnp.unique(omega_array)

    omega_grid, theta_grid = jnp.meshgrid(omega_array, theta_array, indexing = 'ij')
    
    grid_shape = omega_grid.shape

    return omega_array, theta_array, omega_grid, theta_grid, grid_shape

def generate_nambu_space_grid(omega_grid,theta_grid):
    pauli = nambu_support.get_pauli_matrix(None) # returns an array of pauli matrices

    nambu_matrices_on_mesh = [ jnp.tensordot(sigma, jnp.ones_like(omega_grid),axes=0 ) for sigma in pauli ]
    nambu_matrix_shape = nambu_matrices_on_mesh[0].shape

    energy_array = nambu_support.nambu_scalar2nambu(omega_grid)
    angle_array = nambu_support.nambu_scalar2nambu(theta_grid)

    return nambu_matrices_on_mesh, nambu_matrix_shape, energy_array, angle_array


def get_nambu_dict(meshgrid):
    nambu_dict = {}
    nambu_dict['nambu_matrices'] = jnp.array(meshgrid['nambu_matrices'])
    nambu_dict['nambu_matrix_shape'] = meshgrid['nambu_matrix_shape']
    nambu_dict['nambu_omega_grid'] = meshgrid['nambu_omega_grid']
    nambu_dict['nambu_theta_grid'] = meshgrid['nambu_theta_grid']

    return nambu_dict