import jax 
import jax.numpy as jnp

import nambu_support 

""" A module containing all the functions used to generate the meshgrids and nambu spaces on these meshgrids
"""


def get_computation_grid(omega_sampling: int, theta_sampling: int, cutoff: float, fine_omega_sampling = None, fine_cutoff = None) -> dict:
    """A function returning the grid used for all the computations it makes sure the grids point numbers are even
    It returns a dictonary of all the possible sampling points

    Args:
        omega_sampling (int): number of sampling points in frequency space
        theta_sampling (int): number of sampling points in angle space
        cutoff (float): cutoff frequency
        fine_omega_sampling (int, optional): number of sampling points in small grid frequency space. Defaults to None.
        fine_cutoff (float, optional): cutoff frequency in small grid. Defaults to None.

    Returns:
        dict: A dictonary containing:
            omega_size (int): number of sampling points in frequency space
            theta_size (int): number of sampling points in angle space
            cutoff (float): cutoff frequency
            eta (float): broadening parameter used for causality
            fine_omega_size (int): number of sampling points in small grid frequency space
            fine_cutoff (float): cutoff frequency in small grid
            omega_grid (array): frequency meshgrid
            theta_grid (array): angle meshgrid
            omega_array (array): frequency array
            theta_array (array): angle array
            grid_shape (tuple): shape of the grid
            nambu_matrices(array): Pauli matrices on the grid dotted with identitty
            nambu_matrix_shape(tuple): shape of the Nambu matrices
            nambu_omega_grid(array): Frequency grid multiplied by Pauli matrices
            nambu_theta_grid(array): Angle grid multiplied by Pauli matrices
    """

    mesh_dict = {}
    # if the mesh is odd, add one point and make it even 
    mesh_dict['omega_size'] = omega_sampling if omega_sampling % 2 == 0 else omega_sampling + 1
    mesh_dict['theta_size'] = theta_sampling
    mesh_dict['cutoff'] = cutoff
    mesh_dict['eta'] = 2 * cutoff/omega_sampling # default value for broadening
    mesh_dict['fine_omega_size'] = fine_omega_sampling if fine_omega_sampling % 2 == 0 else fine_omega_sampling + 1
    mesh_dict['fine_cutoff'] = fine_cutoff

    # if the finer grid is specified, update the broadening accordingly
    if fine_omega_sampling is not None and fine_cutoff is not None:
        mesh_dict['eta'] = 2 * fine_cutoff/fine_omega_sampling

    #generate the grid
    omega_array, theta_array, omega_grid, theta_grid, grid_shape = generate_mesh_grid(omega_sampling = omega_sampling, theta_sampling = theta_sampling, cutoff = cutoff, fine_omega_sampling = fine_omega_sampling, fine_cutoff = fine_cutoff)

    mesh_dict['omega_grid'] = omega_grid # used to be self.w_grid
    mesh_dict['theta_grid'] = theta_grid # used to be self.theta_grid 
    mesh_dict['omega_array'] = omega_array #used to be self.w_arr
    mesh_dict['theta_array'] = theta_array # used to be self.theta
    mesh_dict['grid_shape'] = grid_shape # used to be self.grid_shape 

    # generate the nambu matrix grid 
    nambu_matrices_on_mesh, nambu_grid_shape, energy_array, angle_array = generate_nambu_space_grid(omega_grid,theta_grid)
    mesh_dict['nambu_matrices'] = nambu_matrices_on_mesh #used to be self.Nambu_matrices
    mesh_dict['nambu_matrix_shape'] = nambu_grid_shape #used to be self.Nambu_shape 

    mesh_dict['nambu_omega_grid'] = energy_array # used to be self.w
    mesh_dict['nambu_theta_grid'] = angle_array # used to be self.theta 

    return mesh_dict


def generate_mesh_grid(omega_sampling: int, theta_sampling: int, cutoff: float, fine_omega_sampling = None, fine_cutoff = None) -> tuple:
    """A function generating the meshgrid of frequency and angle

    Args:
        omega_sampling (int): number of sampling points in frequency space
        theta_sampling (int): number of sampling points in angle space
        cutoff (float): cutoff frequency
        fine_omega_sampling (int, optional): number of sampling points in small grid frequency space. Defaults to None.
        fine_cutoff (float, optional): cutoff frequency in small grid. Defaults to None.

    Returns:
        tuple: A tuple of the frequency and angle meshgrids as well as the pure arrays and the grid shape
    """
    # generate the desired array
    omega_array = jnp.linspace(-cutoff, cutoff, omega_sampling)
    theta_array = jnp.linspace(0., 2. * jnp.pi, theta_sampling, endpoint=False)

    # if fine grid is specified generate it
    fine_omega_array = None    
    if fine_omega_sampling is not None and fine_cutoff is not None:
        fine_omega_array = jnp.linspace(-fine_cutoff, fine_cutoff, fine_omega_sampling)
    
    # concatenate the arrays and remove duplicates 
    omega_array = jnp.concatenate((omega_array, fine_omega_array))
    omega_array = jnp.unique(omega_array)

    # generate the meshgrid of the energy and angle arrays
    omega_grid, theta_grid = jnp.meshgrid(omega_array, theta_array, indexing = 'ij')
    
    # get the shape of the meshgrid
    grid_shape = omega_grid.shape

    return omega_array, theta_array, omega_grid, theta_grid, grid_shape

def generate_nambu_space_grid(omega_grid: jnp.array,theta_grid: jnp.array) -> tuple:
    """A function generating the meshgrid outer product with the Pauli matrices

    Args:
        omega_grid (jnp.array): grid of frequencies
        theta_grid (jnp.array): grid of angles

    Returns:
        tuple: a tuple consisting of the mesh outer product with nambu matrices, its total shape, energy array outer product with the Pauli matrix and the angle array outer product with the Pauli matrices
    """
    # get all the pauli matrices
    pauli = nambu_support.get_pauli_matrix(None) # returns an array of pauli matrices

    # generate matrices on the meshgrid
    nambu_matrices_on_mesh = [ jnp.tensordot(sigma, jnp.ones_like(omega_grid),axes=0 ) for sigma in pauli ]
    nambu_matrix_shape = nambu_matrices_on_mesh[0].shape

    # generate the enegy and angle arrays in nambu space
    energy_array = nambu_support.nambu_scalar2nambu(omega_grid)
    angle_array = nambu_support.nambu_scalar2nambu(theta_grid)

    return nambu_matrices_on_mesh, nambu_matrix_shape, energy_array, angle_array


def get_nambu_dict(meshgrid: dict)-> dict:
    """A function returning the nambu dictionary given the meshgrid dictionary

    Args:
        meshgrid (dict): dictionary generated by the get_computation_grid function

    Returns:
        dict: dictionary containing only the nambu matrix parts
    """
    nambu_dict = {}
    nambu_dict['nambu_matrices'] = jnp.array(meshgrid['nambu_matrices'])
    nambu_dict['nambu_matrix_shape'] = meshgrid['nambu_matrix_shape']
    nambu_dict['nambu_omega_grid'] = meshgrid['nambu_omega_grid']
    nambu_dict['nambu_theta_grid'] = meshgrid['nambu_theta_grid']

    return nambu_dict