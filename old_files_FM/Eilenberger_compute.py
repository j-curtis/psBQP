import jax
import jax.numpy as jnp

import nambu_support
import custom_optimizer 
import numpy as np
import parameter_generator


def _r2a(gr: jnp.ndarray,nambu_dict: dict) -> jnp.ndarray:
    """A function which takes g_r and computes g_a

    Args:
        gr (jnp.ndarray): retarded Green's function
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid

    Returns:
        jnp.ndarray: the advanced Green's function
    """

    #* Let's see how this works with small eta's etc. 
    ga = -jnp.transpose(jnp.conjugate(gr),axes=(1,0,2,3))
    
    ga = nambu_support.nambu_mul(nambu_dict['nambu_matrices'][3],ga)
    ga = nambu_support.nambu_mul(ga,nambu_dict['nambu_matrices'][3])
    
    return ga 

def _f2gk(gr: jnp.ndarray,f: jnp.ndarray,nambu_dict: dict) -> jnp.ndarray:
    """A function converting retarded green's function and the occupation function into the Keldysh Green's function

    Args:
        gr (jnp.ndarray): retarded Green's function
        f (jnp.ndarray): occupation function
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid

    Returns:
        jnp.ndarray: the Keldysh Green's function
    """
    ### This method takes gr and f object and computes the proper Keldysh correlation funciton
    #* Only correct if we ignore the other corrections which come from the convolution
    gk = nambu_support.nambu_mul(gr,f) - nambu_support.nambu_mul(f,_r2a(gr=gr,nambu_dict=nambu_dict)) 
    
    return gk 

def _hr2gr(hr: jnp.ndarray) -> jnp.ndarray:
    """A function normalizing g_r expression

    Args:
        hr (jnp.ndarray): Hamiltonian to be normalized

    Returns:
        jnp.ndarray: Normalized retarded Green's function
    """
    ### Inverts and normalizes a retarded effective Hamiltonian
    return -1.j* hr/jnp.sqrt(nambu_support.nambu_det(hr)) 
    #return np.sign(self.w)*hr/np.sqrt(-self._Nambu_det(hr))


def _Doppler_w_r(Q:float,nambu_dict: dict,eta: float) -> jnp.ndarray:
    """ A function which computes the tau_3 part of the Hamiltonian

    Args:
        Q (float): External vector potential
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        eta (float): The imaginary part of the retarded Green's function, used for broadening

    Returns:
        jnp.ndarray: Tau_3 component of the Hamiltonian
    """
    ### returns the Doppler shifted frequency Nambu tensor with retarded causality 	
    #return ( self.w - Q*np.cos(self.theta) + 0.5j*self.eta*np.ones_like(self.w) )*self.Nambu_matrices[3]
    return (nambu_dict['nambu_omega_grid'] - Q*jnp.cos(nambu_dict['nambu_theta_grid']) + 1.j*1e-8*jnp.ones_like(nambu_dict['nambu_omega_grid']) + 1.j*eta * jnp.ones_like(nambu_dict['nambu_omega_grid']))*nambu_dict['nambu_matrices'][3] 


def _Delta_p(gap,nambu_dict: dict,gap_function: jnp.ndarray)-> jnp.ndarray:
    """A function convering the gap scalar into a momentum resolved Nambu tensor given gap symmetry function

    Args:
        gap (complex): superconducting gap
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        gap_function (jnp.ndarray): The gap function describing symmetry in momentum space

    Returns:
        jnp.ndarray: Momentum resolved Nambu tensor
    """
    ### Returns the momentum resolved Nambu tensor gap  		
    ### Allows for a complex gap 
    return 1.j*jnp.real(gap) * nambu_dict['nambu_matrices'][2]*gap_function +1.j* jnp.imag(gap)*nambu_dict['nambu_matrices'][1]*gap_function

def _sigma_r(gr: jnp.ndarray,nambu_dict: dict,self_energy_coefficients: dict) -> jnp.ndarray:
    """A function returning the retarded self energy given a dictionary of self energy coefficients and the retarred Green's function

    Args:
        gr (jnp.ndarray): retarded Green's function
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        self_energy_coefficients (dict): Dictionary containing the self energy coefficients

    Returns:
        jnp.ndarray: retarded self energy
    """
    out = jnp.zeros(nambu_dict['nambu_matrix_shape'])
    if 'Dynes' in self_energy_coefficients:
        out += _sigma_r_dynes(nambu_dict=nambu_dict,dynes_eta=self_energy_coefficients['Dynes'])
    if 'Elastic' in self_energy_coefficients:
        out += _sigma_r_elastic(gr=gr,nambu_dict=nambu_dict,impurity_scattering=self_energy_coefficients['Elastic'])
    return out
    
def _sigma_r_elastic(gr: jnp.ndarray,nambu_dict: dict,impurity_scattering: float) -> jnp.ndarray:
    """A function computing the elastic impurity scattering self energy

    Args:
        gr (jnp.ndarray): retarded Green's function
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        impurity_scattering (float): impurity scattering rate

    Returns:
        jnp.ndarray: retarded self energy from impurity scattering
    """
    
    ### This method computes the retarded self energy from gr alone

    ### Impurity scattering contributions 
    sigma = -0.5j*impurity_scattering*jnp.mean(gr,axis=3,keepdims=True)
    return sigma 

def _sigma_r_dynes(nambu_dict: dict,dynes_eta: float) -> jnp.ndarray:
    """A function computing the Dynes inelastic scattering self energy

    Args:
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        dynes_eta (float): dynes inelastic scattering rate

    Returns:
        jnp.ndarray: retarded self energy from dynes inelastic scattering
    """
    
    sigma = 0.5j * dynes_eta * nambu_dict['nambu_matrices'][3]
    return sigma

def get_gr(Q:float,sigma_r: jnp.ndarray,delta, nambu_dict: dict, eta: float, gap_function: jnp.ndarray) -> jnp.ndarray:
    """A function which computes g_r given all the external parameters and the self-energies

    Args:
        Q (float): External vector potential
        sigma_r (jnp.ndarray): retarded self energy
        delta (_type_): superconducting gap 
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        eta (float): imaginary broadening introduced to the retarded Green's function
        gap_function (jnp.ndarray): The gap function describing symmetry in momentum space 

    Returns:
        jnp.ndarray: retarded Green's function
    """

    h_r_bare = _Doppler_w_r(Q=Q,nambu_dict=nambu_dict,eta=eta)
    # h_delta from gap 
    h_r_delta = lambda gap : _Delta_p(gap = gap, nambu_dict=nambu_dict,gap_function=gap_function)
    return  _hr2gr(hr = h_r_bare - sigma_r - h_r_delta(delta))

def _calc_gr(f: jnp.ndarray,Q: float, params_dict: dict, meshgrid: dict, nambu_dict: dict,eta: float,gap_function: jnp.ndarray, self_energy_dict: dict, gr0 = None, optimization_parameters = None) -> jnp.ndarray:
    """A function which self-consistently determines the retarded Green's function by solving the self-consistent gap and self energy equations

    Args:
        f (jnp.ndarray): Occupation function
        Q (float): External vector potential
        params_dict (dict): dictionary containing all the system parameters
        meshgrid (dict): a dictionary containing all the information about the meshgrid
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        eta (float): imaginary broadening introduced to the retarded Green's function
        gap_function (jnp.ndarray): The gap function describing symmetry in momentum space
        self_energy_dict (dict): a dictionary containing all the information about the self energy
        gr0 (jnp.ndarray, optional): initial guess for the retarded Green's function. Defaults to None.
        optimization_parameters (dict, optional): a dictionary containing all the information about the optimization parameters. Defaults to None.

    Returns:
        jnp.ndarray: self-consistent retarded Green's function
    """
    # define the bare Hamiltonian: epsling + Doppler
    
    sigma_r, delta = _self_consistent_delta_sigma(f = f, Q = Q, params_dict = params_dict, meshgrid=meshgrid, nambu_dict = nambu_dict, gr0 = gr0, optimization_parameters = optimization_parameters)

    return get_gr(Q = Q, sigma_r = sigma_r,delta = delta, nambu_dict = nambu_dict, eta = eta, gap_function = gap_function)

def _integrate(f: jnp.ndarray, meshgrid: dict) -> jnp.ndarray:
    """A function which integrates a scalar function f over the frequency and angle grids

    Args:
        f (jnp.ndarray): occupation function
        meshgrid (dict): a dictionary containing all the information about the meshgrid

    Returns:
        jnp.ndarray: integral of f
    """
    ### This method will integrate scalar function f over the frequency and angle grids (normalized by 2pi) assuming a possibly adaptive grid 
    ### For the moment we assume that f is a scalar and therefore already has had the Nambu indices traced out 
    
    ### We will simply sum this over all indices to return a single number 
    return custom_optimizer.custom_jax_trapz(jnp.mean(f,axis = 1),meshgrid['omega_array'])


def _calc_gap(gr: jnp.ndarray,f: jnp.ndarray,nambu_dict: dict,gap_function: jnp.ndarray,meshgrid: dict,critical_temperature = 1):
    """A function computing the gap by integrating gk 

    Args:
        gr (jnp.ndarray): retarded Green's function
        f (jnp.ndarray): occupation function
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        gap_function (jnp.ndarray): The gap function describing symmetry in momentum space
        meshgrid (dict): a dictionary containing all the information about the meshgrid
        critical_temperature (int, optional): critical temperature. Defaults to 1.

    Returns:
        complex: superconducting gap
    """
    
    ### This method computes the gap self consistently given the Greens function degree of freedom
    
    ### First we compute the propert Keldysh Green's function 
    gk = _f2gk(gr = gr, f = f, nambu_dict = nambu_dict)
    
    ### Now we compute the relevant Nambu trace 
    ### This will also reduce the tensor shape so we include inside this the gap function which is a tensor with the same shape as the Nambu tensors 
    #tr = np.trace( self.gap_function*self._NambuMul( 0.5*(self.Nambu_matrices[1] - 1.j*self.Nambu_matrices[2]), gk )  ) ### Trace should be over the nambu axes which are the first two axes and default for np.trace 

    integrand = (gap_function*gk)[0,1,:,:] ### We select the lower matrix element 
    
    #? not subtracting the Tc part, but everything is normalized according to Tc?
    ### Now we integrate over energy and frequency and multiply by BCS constant (factor of 0.25 is by definition of Keldysh part)
    return -0.25*parameter_generator.get_BCS_coupling(cutoff = meshgrid['cutoff'],critical_temperature = critical_temperature)*_integrate(f = integrand, meshgrid = meshgrid)### Call custom built integrator which is designed to handle adaptive grids 

def _self_consistent_delta_sigma(f: jnp.ndarray,Q: float,params_dict: dict, meshgrid: dict,nambu_dict: dict,gr0=None, optimization_parameters = None) -> jnp.ndarray:
    """A function which self-consistently solves the gap and the self-energy equations

    Args:
        f (jnp.ndarray): occupation function
        Q (float): External vector potential
        params_dict (dict): a dictionary containing all the information about the system parameters
        meshgrid (dict): a dictionary containing all the information about the meshgrid
        nambu_dict (dict): a dictionary containing all the information about nambu space and the meshgrid
        gr0 (jnp.ndarray, optional): initial guess for the retarded Green's function. Defaults to None.
        optimization_parameters (dict, optional): a dictionary containing all the information about the optimization parameters. Defaults to None.

    Returns:
        jnp.ndarray: self-consistent self-energy value
    """
    # define the bare Hamiltonian: epsling + Doppler
    h_r_bare = jnp.array(_Doppler_w_r(Q = Q, nambu_dict= nambu_dict, eta = meshgrid['eta']))

    # save the original shape of sigma_r
    original_sigma_r_shape = nambu_dict['nambu_matrix_shape']

    # initial guess for the sigma and gap
    if gr0 is None:
        gap_0 = 1.0 * nambu_support.get_bcs_ratio()*params_dict['critical_temperature']
        sigma_r_0 = compactify_sigma(_sigma_r(gr = _hr2gr(h_r_bare),nambu_dict = nambu_dict,self_energy_coefficients = params_dict['self_energy_coefficients']))
    else:
        gap_0 = _calc_gap(gr = gr0,f = f,nambu_dict = nambu_dict,gap_function = params_dict['gap_symmetry'],meshgrid = meshgrid,critical_temperature = params_dict['critical_temperature'])
        sigma_r_0 = compactify_sigma(_sigma_r(gr = gr0,nambu_dict = nambu_dict,self_energy_coefficients = params_dict['self_energy_coefficients']))

    sigma_r_sol, delta_sol = optimize_sigma_delta(Q = Q, f = f,sigma_0 = sigma_r_0,delta_0 = gap_0,params_dict = params_dict, meshgrid = meshgrid, nambu_dict = nambu_dict, optimization_parameters = optimization_parameters) 

    return expand_sigma(sigma_r_sol,grid_shape = meshgrid['nambu_matrix_shape']), delta_sol

def compactify_sigma(sigma: jnp.ndarray) -> jnp.ndarray:
    """A function which averages over the angle

    Args:
        sigma (jnp.ndarray): incoming self-energy

    Returns:
        jnp.ndarray: angle averaged self-energy
    """
    # for now just average over the angle, since sigma has no angle dependence
    return jnp.mean(sigma, axis = 3)

def expand_sigma(sigma: jnp.ndarray, grid_shape: tuple) -> jnp.ndarray:
    """A function which expands the given sigma by outer product with identity in angle dependence 

    Args:
        sigma (jnp.ndarray): incoming angle averaged self-energy
        grid_shape (tuple): shape of the meshgrid

    Returns:
        jnp.ndarray: expanded self-energy 
    """
    # for now just average over the angle, since sigma has no angle dependence
    return jnp.tensordot(sigma,jnp.ones(grid_shape[-1]),axes=0)

def flatten_and_split_sigma(sigma_tensor: jnp.ndarray) -> jnp.ndarray:
    """A function which takes sigma tensor in nambu space and returns the flattened sigma tensor with (re_x, im_x, re_y, im_y, re_z, im_z) structure but flattened

    Args:
        sigma_tensor (jnp.ndarray): Incoming tensor in nambu space

    Returns:
        jnp.ndarray: Flattened tensor
    """
    sigma_tensor_x = (sigma_tensor[0,1,...] + sigma_tensor[1,0,...])[None,None,...]
    sigma_tensor_y = 1j*(sigma_tensor[0,1,...] - sigma_tensor[1,0,...])[None,None,...]
    sigma_tensor_z = (sigma_tensor[0,0,...] - sigma_tensor[1,1,...])[None,None,...]
    #TODO: check if we need this transpose here? 
    sigma_out_matrix = jnp.concatenate((jnp.real(sigma_tensor_x),jnp.imag(sigma_tensor_x),jnp.real(sigma_tensor_y),jnp.imag(sigma_tensor_y),jnp.real(sigma_tensor_z),jnp.imag(sigma_tensor_z))).T
    # this exports sigma as (re_x, im_x, re_y, im_y, re_z, im_z)(epsilon,p) sequence
    #print('flattened to be matrix shape is',jnp.shape(sigma_out_matrix))
    return sigma_out_matrix.flatten()

def reconstruct_sigma(vector: jnp.ndarray, original_shape: tuple) -> jnp.ndarray:
    """A function which takes a vector and remaps it to nambu space

    Args:
        vector (jnp.ndarray): Incoming vector
        original_shape (tuple): target shape

    Returns:
        jnp.ndarray: Reconstructed tensor
    """
    vector_size = jnp.size(vector)//6
    real_part_x = vector[0::6]
    imag_part_x = vector[1::6]
    real_part_y = vector[2::6]
    imag_part_y = vector[3::6]
    real_part_z = vector[4::6]
    imag_part_z = vector[5::6]
    pauli_matrices = nambu_support.get_pauli_matrix(None)
    sigma_x = jnp.tensordot(pauli_matrices[1],real_part_x + 1j*imag_part_x,axes=0)
    sigma_y = jnp.tensordot(pauli_matrices[2],real_part_y + 1j*imag_part_y,axes=0)
    sigma_z = jnp.tensordot(pauli_matrices[3],real_part_z + 1j*imag_part_z,axes=0)
    return sigma_x + sigma_y + sigma_z

def flatten_and_split(tensor: jnp.ndarray) -> jnp.ndarray:
    """A function which takes a tensor and flattens it, splitting real and imaginary parts

    Args:
        tensor (jnp.ndarray): Incoming tensor

    Returns:
        jnp.ndarray: Flattened tensor with real and imaginary parts
    """
    return jnp.append(jnp.real(tensor.flatten()), jnp.imag(tensor.flatten()))

def reconstruct_tensor(vector: jnp.ndarray, original_shape: tuple) -> jnp.ndarray:
    """A function which takes a vector and reconstructs a tensor

    Args:
        vector (jnp.ndarray): Incoming vector
        original_shape (tuple): target shape

    Returns:
        jnp.ndarray: Reconstructed tensor
    """
    vector_size = jnp.size(vector)//2
    real_part = jnp.reshape(vector[:vector_size], original_shape)
    imag_part = jnp.reshape(vector[vector_size:], original_shape) 
    return real_part + 1j*imag_part

def optimize_sigma_delta(Q: float,f: jnp.ndarray, sigma_0: jnp.ndarray, delta_0, params_dict: dict, meshgrid: dict,nambu_dict: dict, optimization_parameters = None) -> jnp.ndarray:
    """A function which finds self-energy and the superconducting gap given initial guess

    Args:
        Q (float): External vector potential
        f (jnp.ndarray): Occupation function
        sigma_0 (jnp.ndarray): Initial guess for the self-energy
        delta_0 (_type_): Initial guess for the gap
        params_dict (dict): a dictionary containing all the information about the system parameters
        meshgrid (dict): a dictionary containing all the information about the meshgrid
        nambu_dict (dict): a dictionary containing all the information about the nambu matrices
        optimization_parameters (dict, optional): a dictionary containing all the information about the optimization parameters. Defaults to None.

    Returns:
        jnp.ndarray: Self-consistent self-energy and the gap
    """

    self_energy_coefficients = params_dict['self_energy_coefficients']
    gr = get_gr(Q =Q, sigma_r=expand_sigma(sigma_0,meshgrid['grid_shape']),nambu_dict=nambu_dict,eta=meshgrid['eta'],delta = delta_0, gap_function = params_dict['gap_symmetry'])

    # decide whether self-consistency needs to be done for sigma, depending on which self-energy is provided

    if 'Elastic' in self_energy_coefficients.keys():
        delta_solver_function = lambda sigma, delta : _delta_solver(Q = Q, f = f,sigma_r = sigma,delta = delta, original_sigma_shape = meshgrid['nambu_matrix_shape'][:-1],nambu_dict = nambu_dict,eta = meshgrid['eta'],params_dict = params_dict, meshgrid = meshgrid)
        sigma_solver_function = lambda sigma, delta : _sigma_solver(Q = Q,f = f, sigma_r = sigma,delta = delta, original_sigma_shape = meshgrid['nambu_matrix_shape'][:-1],nambu_dict = nambu_dict,eta = meshgrid['eta'],params_dict = params_dict, meshgrid = meshgrid)
    
    elif 'Dynes' in self_energy_coefficients.keys():
        sigma_0 = compactify_sigma(_sigma_r_dynes(nambu_dict=nambu_dict,dynes_eta=self_energy_coefficients['Dynes']))
        delta_solver_function = lambda sigma, delta : _delta_solver(Q = Q, f = f,sigma_r = sigma,delta = delta, original_sigma_shape = meshgrid['nambu_matrix_shape'][:-1],nambu_dict = nambu_dict,eta = meshgrid['eta'],params_dict = params_dict, meshgrid = meshgrid)
        sigma_solver_function = lambda sigma, delta: jnp.zeros_like(sigma)
    else:
        sigma_0 = (jnp.zeros_like(sigma_0))
        delta_solver_function = lambda sigma, delta : _delta_solver(Q = Q, f = f,sigma_r = sigma,delta = delta, original_sigma_shape = meshgrid['nambu_matrix_shape'][:-1],nambu_dict = nambu_dict,eta = meshgrid['eta'],params_dict = params_dict, meshgrid = meshgrid)
        sigma_solver_function = lambda sigma, delta: jnp.zeros_like(sigma)
    
    # flattened initial guesses
    sigma_0 = flatten_and_split_sigma(sigma_0)
    delta_0 = flatten_and_split(delta_0)

    
    # define the jacobian function given the external parameters and the meshgrid
    sigma_solver_jacobian = lambda sigma, delta: _sigma_solver_jacobian(Q = Q,f = f, sigma_r = sigma,delta = delta, original_sigma_shape = meshgrid['nambu_matrix_shape'][:-1],nambu_dict = nambu_dict,eta = meshgrid['eta'],params_dict = params_dict, meshgrid = meshgrid)
    #sigma_final, delta_final = custom_optimizer._iterative_jax_solver(sigma_solver_function, delta_solver_function,sigma_0, delta_0, optimization_parameters = optimization_parameters)
    # solve the system self-consistently
    sigma_final, delta_final = custom_optimizer._iterative_jax_solver_with_jacobian(sigma_solver_function, delta_solver_function,sigma_0, delta_0, sigma_solver_jacobian, optimization_parameters = optimization_parameters)
    
    # if there is no elastic scattering then sigma_final will just be the computed sigma
    if 'Dynes' in self_energy_coefficients.keys() and not('Elastic' in self_energy_coefficients.keys()):
        sigma_final = sigma_0

    return reconstruct_sigma(sigma_final,meshgrid['nambu_matrix_shape'][:-1]), delta_final[0] + 1.j*delta_final[1]

def _delta_solver(Q:float, f:jnp.ndarray,sigma_r:jnp.ndarray,delta:jnp.ndarray, original_sigma_shape:tuple, nambu_dict:dict, eta:float, params_dict:dict, meshgrid:dict) -> jnp.ndarray:
    """A function which defined the self-consistent gap equation 

    Args:
        Q (float): External vector potential
        f (jnp.ndarray): Occupation function
        sigma_r (jnp.ndarray): self-energy
        delta (jnp.ndarray): superconducting gap
        original_sigma_shape (tuple): shape of the nambu matrices
        nambu_dict (dict): a dictionary containing all the information about the nambu matrices
        eta (float): broadening introduced to the retarded Green's function
        params_dict (dict): dictionary containing all the system parameters
        meshgrid (dict): dictionary containing all the meshgrid parameters

    Returns:
        jnp.ndarray: self-consistent gap
    """
    sigma_r_native = reconstruct_sigma(vector = sigma_r, original_shape=original_sigma_shape)
    delta_native = delta[0] +  1.j*delta[1]
    gr = get_gr(Q =Q, sigma_r=expand_sigma(sigma_r_native,meshgrid['grid_shape']),nambu_dict=nambu_dict,eta=eta,delta = delta_native, gap_function = params_dict['gap_symmetry'])
    delta_self_cons_result = delta_native - _calc_gap(gr = gr, f = f, nambu_dict= nambu_dict,gap_function = params_dict['gap_symmetry'],meshgrid = meshgrid,critical_temperature = params_dict['critical_temperature'])

    return  flatten_and_split(delta_self_cons_result)

def _sigma_solver(Q:float,f:jnp.ndarray, sigma_r:jnp.ndarray,delta:jnp.ndarray, original_sigma_shape:tuple, nambu_dict:dict, eta:float, params_dict:dict, meshgrid:dict,omega_grid = None):
    """A function which defined the self-consistent sigma equation 

    Args:
        Q (float): External vector potential
        f (jnp.ndarray): Occupation function
        sigma_r (jnp.ndarray): self-energy
        delta (jnp.ndarray): superconducting gap
        original_sigma_shape (tuple): shape of the nambu matrices
        nambu_dict (dict): a dictionary containing all the information about the nambu matrices
        eta (float): broadening introduced to the retarded Green's function
        params_dict (dict): dictionary containing all the system parameters
        meshgrid (dict): dictionary containing all the meshgrid parameters
        omega_grid(jnp.ndarray): grid of frequencies for which the sigma_r is computed, used in the jacobian
    Returns:
        jnp.ndarray: self-consistent gap
    """
    # if a subgrid is provided evaluate sigma on the subgrid not on the full grid
    if not(omega_grid is None):
        meshgrid['omega_grid'] = omega_grid
        nambu_dict['nambu_omega_grid'] = omega_grid

    # reshaping inputs
    sigma_r_native = reconstruct_sigma(vector = sigma_r, original_shape=original_sigma_shape)
    delta_native = delta[0] +  1.j*delta[1]

    gr = get_gr(Q =Q, sigma_r=expand_sigma(sigma_r_native,meshgrid['grid_shape']),nambu_dict=nambu_dict,eta=eta,delta = delta_native, gap_function = params_dict['gap_symmetry'])
    sigma_self_cons_result = sigma_r_native - compactify_sigma(_sigma_r(gr = gr,nambu_dict=nambu_dict,self_energy_coefficients= params_dict['self_energy_coefficients']))
    return  flatten_and_split_sigma(sigma_self_cons_result)

def _sigma_solver_jacobian(Q:float,f:jnp.ndarray, sigma_r:jnp.ndarray,delta:jnp.ndarray, original_sigma_shape:tuple, nambu_dict:dict, eta:float, params_dict:dict, meshgrid:dict,omega_grid = None) -> jnp.ndarray:
    """A function which computes the sigma_solver jacobian by splitting sigma's into individual epsilon dependencies 

    Args:
        Q (float): External vector potential
        f (jnp.ndarray): Occupation function
        sigma_r (jnp.ndarray): self-energy
        delta (jnp.ndarray): superconducting gap
        original_sigma_shape (tuple): shape of the nambu matrices
        nambu_dict (dict): a dictionary containing all the information about the nambu matrices
        eta (float): broadening introduced to the retarded Green's function
        params_dict (dict): dictionary containing all the system parameters
        meshgrid (dict): dictionary containing all the meshgrid parameters
        omega_grid (jnp.ndarray, optional): a subgrid of frequencies for which the sigma_r is computed, used in the jacobian. Defaults to None.

    Returns:
        jnp.ndarray: list of jacobians for each epsilon
    """
    # here sigma is in format (6) * epsilon_size where 6 because of sigma xyz * real/imag 
    #sigma_r_native = reconstruct_sigma(vector = sigma_r, original_shape=original_sigma_shape)
    delta_native = delta[0] +  1.j*delta[1]

    pauli_omega_grid = nambu_dict['nambu_omega_grid'] # third index tells the omega grid

    omega_grid_list = []

    #TODO: This for loop can be vectorized
    for n in range(jnp.shape(pauli_omega_grid)[2]):
        omega_grid_list += [pauli_omega_grid[:,:,n:n+1,:]]
    

    # magic thingies to make sure we evaluate sigma_r and g_r only on specific subgrids 
    local_nambu_dict = nambu_dict.copy()
    local_nambu_dict['nambu_theta_grid'] = local_nambu_dict['nambu_theta_grid'][:,:,0:1,:]
    local_nambu_dict['nambu_matrices'] = local_nambu_dict['nambu_matrices'][:,:,:,0:1,:]
    local_nambu_dict['nambu_matrix_shape'] = (2,2,1,20)
    omega_grid_list = jnp.array(omega_grid_list)

    #dictionary_list = jnp.array(dictionary_list)
    local_params_dict = params_dict.copy()
    local_params_dict['gap_symmetry'] = local_params_dict['gap_symmetry'][:,:,0:1,:]
    # solver for the specific epsilon
    subdivided_solver = lambda sigma_n, n, delta : _sigma_solver(Q = Q,f = f[:,:,n,:], sigma_r = sigma_n,delta = delta, original_sigma_shape = original_sigma_shape[:-1],omega_grid = omega_grid_list[n],nambu_dict = local_nambu_dict,eta = eta,params_dict = local_params_dict, meshgrid = meshgrid) 
    # jacobian for the specific epsilon computed automatically using jax.jacobian i.e. derivatives
    subdivided_jacobian = lambda sigma_n, n, delta : jax.jacobian(subdivided_solver, argnums = 0)(sigma_n,n,delta)
    # vectorize the derivative term 
    vectorized_derivative = jax.vmap(subdivided_jacobian, in_axes = (0,0,None))

    # generate the sigma collection (6,epsilon_size)
    sigma_collection = sigma_r.reshape(jnp.size(sigma_r)//6,6)
    # number indicies to be passed, which determine which epsilon value is read from the meshgrid
    numbers = jnp.array((range(0,jnp.size(sigma_r)//6)),dtype=jnp.int32)

    # compute the jacobian list
    jacobian_out = vectorized_derivative(sigma_collection, numbers,delta)

    return jacobian_out