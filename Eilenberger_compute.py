import jax
import jax.numpy as jnp

import nambu_support
import custom_optimizer 
import numpy as np
import parameter_generator
def _r2a(gr,nambu_dict):
    ### This method conjugates a retarded object to get an advanced one 
    #* Let's see how this works with small eta's etc. 
    ga = -jnp.transpose(jnp.conjugate(gr),axes=(1,0,2,3))
    
    ga = nambu_support.nambu_mul(nambu_dict['nambu_matrices'][3],ga)
    ga = nambu_support.nambu_mul(ga,nambu_dict['nambu_matrices'][3])
    
    return ga 

def _f2gk(gr,f,nambu_dict):
    ### This method takes gr and f object and computes the proper Keldysh correlation funciton
    #* Only correct if we ignore the other corrections which come from the convolution
    gk = nambu_support.nambu_mul(gr,f) - nambu_support.nambu_mul(f,_r2a(gr=gr,nambu_dict=nambu_dict)) 
    
    return gk 

def _hr2gr(hr):
    ### Inverts and normalizes a retarded effective Hamiltonian
    return -1.j* hr/jnp.sqrt(nambu_support.nambu_det(hr)) 
    #return np.sign(self.w)*hr/np.sqrt(-self._Nambu_det(hr))


def _Doppler_w_r(Q,nambu_dict,eta):
    ### returns the Doppler shifted frequency Nambu tensor with retarded causality 	
    #return ( self.w - Q*np.cos(self.theta) + 0.5j*self.eta*np.ones_like(self.w) )*self.Nambu_matrices[3]
    return (nambu_dict['nambu_omega_grid'] - Q*jnp.cos(nambu_dict['nambu_theta_grid']) + 1.j*1e-8*jnp.ones_like(nambu_dict['nambu_omega_grid']) + 1.j*eta * jnp.ones_like(nambu_dict['nambu_omega_grid']))*nambu_dict['nambu_matrices'][3] 

		
def _Delta_p(gap,nambu_dict,gap_function):
    ### Returns the momentum resolved Nambu tensor gap  		
    ### Allows for a complex gap 
    return 1.j*jnp.real(gap) * nambu_dict['nambu_matrices'][2]*gap_function +1.j* jnp.imag(gap)*nambu_dict['nambu_matrices'][1]*gap_function

def _sigma_r(gr,nambu_dict,self_energy_coefficients):
    out = jnp.zeros(nambu_dict['nambu_matrix_shape'])
    if 'Dynes' in self_energy_coefficients:
        out += _sigma_r_dynes(nambu_dict=nambu_dict,dynes_eta=self_energy_coefficients['Dynes'])
    if 'Elastic' in self_energy_coefficients:
        out += _sigma_r_elastic(gr=gr,nambu_dict=nambu_dict,impurity_scattering=self_energy_coefficients['Elastic'])
    return out
    
def _sigma_r_elastic(gr,nambu_dict,impurity_scattering): 
    ### This method computes the retarded self energy from gr alone

    ### Impurity scattering contributions 
    sigma = -0.5j*impurity_scattering*jnp.mean(gr,axis=3,keepdims=True)
    return sigma 

def _sigma_r_dynes(nambu_dict,dynes_eta):
    sigma = 0.5j * dynes_eta * nambu_dict['nambu_matrices'][3]
    return sigma

def get_gr(Q,sigma_r,delta, nambu_dict, eta, gap_function):
    h_r_bare = _Doppler_w_r(Q=Q,nambu_dict=nambu_dict,eta=eta)
    # h_delta from gap 
    h_r_delta = lambda gap : _Delta_p(gap = gap, nambu_dict=nambu_dict,gap_function=gap_function)
    return  _hr2gr(hr = h_r_bare - sigma_r - h_r_delta(delta))

def _calc_gr(f,Q, params_dict, meshgrid, nambu_dict,eta,gap_function, self_energy_dict, gr0 = None, optimization_parameters = None):
    # define the bare Hamiltonian: epsling + Doppler
    
    sigma_r, delta = _self_consistent_delta_sigma(f = f, Q = Q, params_dict = params_dict, meshgrid=meshgrid, nambu_dict = nambu_dict, gr0 = gr0, optimization_parameters = optimization_parameters)

    return get_gr(Q = Q, sigma_r = sigma_r,delta = delta, nambu_dict = nambu_dict, eta = eta, gap_function = gap_function)

def _integrate(f, meshgrid):
    ### This method will integrate scalar function f over the frequency and angle grids (normalized by 2pi) assuming a possibly adaptive grid 
    ### For the moment we assume that f is a scalar and therefore already has had the Nambu indices traced out 
    
    ### We will simply sum this over all indices to return a single number 
    return custom_optimizer.custom_jax_trapz(jnp.mean(f,axis = 1),meshgrid['omega_array'])


def _calc_gap(gr,f,nambu_dict,gap_function,meshgrid,critical_temperature = 1):
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

def _self_consistent_delta_sigma(f,Q,params_dict, meshgrid,nambu_dict,gr0=None, optimization_parameters = None):
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

def compactify_sigma(sigma):
    # for now just average over the angle, since sigma has no angle dependence
    return jnp.mean(sigma, axis = 3)

def expand_sigma(sigma, grid_shape):
    # for now just average over the angle, since sigma has no angle dependence
    return jnp.tensordot(sigma,jnp.ones(grid_shape[-1]),axes=0)

def flatten_and_split_sigma(sigma_tensor):
    sigma_tensor_x = (sigma_tensor[0,1,...] + sigma_tensor[1,0,...])[None,None,...]
    sigma_tensor_y = 1j*(sigma_tensor[0,1,...] - sigma_tensor[1,0,...])[None,None,...]
    sigma_tensor_z = (sigma_tensor[0,0,...] - sigma_tensor[1,1,...])[None,None,...]
    #TODO: check if we need this transpose here? 
    sigma_out_matrix = jnp.concatenate((jnp.real(sigma_tensor_x),jnp.imag(sigma_tensor_x),jnp.real(sigma_tensor_y),jnp.imag(sigma_tensor_y),jnp.real(sigma_tensor_z),jnp.imag(sigma_tensor_z))).T
    # this exports sigma as (re_x, im_x, re_y, im_y, re_z, im_z)(epsilon,p) sequence
    #print('flattened to be matrix shape is',jnp.shape(sigma_out_matrix))
    return sigma_out_matrix.flatten()

def reconstruct_sigma(vector, original_shape):
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

def flatten_and_split(tensor):
    return jnp.append(jnp.real(tensor.flatten()), jnp.imag(tensor.flatten()))

def reconstruct_tensor(vector, original_shape):
    vector_size = jnp.size(vector)//2
    real_part = jnp.reshape(vector[:vector_size], original_shape)
    imag_part = jnp.reshape(vector[vector_size:], original_shape) 
    return real_part + 1j*imag_part

def optimize_sigma_delta(Q,f, sigma_0, delta_0, params_dict, meshgrid,nambu_dict, optimization_parameters = None):

    self_energy_coefficients = params_dict['self_energy_coefficients']
    gr = get_gr(Q =Q, sigma_r=expand_sigma(sigma_0,meshgrid['grid_shape']),nambu_dict=nambu_dict,eta=meshgrid['eta'],delta = delta_0, gap_function = params_dict['gap_symmetry'])

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
    
    sigma_0 = flatten_and_split_sigma(sigma_0)
    delta_0 = flatten_and_split(delta_0)

    #jac_test = _sigma_solver_jacobian(Q = Q,f = f, sigma_r = sigma_0,delta = delta_0, original_sigma_shape = meshgrid['nambu_matrix_shape'][:-1],nambu_dict = nambu_dict,eta = meshgrid['eta'],params_dict = params_dict, meshgrid = meshgrid)
    
    sigma_solver_jacobian = lambda sigma, delta: _sigma_solver_jacobian(Q = Q,f = f, sigma_r = sigma,delta = delta, original_sigma_shape = meshgrid['nambu_matrix_shape'][:-1],nambu_dict = nambu_dict,eta = meshgrid['eta'],params_dict = params_dict, meshgrid = meshgrid)
    #sigma_final, delta_final = custom_optimizer._iterative_jax_solver(sigma_solver_function, delta_solver_function,sigma_0, delta_0, optimization_parameters = optimization_parameters)
    sigma_final, delta_final = custom_optimizer._iterative_jax_solver_with_jacobian(sigma_solver_function, delta_solver_function,sigma_0, delta_0, sigma_solver_jacobian, optimization_parameters = optimization_parameters)
    
    
    if 'Dynes' in self_energy_coefficients.keys() and not('Elastic' in self_energy_coefficients.keys()):
        sigma_final = sigma_0

    return reconstruct_sigma(sigma_final,meshgrid['nambu_matrix_shape'][:-1]), delta_final[0] + 1.j*delta_final[1]

def _delta_solver(Q, f,sigma_r,delta, original_sigma_shape, nambu_dict, eta, params_dict, meshgrid):
    sigma_r_native = reconstruct_sigma(vector = sigma_r, original_shape=original_sigma_shape)
    delta_native = delta[0] +  1.j*delta[1]
    gr = get_gr(Q =Q, sigma_r=expand_sigma(sigma_r_native,meshgrid['grid_shape']),nambu_dict=nambu_dict,eta=eta,delta = delta_native, gap_function = params_dict['gap_symmetry'])
    delta_self_cons_result = delta_native - _calc_gap(gr = gr, f = f, nambu_dict= nambu_dict,gap_function = params_dict['gap_symmetry'],meshgrid = meshgrid,critical_temperature = params_dict['critical_temperature'])

    return  flatten_and_split(delta_self_cons_result)

def _sigma_solver(Q,f, sigma_r,delta, original_sigma_shape, nambu_dict, eta, params_dict, meshgrid,omega_grid = None):
    if not(omega_grid is None):
        meshgrid['omega_grid'] = omega_grid
        nambu_dict['nambu_omega_grid'] = omega_grid

    sigma_r_native = reconstruct_sigma(vector = sigma_r, original_shape=original_sigma_shape)
    delta_native = delta[0] +  1.j*delta[1]

    gr = get_gr(Q =Q, sigma_r=expand_sigma(sigma_r_native,meshgrid['grid_shape']),nambu_dict=nambu_dict,eta=eta,delta = delta_native, gap_function = params_dict['gap_symmetry'])
    sigma_self_cons_result = sigma_r_native - compactify_sigma(_sigma_r(gr = gr,nambu_dict=nambu_dict,self_energy_coefficients= params_dict['self_energy_coefficients']))
    return  flatten_and_split_sigma(sigma_self_cons_result)

# will return the Jacobian as a subdivided list! 
def _sigma_solver_jacobian(Q,f, sigma_r,delta, original_sigma_shape, nambu_dict, eta, params_dict, meshgrid):
    # here sigma is in format (6) * epsilon_size where 6 because of sigma xyz * real/imag 
    #sigma_r_native = reconstruct_sigma(vector = sigma_r, original_shape=original_sigma_shape)
    delta_native = delta[0] +  1.j*delta[1]

    pauli_omega_grid = nambu_dict['nambu_omega_grid'] # third index tells the omega grid

    omega_grid_list = []

    for n in range(jnp.shape(pauli_omega_grid)[2]):
        omega_grid_list += [pauli_omega_grid[:,:,n:n+1,:]]
    
    local_nambu_dict = nambu_dict.copy()
    local_nambu_dict['nambu_theta_grid'] = local_nambu_dict['nambu_theta_grid'][:,:,0:1,:]
    local_nambu_dict['nambu_matrices'] = local_nambu_dict['nambu_matrices'][:,:,:,0:1,:]
    local_nambu_dict['nambu_matrix_shape'] = (2,2,1,20)
    omega_grid_list = jnp.array(omega_grid_list)

    #dictionary_list = jnp.array(dictionary_list)
    local_params_dict = params_dict.copy()
    local_params_dict['gap_symmetry'] = local_params_dict['gap_symmetry'][:,:,0:1,:]
    subdivided_solver = lambda sigma_n, n, delta : _sigma_solver(Q = Q,f = f[:,:,n,:], sigma_r = sigma_n,delta = delta, original_sigma_shape = original_sigma_shape[:-1],omega_grid = omega_grid_list[n],nambu_dict = local_nambu_dict,eta = eta,params_dict = local_params_dict, meshgrid = meshgrid) 
    subdivided_jacobian = lambda sigma_n, n, delta : jax.jacobian(subdivided_solver, argnums = 0)(sigma_n,n,delta)

    vectorized_derivative = jax.vmap(subdivided_jacobian, in_axes = (0,0,None))

    sigma_collection = sigma_r.reshape(jnp.size(sigma_r)//6,6)
    numbers = jnp.array((range(0,jnp.size(sigma_r)//6)),dtype=jnp.int32)

    jacobian_out = vectorized_derivative(sigma_collection, numbers,delta)

    return jacobian_out