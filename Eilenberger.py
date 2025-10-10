### Jonathan Curtis 
### 08/11/2025
### Code to simulate dynamics of Eilenberger equation

import numpy as np
from scipy import integrate as intg
from scipy import optimize as opt
import time

# optimization goes brrr 
import jax
import jax.numpy as jnp
from functools import partial 
# import plotting for debugging, should be removed later
import matplotlib.pyplot as plt
from tqdm import tqdm

import nambu_support
# importing custom made debugger which will ensure that at certain points the code runs as expected

zero = 1.e-8 ### small number for causality 

""" 
BCS_gap_constant = 2.*jnp.exp(np.euler_gamma)/jnp.pi ### 2e^gamma/pi constant often appearing in BCS integrals 
BCS_ratio = 2./BCS_gap_constant #1.765387449618725 ### Ratio of Delta(0)/Tc in BCS limit 


### Various Pauli matrices 
Pauli = [jnp.eye(2,dtype=complex), jnp.array([[0.j,1.],[1.,0.j]]), jnp.array([[0.j,-1.j],[1.j,0.j]]), jnp.array([[1.0,0.j],[0.j,-1.]]) ]
Paulimin = 0.5*(Pauli[1] -1.j*Pauli[2])
Pauliplus = 0.5*(Pauli[1] +1.j*Pauli[2])
""" 

"""
### Methods for packing and unpacking complex to real tensors for scipy methods
def _pack(z: jnp.ndarray) -> jnp.ndarray:
	z = jnp.asarray(z, jnp.complex128)
	return jnp.concatenate([z.real.ravel(), z.imag.ravel()])

def _unpack(y: jnp.ndarray, shape) -> jnp.ndarray:
    n = int(jnp.prod(shape))
    re = y[:n].reshape(shape)
    im = y[n:].reshape(shape)
    return re + 1j*im
""" 

import jax.numpy as jnp
#!
def _jax_trapz(y, x):
    dx = jnp.diff(x)
    avg_y = 0.5 * (y[:-1] + y[1:])
    return jnp.sum(dx * avg_y)

class Eilenberger:
	def __init__(self, nw, ntheta, cutoff,fine_grid=(None,None)):
		self.verbose = False ### If this is true we will have more information and feedback given during calculations
	
		# defining the size of the frequency grid and angle grid and setting the cut-off energy integration cut-off
		self.nw = nw if nw % 2 == 0 else nw + 1  # Ensure even number
		self.ntheta = ntheta
		self.cutoff = cutoff
		
		self.Tc = 1. ### By default we use units where Tc is one 	
		
		### Frequency and angular grids -- we will later implement adaptively sampled frequency grid to reduce need for number of points to get good resolution 
		self.w_arr = jnp.linspace(-self.cutoff,self.cutoff,self.nw)
		self.theta_arr = jnp.linspace(0.,2.*np.pi,self.ntheta,endpoint=False)	
		
		### Internal default eta for broadening of spectral functions 
		self.eta = 1.*(self.w_arr[1]-self.w_arr[0]) ### This will be the small broadening for just the large grid ~= frequency step size 
		
		### We allow for an optional specification of an additional finer grid region 
		### This is done by passing a tuple fine_grid = (fine_nw, fine_cutoff) 
		### We then generate a finer grid of fine_nw points up to fine_cutoff before switching to a coarser grid
		self.fine_nw, self.fine_cutoff = fine_grid

		if self.fine_nw is None:
			self.fine_grid = None 
			
		else:
			if self.fine_nw %2 == 0: self.fine_nw += 1
			self.fine_grid = jnp.linspace(-self.fine_cutoff,self.fine_cutoff,self.fine_nw)
			self.eta = 2.*(self.fine_grid[1]-self.fine_grid[0])
	
			w_arr = jnp.concatenate([self.w_arr, self.fine_grid]) ### this joins the two arrays 
			self.w_arr = jnp.unique(w_arr) ### sorts and removes duplicates 
		
		self.w_grid, self.theta_grid = jnp.meshgrid( self.w_arr , self.theta_arr ,indexing = 'ij') 
		
		self.grid_shape = self.w_grid.shape  # (Nw, Ntheta)
		
		### Internal default parameters for SCBA solver 
		### Taken from ChatGPT implementation of Anderson accelerated solver 
		### Also used in the clunky hand-written Picard solver which seems to anyways work better
		self.scba_hist = 5 ### For Anderson root finding algorithm this is the history of number of previous guesses we use 
		self.scba_step = 0.05 ### Update gradient step 
		self.scba_err = 1.e-3 ### relative error threshold for SCBA convergence 
		self.scba_max_steps = 4000 ### Total number of iterations before we throw an error 
	
		### Generate the necessary Nambu-shaped tensors 
		### Nambu tensor class is not yet working 
		pauli = nambu_support.get_pauli_matrix(None)
		self.Nambu_matrices = [ jnp.tensordot(sigma, jnp.ones_like(self.w_grid),axes=0 ) for sigma in pauli ] ### These are now Pauli matrices and identity function on the momenta/frequency 
		self.Nambu_shape = self.Nambu_matrices[0].shape ### Should be (2,2,nw,ntheta) 
		
		self.w = nambu_support.nambu_scalar2nambu(self.w_grid)
		self.theta = nambu_support.nambu_scalar2nambu(self.theta_grid) 
		
		print(jnp.shape(self.theta))
		self.gap_function = jnp.ones_like(self.theta) 
	
		### We now set up the shapes for the Keldysh degrees of freedom
		#? Removed this in the new iteration of the code
		self.Keldysh_shape = (2,*self.Nambu_shape) ### We now double for the Keldysh dof (we store only [R,K] )
		
		### Default function call for supercurrent will just return zero 
		self.Q_t = lambda x: 0. 
		self.Q0 =0. ### A static background which is set to zero by default 
	
	########################
	### INTERNAL METHODS ### 
	########################
	#!
	def _integrate(self,f):
		### This method will integrate scalar function f over the frequency and angle grids (normalized by 2pi) assuming a possibly adaptive grid 
		### For the moment we assume that f is a scalar and therefore already has had the Nambu indices traced out 
		
		### We will simply sum this over all indices to return a single number 
		return _jax_trapz(jnp.mean(f,axis = 1),self.w_arr)
	"""
	def _NambuMul(self,x,y):
		### For the time being we will use a homebuilt overload for matrix multiplication for Nambu tensors until the tensor class can be tested more 
		return jnp.einsum('ijnm,jknm->iknm', x,y)
		
	def _scalar2Nambu(self,x):
		### Promotes a scalar tensor function to a Nambu compatible tensor 
		return jnp.tensordot(jnp.ones((2,2),dtype=complex),x,axes=0) 
	""" 
	#TODO: remove in new version
	def _rf2g(self,gr,f):
		### Promotes a pair (gr,f) to a single Keldysh object
		return jnp.stack([gr,f]) 
	#! 
	def _r2a(self,gr):
		### This method conjugates a retarded object to get an advanced one 
		#* Let's see how this works with small eta's etc. 
		ga = -jnp.transpose(jnp.conjugate(gr),axes=(1,0,2,3))
		
		ga = nambu_support.nambu_mul(self.Nambu_matrices[3],ga)
		ga = nambu_support.nambu_mul(ga,self.Nambu_matrices[3])
		
		return ga 
	#!
	def _f2gk(self,g):
		### This method takes a gf = [gr, f] object and computes the proper Keldysh correlation funciton
		gr = g[0,...]
		f = g[1,...]
		#* Only correct if we ignore the other corrections which come from the convolution
		gk = nambu_support.nambu_mul(gr,f) - nambu_support.nambu_mul(f,self._r2a(gr)) 
		
		return gk 
	
	"""
	def _Nambu_det(self,a):
		### Computes the determinant of a Nambu matrix as a tensor over the grid of frequency and angle 			
		det = a[0,0,...] * a[1,1,...] - a[0,1,...]*a[1,0,...] ### Has shape of the frequency and mesh grid
		#* Note sure how this part works, but ok 
		out = det[None,None,...]

		return out 
	""" 
	#!
	def _hr2gr(self,hr):
		### Inverts and normalizes a retarded effective Hamiltonian
		#* This should be checked the roots are taken properly. Since numpy.sqrt() does not know which root you want to pick
		#* Integration close to square root divergence needs to be properly regularized! 
		return -1.j* hr/jnp.sqrt(nambu_support.nambu_det(hr)) 
		#return np.sign(self.w)*hr/np.sqrt(-self._Nambu_det(hr))
	#!
	def _Doppler_w_r(self,Q):
		### returns the Doppler shifted frequency Nambu tensor with retarded causality 	
		### The value of eta is used in the self energy, here we only put a very small eta to choose retarded causality	
		#return ( self.w - Q*np.cos(self.theta) + 0.5j*self.eta*np.ones_like(self.w) )*self.Nambu_matrices[3] 
		#? This zero here has to be larger than the energy spacing? Not 1e-8?
		return ( self.w - Q*np.cos(self.theta) + 1.j*zero*jnp.ones_like(self.w) + 1.j*self.eta * jnp.ones_like(self.w))*self.Nambu_matrices[3] 
	#!
	def _Delta_p(self,gap):
		### Returns the momentum resolved Nambu tensor gap  		
		### Allows for a complex gap 
		return 1.j*jnp.real(gap) * self.Nambu_matrices[2]*self.gap_function +1.j* jnp.imag(gap)*self.Nambu_matrices[1]*self.gap_function
	#!
	def _sigma_r(self,gr): 
		### This method computes the retarded self energy from gr alone
		sigma = jnp.zeros_like(gr) 
		
		### Impurity scattering contributions 
		sigma += -0.5j*self.gamma_imp*jnp.mean(gr,axis=3,keepdims=True)
		
		### Dynes inelastic scattering 
		#? Should this be here our in g somewhere? 
		sigma += -0.5j*self.eta*self.Nambu_matrices[3]
		
		return sigma 
		
	# jaxified version of the self-consistency method
	# define a newton method jax solver

	def _newton_nd(self,f, x0, tol=1e-3, maxiter=20):
		"""
		Multidimensional Newton's method for vector input x ∈ R^n.
		Uses JAX Jacobian automatically.
		"""
		def cond_fun(state):
			x, i = state
			return jnp.logical_and(jnp.linalg.norm(f(x)) > tol, i < maxiter)

		def body_fun(state):
			x, i = state
			J = jax.jacobian(f)(x)          # Jacobian (n×n)
			dx = jnp.linalg.solve(J, f(x))  # Newton step
			return (x - dx, i + 1)

		x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))
		return x_final


	def _iterative_jax_solver(self,f1, f2, x0, y0, tol=1e-3, maxiter=200):
		"""
		Alternating self-consistent root finder for vector x, y.

		f1: function f1(x, y) → R^n
		f2: function f2(x, y) → R^m
		"""
		def cond_fun(state):
			x, y, i = state
			err = jnp.maximum(jnp.linalg.norm(f1(x, y)), jnp.linalg.norm(f2(x, y)))
			return jnp.logical_and(err > tol, i < maxiter)

		def body_fun(state):
			x, y, i = state

			# Solve f1(x, y) = 0 for x given y
			#x_new = jnp.zeros_like(x)
			x_new = self._newton_nd(lambda x_: f1(x_, y), x)
			# Solve f2(x, y) = 0 for y given x_new
			y_new = self._newton_nd(lambda y_: f2(x_new, y_), y)

			return (x_new, y_new, i + 1)

		x_final, y_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, y0, 0))
		return x_final, y_final
	#!
	def _delta_solver(self,sigma,delta):
		sigma_size = 2 * 2 * (self.nw + self.fine_nw)

		Q = self.Q0
		f = self.fd_tensor
		h_r_bare = jnp.array(self._Doppler_w_r(Q))

		# h_delta from gap 
		h_r_delta = lambda gap : self._Delta_p(gap)

		# g as a function of sigma_r and delta
		gr_function = lambda sigma_r, delta : self._hr2gr(h_r_bare - sigma_r - h_r_delta(delta))

		def _delta_flat_complex(crt_sigma_real, crt_sigma_imag , crt_gap_real, crt_gap_imag, sigma_shape):		
			# sigma_r and delta self-consistencies as a function of sigma_r and delta
			delta_self_cons = lambda crt_sigma, crt_gap : crt_gap - self._calc_gap(self._rf2g(gr_function(jnp.tensordot(crt_sigma,jnp.ones(self.ntheta),axes=0),crt_gap),f)) 
			sigma_r = np.reshape(crt_sigma_real + 1.j*crt_sigma_imag,sigma_shape)
			delta = crt_gap_real + 1.j*crt_gap_imag
			return jnp.append(jnp.real(delta_self_cons(sigma_r,delta).flatten()),jnp.imag(delta_self_cons(sigma_r,delta).flatten())) 
		return  _delta_flat_complex(sigma[:sigma_size],sigma[sigma_size:],delta[0],delta[1],(2,2,(self.nw + self.fine_nw)))
	#!
	def _sigma_solver(self,sigma,delta):
		sigma_size = 2 * 2 * (self.nw + self.fine_nw)

		Q = self.Q0
		f = self.fd_tensor

		def _sigma_flat_complex(crt_sigma_real, crt_sigma_imag , crt_gap_real, crt_gap_imag, sigma_shape):
			h_r_bare = jnp.array(self._Doppler_w_r(Q))

			# h_delta from gap 
			h_r_delta = lambda gap : self._Delta_p(gap)

			# g as a function of sigma_r and delta
			gr_function = lambda sigma_r, delta : self._hr2gr(h_r_bare - sigma_r - h_r_delta(delta))
			
			# sigma_r and delta self-consistencies as a function of sigma_r and delta
			# note, here sigma_r is the average value so we remove the p-hat direction redundancy which just makes runs longer
			sigma_r_self_cons = lambda crt_sigma, crt_gap : crt_sigma - jnp.mean(self._sigma_r(gr_function(jnp.tensordot(crt_sigma,jnp.ones(self.ntheta),axes=0),crt_gap)),axis=3)
			sigma_r = jnp.reshape(crt_sigma_real + 1.j*crt_sigma_imag,sigma_shape)
			delta = crt_gap_real + 1.j*crt_gap_imag
			return jnp.append(jnp.real(sigma_r_self_cons(sigma_r,delta).flatten()),jnp.imag(sigma_r_self_cons(sigma_r,delta).flatten()))

		return  _sigma_flat_complex(sigma[:sigma_size],sigma[sigma_size:],delta[0],delta[1],(2,2,(self.nw + self.fine_nw)))

	def _self_cons_delta_sigma(self,f,Q,gr0=None, root_method = 'anderson'):
		# define the bare Hamiltonian: epsling + Doppler
		h_r_bare = jnp.array(self._Doppler_w_r(Q))

		# h_delta from gap 
		h_r_delta = lambda gap : self._Delta_p(gap)

		# g as a function of sigma_r and delta
		gr_function = lambda sigma_r, delta : self._hr2gr(h_r_bare - sigma_r - h_r_delta(delta))

		# save the original shape of sigma_r
		original_sigma_r_shape = jnp.shape(h_r_bare)
		sigma_size = np.prod(original_sigma_r_shape[:-1])
		# Method which optimizes sigma^r
		
		# initial guess for the sigma and gap
		if gr0 is None:
			gap_0 = 1.2 * nambu_support.get_bcs_ratio()*self.Tc 
			sigma_r_0 = jnp.mean(self._sigma_r(self._hr2gr(h_r_bare)),axis=3)
		else:
			gap_0 = self._calc_gap(self._rf2g(gr0,f))
			sigma_r_0 = jnp.mean(self._sigma_r(gr0),axis=3)

		sigma_delta_0 = jnp.append(jnp.append(np.real(sigma_r_0.flatten()),jnp.imag(sigma_r_0.flatten())),jnp.append(jnp.real(gap_0),jnp.imag(gap_0)))
	
		crt_sigma = sigma_delta_0[:-2]
		crt_delta = sigma_delta_0[-2:]
		crt_sigma = jnp.zeros(sigma_size * 2)

		tol = 1e-3 
		crt_sigma, crt_delta = total_solver(self,crt_sigma,crt_delta) 

		# initial guess for the solver
		#sigma_delta_0 = np.append(np.real(gap_0),np.imag(gap_0)) #* Guess for sigma = 0 case
		
		# solve and save the solution
		#sigma_delta = opt.root(sigma_delta_solver,sigma_delta_0,method=root_method) 
		
		# convert the solution into canonical form
		#sigma_flat = sigma_delta.x[:sigma_size ] + 1j*sigma_delta.x[sigma_size:-2]
		sigma_flat = crt_sigma[:sigma_size ] + 1j*crt_sigma[sigma_size:]
		delta_sol = crt_delta[0] + 1j*crt_delta[1]
		sigma_r_sol = jnp.reshape(sigma_flat,original_sigma_r_shape[:-1])

		return jnp.tensordot(sigma_r_sol,jnp.ones(original_sigma_r_shape[-1]),axes=0), delta_sol
	#!
	def _calc_gr(self,f,Q,gr0=None, root_method = 'broyden1'):
		# define the bare Hamiltonian: epsling + Doppler
		h_r_bare = self._Doppler_w_r(Q)

		# h_delta from gap 
		h_r_delta = lambda gap : self._Delta_p(gap)

		# g as a function of sigma_r and delta
		gr_function = lambda sigma_r, delta : self._hr2gr(h_r_bare - sigma_r - h_r_delta(delta))

		sigma_r, delta = self._self_cons_delta_sigma(f,Q,gr0,root_method)

		return gr_function(sigma_r,delta)
	#! 
	def _calc_gap(self,g):
		### This method computes the gap self consistently given the Greens function degree of freedom
		
		### First we compute the propert Keldysh Green's function 
		gk = self._f2gk(g)
		
		### Now we compute the relevant Nambu trace 
		### This will also reduce the tensor shape so we include inside this the gap function which is a tensor with the same shape as the Nambu tensors 
		#tr = np.trace( self.gap_function*self._NambuMul( 0.5*(self.Nambu_matrices[1] - 1.j*self.Nambu_matrices[2]), gk )  ) ### Trace should be over the nambu axes which are the first two axes and default for np.trace 
	
		integrand = (self.gap_function*gk)[0,1,:,:] ### We select the lower matrix element 
		
		#? not subtracting the Tc part, but everything is normalized according to Tc?
		### Now we integrate over energy and frequency and multiply by BCS constant (factor of 0.25 is by definition of Keldysh part)
		return -0.25*self.BCS_coupling*self._integrate(integrand)### Call custom built integrator which is designed to handle adaptive grids 

	#################################
	### SET SIMULATION PARAMETERS ### 
	#################################

	#!
	def set_d_wave(self,nodal=False):
		### We change from s-wave to d-wave gap function (option to switch nodal and anti-nodal, default is antinodal)
		if nodal: self.gap_function = jnp.sqrt(2.)*jnp.sin(2.*self.theta_grid) ### The factor of sqrt(2) is normalization 
		else: self.gap_function = jnp.sqrt(2.)*jnp.cos(2.*self.theta_grid)
	#!	
	def set_s_wave(self):
		### We change from d-wave to s-wave gap function 
		self.gap_function = jnp.ones_like(self.theta_grid) ### Trivial isotropic gap
	#!
	def set_BCS_coupling(self,BCS_coupling):
		### Sets the BCS coupling, often paired with an estimate based on clean s-wave theory
		self.BCS_coupling = BCS_coupling
	#! 
	def set_Tc(self,Tc):
		### Allows to set the nominal Tc scale from default of one 
		self.Tc = Tc 
	#!
	def set_gamma_imp(self,gamma_imp):
		### Set the elastic scattering rate
		self.gamma_imp = gamma_imp
	#! 
	def set_Dynes_eta(self,eta):
		### Sets a finite value of the Dynes broadening (eta) parameter -- PAIR BREAKING 
		self.eta = eta 
		
		### This will strongly renormalize Tc approximately linearly at small eta with coefficient dTc/deta = -pi/4 
	#! 
	def set_temperature(self,T):
		### Set the base temperature 
		self.T = T 
		
		### We also form the appropriate occupation function tensor 
		self.fd_tensor = self.Nambu_matrices[0]*jnp.tanh(0.5*self.w/self.T)
	#? to be implemented later 
	def set_times(self,times):
		### Simulation times passed as an array
		self.times = times
		self.ntimes = len(self.times)
		self.t0 = times[0]
		self.tf = times[-1] 
	#! 	
	def set_Q0(self,Q0):
		### Sets a state equilibrium value of Q 
		self.Q0 = Q0
	#! 
	def set_Q_function(self,Q_t):
		### Because we often deal with time dependent vector potential here we pass a call to the function which will return the instanenous value of Q(t) as a vector 
		self.Q_t = Q_t 
		
		### We also generate an array of the values for each simulation time
		self.Q_vs_t = self.Q_t(self.times) + self.Q0
	
	####################################
	### RUN EQUILIBRIUM CALCULATIONS ###
	#################################### 
	#! 
	def calc_BCS_coupling(self):
		### This is a useful function which gives the relation between BCS lambda and Tc for a fixed cutoff in the case of clean s-wave BCS equation 
		return 1./jnp.log(nambu_support.get_bcs_gap_constant()*self.cutoff/self.Tc) 
	#! 
	def calc_eq(self,gr0=None):
		### This computes the equilibrium gap and Green's function (optionally) given initial guesses to pass to the solver 
		
		gr = self._calc_gr(self.fd_tensor,self.Q0,gr0) 
		
		return self._calc_gap(self._rf2g(gr,self.fd_tensor)), gr
		
	def precompute_hr(self,nDelta,nQ = None,Q_max = None):
		"""This will run a precomputing routine where gR is computed as a function of Delta(t) and Q(t) and then stored with interpolator to enable fast usage for ODE solver"""
		
		### We will first generate the grid of points to interpolate over 
		self.Delta_max = 2.*nambu_support.get_bcs_ratio()*self.Tc 
		self.Q_max = 10.*nambu_support.get_bcs_ratio()*self.Tc ### voltage can be very large potentially 
		if Q_max is not None: self.Q_max = Q_max ### A more refined specification
		self.nDelta = nDelta 
		self.nQ = nQ 
		
		self.Deltas = self.Delta_max*( jnp.linspace(0.,1.,self.nDelta,dtype=complex) )**4 ### We use a non-uniform sampling which is denser at small values of Delta 
	
		if nQ is not None: 
			self.Qs = jnp.linspace(-self.Q_max,self.Q_max,nQ)
		if nQ is None:
			self.Q_max = 0.
			self.nQ = 1 
			self.Qs = jnp.array([0.]) 
			
		self.Delta_grid, self.Q_grid = jnp.meshgrid(self.Deltas,self.Qs,indexing='ij')
		self.precompute_grid_shape = self.Delta_grid.shape
		self.sigma_r_grid = jnp.zeros((*self.Nambu_shape, *self.precompute_grid_shape),dtype=complex )
		### Now we precompute the solution to the SCBA for each point in the grid
		
		for i in range(self.nDelta):
			for j in range(self.nQ):
				gap = self.Delta_grid[i,j]
				Q = self.Q_grid[i,j]
				
				if self.verbose: 
					print(f"Precompute loop: {i}/{self.nDelta} x {j}/{self.nQ}")
					print(f"Gap: {jnp.abs(gap):0.3f}")
					print(f"Q: {Q:0.3f}")
				t0 = time.time()
				gr = self._calc_gr(gap,Q)
				if gr is not None: 
					self.sigma_r_grid[...,i,j] = self._sigma_r(gr)
				t1 = time.time()
				if self.verbose: print(f"Time: {t1-t0:0.2f}s\n")

def total_solver_jitted(obj,sigma,delta):
	jitted_function = jax.jit(total_solver)
	return jitted_function(obj._sigma_solver,obj._delta_solver,sigma,delta)

def total_solver(obj,sigma,delta):
	return obj._iterative_jax_solver(obj._sigma_solver,obj._delta_solver,sigma,delta,tol = 1e-3)




















		
