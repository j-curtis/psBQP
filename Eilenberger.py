### Jonathan Curtis 
### 08/11/2025
### Code to simulate dynamics of Eilenberger equation

import numpy as np
from scipy import integrate as intg
from scipy import optimize as opt


BCS_gap_constant = 2.*np.exp(np.euler_gamma)/np.pi ### 2e^gamma/pi constant often appearing in BCS integrals 
BCS_ratio = 1./(0.5*BCS_gap_constant) #1.765387449618725 ### Ratio of Delta(0)/Tc in BCS limit 

### Scope for defining and handling units more easily 

	
	

### Various Pauli matrices 
Pauli = [ np.eye(2,dtype=complex), np.array([[0.j,1.],[1.,0.j]]), np.array([[0.j,-1.j],[1.j,0.j]]), np.array([[1.0,0.j],[0.j,-1.]]) ]
Paulimin = 0.5*(Pauli[1] -1.j*Pauli[2] )

class Eilenberger:
	def __init__(self, nw, ntheta, cutoff):
		self.verbose = False ### If this is true we will have more information and feedback given during calculations
	
		self.nw = nw if nw % 2 == 0 else nw + 1  # Ensure even number
		self.ntheta = ntheta
		self.cutoff = cutoff
		
		self.Tc = 1. ### By default we use units where Tc is one 
		
		self.w_grid, self.theta_grid = np.meshgrid( np.linspace(-self.cutoff,self.cutoff,self.nw), np.linspace(0.,2.*np.pi,self.ntheta,endpoint=False),indexing = 'ij') 
		
		self.grid_shape = self.w_grid.shape  # (Nw, Ntheta)
		self.grid_size = np.prod(self.grid_shape)
		
		self.dw = self.w_grid[1,0] - self.w_grid[0,0]
		
		### Internal eta for broadening of spectral functions 
		self.zero = 0.1*self.dw ### Should be small but precise value should not matter 
		
		### Internal default parameters for SCBA solver 
		self.scba_step = 0.05 ### Update gradient step 
		self.scba_err = 1.e-3 ### total error for SCBA convergence 
		self.scba_max_step = 1000 ### Total number of iterations before we throw an error 
	
		### Generate the necessary Nambu-shaped tensors 
		### Nambu tensor class is not yet working 
		self.Nambu_matrices = [ np.tensordot(sigma, np.ones_like(self.w_grid),axes=0 ) for sigma in Pauli ] ### These are now Pauli matrices and identity function on the momenta/frequency 
		self.Nambu_shape = self.Nambu_matrices[0].shape ### Should be (2,2,nw,ntheta) 
		
		self.w = self._scalar2Nambu(self.w_grid)
		self.theta = self._scalar2Nambu(self.theta_grid) 
		
		self.gap_function = np.ones_like(self.theta) 
	
		### We now set up the shapes for the Keldysh degrees of freedom
		self.Keldysh_shape = (2,*self.Nambu_shape) ### We now double for the Keldysh dof (we store only [R,K] )
		
		### Default function call for supercurrent will just return zero 
		self.Q_t = lambda x: np.array([0.,0.]) 
		self.Q0 = np.array([0.,0.]) ### A static background which is set to zero by default 
	
	########################
	### INTERNAL METHODS ### 
	########################

	def _NambuMul(self,x,y):
		### For the time being we will use a homebuilt overload for matrix multiplication for Nambu tensors until the tensor class can be tested more 
		return np.einsum('ijnm,jknm->iknm', x,y)
		
	def _scalar2Nambu(self,x):
		### Promotes a scalar tensor function to a Nambu compatible tensor 
		return np.tensordot(np.ones((2,2),dtype=complex),x,axes=0) 
	
	def _rf2g(self,gr,f):
		### Promotes a pair (gr,f) to a single Keldysh object
		g = np.zeros(self.Keldysh_shape,dtype=complex)
		g[0,...] = gr
		g[1,...] = f 
		
		return g 
	
	def _r2a(self,gr):
		### This method conjugates a retarded object to get an advanced one 
		ga = np.transpose(np.conjugate(gr),axes=(1,0,2,3))
		
		ga = self._NambuMul(self.Nambu_matrices[3],ga)
		ga = self._NambuMul(ga,self.Nambu_matrices[3])
		
		return ga 
		
	def _f2gk(self,g):
		### This method takes a gf = [gr, f] object and computes the proper Keldysh correlation funciton
		gr = g[0,...]
		f = g[1,...]
		
		gk = self._NambuMul(gr,f) - self._NambuMul(f,self._r2a(gr)) 
		
		return gk 
				
	def _sigma_r(self,gr): 
		### This method computes the retarded self energy from gr   
		sigma = np.zeros_like(gr) 
		
		### Impurity scattering contributions 
		impurity_scattering_tensor = 0.5/self.tau_imp*np.ones((self.ntheta,self.ntheta),dtype=complex)/self.ntheta 
		
		sigma += np.tensordot(gr,impurity_scattering_tensor,axes=[[3],[0]]) ### This integrates over the angle and replaces it by a constant 
		
		return sigma 
		
	def _Doppler_w_r(self,Q):
		### returns the Doppler shifted frequency Nambu tensor with retarded causality 		
		return (self.w - Q[0]*np.cos(self.theta) - Q[1]*np.sin(self.theta) + 1.j*self.zero*np.ones_like(self.w) )*self.Nambu_matrices[3] 
			
	def _Delta_p(self,gap):
		### Returns the momentum resolved Nambu tensor gap  		
		### Allows for a complex gap 
		return 1.j*np.real(gap) * self.Nambu_matrices[2]*self.gap_function -1.j* np.imag(gap)*self.Nambu_matrices[1]*self.gap_function
		
	def _Nambu_det(self,a):
		### Computes the determinant of a Nambu matrix as a tensor over the grid of frequency and angle 			
		det = a[0,0,...] * a[1,1,...] - a[0,1,...]*a[1,0,...] ### Has shape of the frequency and mesh grid
		out = self._scalar2Nambu(det)

		return out 
				
	def _hr2gr(self,hr):
		### Inverts and normalizes a retarded effective Hamiltonian
		return - hr/np.sqrt(self._Nambu_det(hr)) 
	
	def _calc_gr(self,f,gr0 = None,gap0 = None):
		### This method computes the retarded Greens function and gap self consistently given the Keldysh function f and (optionally) an initial guess for gr0 and gap0
		
		### Bare inverse Green's function
		### Not a guess but is the static part of gr0 inverse 
		hr0 = self._Doppler_w_r(self.Q0) ### For now we just use the equilibrium current 
		
		### Initial guess for gap
		if gap0 is None:
			gap0 = BCS_ratio*self.Tc
		
		### Initial guess for gr0
		if gr0 is None: 
			gr0 = self._hr2gr(hr0)
		
		hr = hr0  - self._Delta_p(gap0) - self._sigma_r(gr0) 
		gr = self._hr2gr(hr) 
		
		### Update function 
		def _update_func(hr):
			gr = self._hr2gr(hr)
			sigma = self._sigma_r(gr) 
			
			gap = self._calc_gap(self._rf2g(gr,f))
			
			### Now we update the inverse Green's function 
			return hr0 - self._Delta_p(gap) - sigma 
		
		count = 0 
		hr_new = hr + self.scba_step*( _update_func(hr) - hr ) 
		gr_new = self._hr2gr(hr_new) 
		
		err = np.sum(np.abs( gr_new - gr )) 
		
		if self.verbose: print("Starting error: "+str(err)) 
		count_exceeded = False
		while err > self.scba_err: 
			if count > self.scba_max_step:
				if self.verbose: print("Max step count {m} exceeded.".format(m=self.scba_max_step))
				count_exceeded = True
				break
	

			hr_new = hr + self.scba_step*( _update_func(hr) - hr )  ### Increment the inverse GF  
			gr_new = self._hr2gr(hr_new) ### Compute new Green's function 

			err = np.sum( np.abs(hr_new - hr) )
			if self.verbose: print("Loop {c}: Error {e:0.4f}".format(c=count, e = err ))
			
			count += 1 
			
			### Update the new functions to the old ones 		
			hr = hr_new 
			gr = gr_new
			
		### We must decide what to return if the result doesn't converge
		### For now we will return the gap = 0 and the greens function will be returned as None 
		if count_exceeded:
			gap = 0.
			gr = None 
			
		else:
			gap = self._calc_gap(self._rf2g(gr,f)) 
				
		### Now we return the greens function and the gap

		return gr, gap
		
	def _calc_gap(self,g):
		### This method computes the gap self consistently given the Greens function degree of freedom
		
		### First we compute the propert Keldysh Green's function 
		gk = self._f2gk(g)
		
		### Now we compute the relevant Nambu trace 
		### This will also reduce the tensor shape so we include inside this the gap function which is a tensor with the same shape as the Nambu tensors 
		tr = np.trace( self.gap_function*self._NambuMul( 0.5*(self.Nambu_matrices[1] - 1.j*self.Nambu_matrices[2]), gk )  ) ### Trace should be over the nambu axes which are the first two axes and default for np.trace 
	
		### Now we integrate over energy and frequency 
		rhs = -0.25j*np.sum(tr)*self.dw/self.ntheta
	
		### We now multiply by the BCS constant which has been preset 
		return self.BCS_coupling*rhs 
	
	#################################
	### SET SIMULATION PARAMETERS ### 
	#################################

	def set_d_wave(self):
		### We change from s-wave to d-wave gap function 
		self.gap_function = np.sqrt(2.)*np.cos(2.*self.theta_grid) ### The factor of sqrt(2) is normalization 
		
	def set_s_wave(self):
		### We change from d-wave to s-wave gap function 
		self.gap_function = np.ones_like(self.theta_grid) ### Trivial isotropic gap
	
	def set_BCS_coupling(self,BCS_coupling):
		### Sets the BCS coupling, often paired with an estimate based on clean s-wave theory
		self.BCS_coupling = BCS_coupling
		
	def set_Tc(self,Tc):
		### Allows to set the nominal Tc scale from default of one 
		self.Tc = Tc 

	def set_tau_imp(self,tau_imp):
		### Set the elastic scattering rate
		self.tau_imp = tau_imp 
		
	def set_temperature(self,T):
		### Set the base temperature 
		self.T = T 
		
		### We also form the appropriate occupation function tensor 
		self.fd_tensor = np.tanh(0.5*self.w/self.T)
	
	def set_times(self,times):
		### Simulation times passed as an array
		self.times = times
		self.ntimes = len(self.times)
		self.t0 = times[0]
		self.tf = times[-1] 
		
	def set_Q0(self,Q0):
		### Sets a state equilibrium value of Q 
		self.Q0 = Q0
		
	def set_Q_function(self,Q_t):
		### Because we often deal with time dependent vector potential here we pass a call to the function which will return the instanenous value of Q(t) as a vector 
		self.Q_t = Q_t 
		
		### We also generate an array of the values for each simulation time
		self.Q_vs_t = self.Q_t(self.times) + self.Q0
	
	####################################
	### RUN EQUILIBRIUM CALCULATIONS ###
	#################################### 
	
	def BCS_coupling(self):
		### This is a useful function which gives the relation between BCS lambda and Tc for a fixed cutoff in the case of clean s-wave BCS equation 
		return 1./np.log(BCS_gap_constant*self.cutoff/self.Tc) 
	
	def calc_eq(self,gr0=None,gap0=None):
		### This computes the equilibrium gap and Green's function (optionally) given initial guesses to pass to the solver 
		gr, gap = self._calc_gr(self.fd_tensor,gr0,gap0) 
		
		return gr, gap 
		






		
