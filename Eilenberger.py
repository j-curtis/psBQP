### Jonathan Curtis 
### 08/11/2025
### Code to simulate dynamics of Eilenberger equation

import numpy as np
from scipy import integrate as intg
from scipy import optimize as opt

### All energies are in units of Tc 
### All velocities are in units of vF 
### Current is in units of zero-temperature superfluid density (I think?) 


BCS_ratio = 1.765387449618725 ### Ratio of Delta(0)/Tc in BCS limit 

### Various Pauli matrices 
Pauli = [ np.eye(2,dtype=complex), np.array([[0.j,1.],[1.,0.j]]), np.array([[0.j,-1.j],[1.j,0.j]]), np.array([[1.0,0.j],[0.j,-1.]]) ]
Paulmin = 0.5*(Pauli[1] -1.j*Pauli[2] )

class Eilenberger:
	def __init__(self, nw, ntheta, cutoff):
		self.nw = nw if nw % 2 == 0 else nw + 1  # Ensure even number
		self.ntheta = ntheta
		self.cutoff = cutoff
		
		self.w_grid, self.theta_grid = np.meshgrid( np.linspace(-self.cutoff,self.cutoff,self.nw), np.linspace(0.,2.*np.pi,self.ntheta,endpoint=False),indexing = 'ij') 
		
		self.grid_shape = self.w_grid.shape  # (Nw, Ntheta)
		self.grid_size = np.prod(self.grid_shape)
		
		self.dw = self.w_grid[1,0] - self.w_grid[0,0]
		
		### Internal eta for broadening of spectral functions 
		self.zero = 1.5*self.dw ### Should be small but precise value should not matter 
		
		### Internal default parameters for SCBA solver 
		self.scba_step = 0.25 ### Update gradient step 
		self.scba_err = 1.e-3 ### total error for SCBA convergence 
		self.scba_max_step = 1000 ### Total number of iterations before we throw an error 
	
		### We now set up the shapes for the degrees of freedom
		self.Nambu_shape = (2,2,*self.grid_shape)  ### Nambu matrices (i.e. retarded and Keldysh propagators have this shape 
		self.Keldysh_shape = (2,*self.Nambu_shape) ### We now double for the Keldysh dof (we store only [R,K] )
		
		### Now we promote all grids and matrices to the correct Nambu shape 
		self.Nambu_matrices = [ np.tensordot( sigma ,np.ones_like(self.w_grid) ,axes=0) for sigma in Pauli ] ### These are now Pauli matrices with the correct tensor shape for Nambu Greens functions 
		self.Nambu_w_grid = np.tensordot(np.ones((2,2),dtype=complex),self.w_grid ,axes = 0) ### This is the tensor grid of w values with the correct shape for Nambu GF 
		self.Nambu_theta_grid = np.tensordot(np.ones((2,2),dtype=complex),self.theta_grid,axes=0)  ### This is the tensor grid of theta values with the correct shape for Nambu GF 
		
		### Gap functions 
		self.gap_function = np.ones_like(self.theta_grid) ### default is s-wave gap, can be set internally for d-wave 
		self.Nambu_gap_function = np.ones_like(self.Nambu_matrices[0])
		
		### Default function call for supercurrent will just return zero 
		self.Q_t = lambda x: np.array([0.,0.]) 

	
	########################
	### INTERNAL METHODS ### 
	########################
	@classmethod
	def _Nambu_mult(cls,x,y):
		### Matrix multiplies two tensors according to their Nambu indices and tensor multiplies the rest 
		z = np.tensordot( x,y,axes=[[1],[0]] ) 
		return z 
	
	def _r2a(self,gr):
		### In the quasiclassical formalism a retarded object can be made advanced by conjugating 
		ga = np.transpose(np.conjugate(gr),axes=(1,0,2,3))
		
		ga = np.tensordot(self.Nambu_matrices[3],ga,axes=[[1],[0]])
		ga = np.tensordot(ga,self.Nambu_matrices[3],axes=[[1],[0]])
		
		return ga 
		
	def _f2gk(self,dof):
		### This method will convert the dof = [gr,f] tensor in to a Keldysh correlation function 
		gr = dof[0,...]
		f = dof[1,...]
		
		gk = np.tensordot
		
		

	def _sigma_r(self,gr): 
		### This method computes the retarded self energy from gr   
		sigma = np.zeros_like(gr) 
		
		### Impurity scattering contributions 
		impurity_scattering_tensor = 0.5/self.tau_imp*np.ones((self.ntheta,self.ntheta),dtype=complex)/self.ntheta 
		
		sigma = np.tensordot(gr,impurity_scattering_tensor,axes=[[3],[0]]) ### This integrates over the angle and replaces it by a constant 
		
		return sigma 
		
	def _Doppler_w_r(self,Q):
		### returns the Doppler shifted frequencies with retarded causality 
		return self.Nambu_w_grid - Q[0]*np.cos(self.Nambu_theta_grid) - Q[1]*np.sin(self.Nambu_theta_grid) + 1.j*np.ones_like(self.Nambu_w_grid)
			
	def _Delta_p(self,gap):
		### Returns the momentum resolved gap given gap amplitude 
		return gap*self.Nambu_gap_function 
			
	def _Nambu_det(self,a):
		### Computes the determinant of a Nambu matrix as a tensor over the grid of frequency and angle 
		det = a[0,0,...] * a[1,1,...] - a[0,1,...]*a[1,0,...] ### Has shape of the frequency and mesh grid
		out = np.tensordot(np.ones((2,2),dtype=complex),det,axes=0) 

		return out 
		
	def _hr2gr(self,hr):
		### Inverts and normalizes a retarded effective Hamiltonian
		return - hr/np.sqrt(-self._Nambu_det(hr)) 
			
	def _calc_gr(self,gap,Q):

		### This method solves the Dyson equation for the retarded Green's function given the gap and vector potential Q 
		### We iterate self-consistently 
		
		w_Q = self._Doppler_w_r(Q)  
		delta_p = self._Delta_p(gap) 
		
		### Bare inverse Greens function 
		hr0 = self.Nambu_matrices[3]*w_Q - 1.j*delta_p*self.Nambu_matrices[2]

		### Initial Greens function 
		hr = hr0.copy()
		gr = self._hr2gr(hr)
		
		### New Greens functions
		hr_new = hr +self.scba_step*( hr0 - self._sigma_r(gr)  - hr)
		gr_new = self._hr2gr(hr_new)  
		
		err = np.sum( np.abs(hr_new - hr) )
		count = 0 
		print("Starting error: "+str(err)) 
	
		
		while err > self.scba_err: 
			if count > self.scba_max_step:
				print("Max step count {m} exceeded.".format(m=self.scba_max_step))
				return None 
	
			### Update the new functions to the old ones 		
			hr = hr_new 
			gr = gr_new
			
			hr_new = hr + self.scba_step*(hr0 - self._sigma_r(gr) - hr )  ### Increment the inverse GF  
			gr_new = self._hr2gr(hr_new) ### Compute new Green's function 

			err = np.sum( np.abs(hr_new - hr) )
			print("Loop {c}: Error {e:0.4f}".format(c=count, e = err ))
			
			count += 1 
		
		return gr 
		
	
	
	
	#################################
	### SET SIMULATION PARAMETERS ### 
	#################################

	def set_d_wave(self):
		### We change from s-wave to d-wave gap function 
		self.gap_function = np.sqrt(2.)*np.cos(2.*self.theta_grid) ### The factor of sqrt(2) is normalization 
		
	def set_s_wave(self):
		### We change from d-wave to s-wave gap function 
		self.gap_function = np.ones_like(self.theta_grid) ### Trivial isotropic gap

	def set_impurity_scattering(self,tau_imp):
		### Set the elastic scattering rate in units of Tc 
		self.tau_imp = tau_imp 
		
	def set_temperature(self,T):
		### Set the base temperature in units of Tc 
		self.T = T 
		### We also form the appropriate occupation function tensor 
		self.fd_tensor = np.zeros(self.Nambu_shape,dtype=complex)
		self.fd_tensor[0,0,...] = np.tanh(0.5*self.w_grid/self.T)
		self.fd_tensor[1,1,...] = np.tanh(0.5*self.w_grid/self.T)
	
	def set_times(self,times):
		### Simulation times passed as an array
		self.times = times
		self.ntimes = len(self.times)
		self.t0 = times[0]
		self.tf = times[-1] 
		
	def set_Q_function(self,Q_t):
		### Because we often deal with time dependent vector potential here we pass a call to the function which will return the instanenous value of Q(t) as a vector 
		self.Q_t = Q_t 
		
		### We also generate an array of the values for each simulation time
		self.Q_vs_t = self.Q_t(self.times)
	

		
