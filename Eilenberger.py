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
Pauli = [ np.eye(2,dtype=complex), np.array([[0.j,1.],[1.,0.j]]), np.array([[0.j,-1.j],[1.j,0.j]]), np,array([[1.0,0.j],[0.j,-1.]] ) ] ]
Paulmin = 0.5*(Pauli[1] -1.j*Pauli[2] )

class Eilenberger:
	def __init__(self, nw, ntheta, cutoff):
		self.nw = nw if nw % 2 == 0 else nw + 1  # Ensure even number
		self.ntheta = ntheta
		self.cutoff = cutoff
		
		self.w_grid, self.theta_grid = np.meshgrid( np.linspace(-self.cutoff,self.cutoff,self.nw), np.linspace(0.,2.*np.pi,self.ntheta,endpoint=False),indexing = 'ij') 
		
		self.dw = self.w_grid[1,0] - self.w_grid[0,0]
		
		self.grid_shape = self.w_grid.shape  # (Nw, Ntheta)
		self.grid_size = np.prod(self.grid_shape)
		
		
		### We now set up the shapes for the matrix equations of motion 
		self.Nambu_shape = (2,2,*self.grid_shape)  ### Nambu matrices (i.e. retarded and Keldysh propagators have this shape 
		self.Keldysh_shape = (2,*self.Nambu_shape) ### We now double for the Keldysh dof (we store only [R,K] )
		
		self.gap_function = np.ones_like(self.theta_grid) ### default is s-wave gap, can be set internally for d-wave 

	#################################
	### SET SIMULATION PARAMETERS ### 
	#################################

	def set_d_wave(self):
		### We change from s-wave to d-wave gap function 
		self.gap_function = np.sqrt(2.)*np.cos(2.*self.theta_grid) ### The factor of sqrt(2) is normalization 
		
	def set_s_wave(self):
		### We change from s-wave to d-wave gap function 
		self.gap_function = np.ones_like(self.theta_grid) ### Trivial isotropic gap

	def set_elastic_scattering(self,tau_imp):
		### Set the elastic scattering rate in units of Tc 
		self.tau_imp = tau_imp 
		
	def set_temperature(self,T):
		### Set the base temperature in units of Tc 
		self.T = T 
		### We also form the appropriate occupation function tensor 
		self.fd_tensor = np.zeros(self.Nambu_shape,dtype=complex)
		self.fd_tensor[0,0,...] = np.tanh(0.5*self.w_grid/self.T)
		self.fd_tensor[1,1,...] = np.tanh(0.5*self.w_grid/self.T)
		
		

		
	def gap_eqn_rhs(self,gap,nks):
		"""Returns the function we should find the root of in order to solve the gap equation"""
		E = self.BQP_energy(gap)
		
		integrand = 0.5*self.gap_function**2*(np.ones_like(E) - np.sum(nks,axis=0) )/E - 0.5*np.tanh(self.xi_grid/2.)/self.xi_grid ### Added the possibiltiy of a non-trivial pairing channel 
		
		return np.sum(integrand)*self.dxi/self.ntheta 
		
	def eq_nks(self,gap,Q,T):
		### Equilibrium occupation functions for given gap, doppler shift, and temperature 
		return 1./(np.exp(self.BQP_doppler(gap,Q)/T) + 1. )
	
	def solve_gap_eq(self,Q,T):
		f = lambda x: self.gap_eqn_rhs(x,self.eq_nks(x,Q,T))
		
		return self._root_finder(f)
					
	def solve_gap_nks(self,nks):
		"""This solves the gap equation using a generic quasiparticle distribution function which may be out of equilibrium"""
		f = lambda x : self.gap_eqn_rhs(x,nks)
		
		return self._root_finder(f)

		
	### METHODS FOR EQUATIONS OF MOTION 
	def eom_rhs(self,t,X,Q_t,T,rta_rate):
		"""RHS of the equations of motion for the RTA kinetic equation"""
		### This method will be the relaxation time approximation to the kinetic equation for the BQPs
		### We allow a time dependent vector potential by passing the function Q_t which we call to get the instantaneous potential
		
		### First we reshape the occupation functions which are in the X dof 
		nks = X.reshape(self.nks_shape)
		
		### We now need to compute the instantenous self-consistent gap 
		gap = self.solve_gap_nks(nks)
		
		### Now we compute the instantenous occupation functions 
		rhs = -rta_rate*( nks - self.eq_nks(gap,Q_t(t),T)) 
		
		### Now we flatten and return 
		return rhs.ravel()
		
	def solve_eom(self,nks0,Q_t,T,rta_rate,times):
		"""Solves the self-consistent BQP EOM for a given initial occupation function and time-dependent vector potential"""
		
		### Simulation specific parameters 
		self.times = times 
		self.dt = self.times[1] - self.times[0]
		self.t0 = self.times[0]
		self.tf = self.times[-1]
		self.ntimes = len(self.times)
		
		X0 = nks0.ravel()
		
		sol = intg.solve_ivp(self.eom_rhs,(self.t0,self.tf),X0,args=(Q_t,T,rta_rate),t_eval=self.times,maxstep = 0.1*self.dt )
		
		### Now we reshape and return the solution/save to the simulation class
		nks_vs_t = sol.y
		new_shape = (*(self.nks_shape),self.ntimes)
		nks_vs_t = nks_vs_t.reshape(new_shape)

		
		self.nks_vs_t = nks_vs_t
		
		return nks_vs_t
		
	def calc_current(self,nks,Q):
		"""Computes the total current for a given vector potential and quasiparticle distribution"""
		current = [0.,0.]
		
		vhat = [ np.cos(self.theta_grid),np.sin(self.theta_grid) ] 
		for i in range(2):
			current[i] +=  Q[i] +2.*np.tensordot(vhat[i], nks[0,...] - nks[1,...],axes=[[0,1],[0,1]] )*self.dxi/self.ntheta
		
		return current
			
		
		
		
		
