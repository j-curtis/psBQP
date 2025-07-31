### Synthesized by ChatGPT based on a jupyter notebook which it then cleaned up and compactified
### Jonathan Curtis 
### 07/31/2025

import numpy as np
from scipy import integrate as intg
from scipy import optimize as opt

### All energies are in units of Tc 
### All velocities are in units of vF 
### Current is in units of zero-temperature superfluid density 

class BQPDynamics:
	def __init__(self, nxi, ntheta, cutoff):
		self.nxi = nxi if nxi % 2 == 0 else nxi + 1  # Ensure even number
		self.ntheta = ntheta
		self.cutoff = cutoff
		
		self.xi_grid, self.theta_grid = self._generate_momentum_grid()
		
		self.dxi = self.xi_grid[1,0] - self.xi_grid[0,0]
		
		self.grid_shape = self.xi_grid.shape  # (Nxi, Ntheta)
		self.grid_size = np.prod(self.grid_shape)
		
		self.nks_shape = (2,*self.grid_shape)

	def _generate_momentum_grid(self):
		xis = np.linspace(-self.cutoff, self.cutoff, self.nxi)
		thetas = np.linspace(0., 2. * np.pi, self.ntheta, endpoint=False)
		xi_grid, theta_grid = np.meshgrid(xis, thetas, indexing='ij')
		return xi_grid, theta_grid

	def BQP_energy(self, gap):
		"""Compute Bogoliubov quasiparticle energy (no Doppler shift)"""
		return np.sqrt(self.xi_grid**2 + gap**2)

	def BQP_doppler(self,gap,Q):
		"""Compute Bogoliubov quasiparticle energy including the Doppler shift"""
		### Returns a tensor of shape (2,Nxi,Ntheta)
		epm = np.zeros(self.nks_shape)
		epm[0,...] = self.BQP_energy(gap) + 0.5*Q[0]*np.cos(self.theta_grid) + 0.5*Q[1]*np.sin(self.theta_grid)
		epm[1,...] = self.BQP_energy(gap) - 0.5*Q[0]*np.cos(self.theta_grid) - 0.5*Q[1]*np.sin(self.theta_grid)
		
		return epm
		
	def gap_eqn_rhs(self,gap,nks):
		"""Returns the function we should find the root of in order to solve the gap equation"""
		E = self.BQP_energy(gap)
		
		integrand = 0.5*(np.ones_like(E) - np.sum(nks,axis=0) )/E - 0.5*np.tanh(self.xi_grid/2.)/self.xi_grid
		
		return np.sum(integrand)*self.dxi/self.ntheta 
		
	def eq_nks(self,gap,Q,T):
		### Equilibrium occupation functions for given gap, doppler shift, and temperature 
		return 1./(np.exp(self.BQP_doppler(gap,Q)/T) + 1. )
		
	def solve_gap_eq(self,Q,T):
		f = lambda x: self.gap_eqn_rhs(x,self.eq_nks(x,Q,T))
		
		lpoint = 0.
		rpoint = self.cutoff
		
		if f(lpoint)*f(rpoint) >0.:
			return 0. 
		else:
			return opt.brentq(f,lpoint,rpoint) ### brentq is a numerical method which finds the root of a function provided it changes sign on the interval lpoint, rpoint. Here we expect that either the gap is zero or it has a sign change -- this should be verified out of equilibrium to be true as well. 
			
	def solve_gap_nks(self,nks):
		"""This solves the gap equation using a generic quasiparticle distribution function which may be out of equilibrium"""
		f = lambda x : self.gap_eqn_rhs(x,nks)
		
		lpoint = 0.
		rpoint = self.cutoff
		
		if f(lpoint)*f(rpoint) >0.:
			return 0. 
		else:
			return opt.brentq(f,lpoint,rpoint) ### brentq is a numerical method which finds the root of a function provided it changes sign on the interval lpoint, rpoint. Here we expect that either the gap is zero or it has a sign change -- this should be verified out of equilibrium to be true as well. 

		
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
			
		
		
		
		
