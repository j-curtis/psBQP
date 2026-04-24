import jax 
import jax.numpy as jnp

from nambu_class import NambuTensor 
from system_state import SupercondctingState
import custom_optimizer as copt 
from jax import tree_util
import numpy as np
import matplotlib.pyplot as plt
from jax import core
import self_energy_class 

from matplotlib import pyplot as plt 

class UsadelEvolution:
    def __init__(self, grid_parameters, system_parameters, optimization_parameters = None, sigma_scatterings = None):
        # Extract grid parameters (support both old and new naming)
        # Omega grid
        if 'omega_grid' in grid_parameters:
            # Use pre-computed grid (new convention)
            omega_array = jnp.array(grid_parameters['omega_grid'])
            self.omega_size = len(omega_array)
        elif 'omega_sampling' in grid_parameters:
            # Generate linear grid (old convention)
            self.omega_size = grid_parameters['omega_sampling']
            if self.omega_size % 2 == 0:
                self.omega_size += 1  # Ensure odd for symmetry
            self.cutoff = grid_parameters.get('cutoff', grid_parameters.get('energy_cutoff'))
            omega_array = jnp.linspace(-self.cutoff, self.cutoff, self.omega_size)
        else:
            raise ValueError("grid_parameters must contain 'omega_grid' or 'omega_sampling'")

        # Cutoff (support both names)
        self.cutoff = grid_parameters.get('energy_cutoff', grid_parameters.get('cutoff'))

        # Eta parameter - just read directly, no modifications
        self.eta = grid_parameters['eta']

        # Store omega grid
        self.w_arr = jax.lax.stop_gradient(omega_array)

        # Extract system parameters
        self.critical_temperature = system_parameters['critical_temperature']
        self.temperature = system_parameters['temperature']
        self.bcs_coupling = self._get_BCS_coupling()

        # Current maximum (optional, for dynamics)
        self.current_maximum = system_parameters.get('current_maximum', 1.0)

        # Optimization and scattering parameters
        self.optimization_parameters = optimization_parameters
        self.sigma_scatterings = sigma_scatterings
        sigmas, breakoffs = self._generate_sigma_objects()
        self.sigma_objects = sigmas
        self.break_off_indicies = jax.lax.stop_gradient(breakoffs)


    @staticmethod
    def get_bcs_gap_constant() -> float:
        """A function returning the value of the bcs gap constant

        Returns:
            float: BCS gap constant 
        """
        return 2.*jnp.exp(jnp.euler_gamma)/jnp.pi

    @staticmethod
    def get_bcs_ratio() -> float:
        """ A function returning the value of the ratio of Delta(0)/Tc in the BCS limit

        Returns:
            float: BCS ratio
        """
        return 2./UsadelEvolution.get_bcs_gap_constant()  

    def _get_BCS_coupling(self)->float:
        """A function returning the bcs coupling given the critical temperature

        Args:
            cutoff (float): high energy cutoff of the theory
            critical_temperature (float): critical temperature in the system

        Returns:
            float: Bcs coupling constant lambda
        """
        ### This is a useful function which gives the relation between BCS lambda and Tc for a fixed cutoff in the case of clean s-wave BCS equation 
        return 1./jnp.log(UsadelEvolution.get_bcs_gap_constant()*self.cutoff/self.critical_temperature)* (2 *jnp.pi)

    def _integrate_over_omega(self,vals, axis):
        jitted_omega = jax.jit(copt.custom_jax_trapz, static_argnums = (2))
        return jitted_omega(y = vals, x = self.w_arr, axis = axis)

    def _generate_sigma_objects(self):
        break_off_indicies = [0]
        if self.sigma_scatterings is None:
            return None, break_off_indicies
        sigma_objects = []
        if 'Dynes' in self.sigma_scatterings:
            sigma_objects += [(self_energy_class.DynesScattering(scattering_rate= self.sigma_scatterings['Dynes'], omega_arr = self.w_arr))]
        
        if 'Elastic' in self.sigma_scatterings:
            sigma_objects += [(self_energy_class.ElasticScattering(scattering_rate= self.sigma_scatterings['Elastic'], omega_arr = self.w_arr))]

        if 'Phonon' in self.sigma_scatterings:
            sigma_objects += [(self_energy_class.PhononScattering(scattering_rate= self.sigma_scatterings['Phonon'], omega_arr = self.w_arr, temperature = self.temperature))]

        for crt_object in sigma_objects:
            break_off_indicies += [2 * np.prod((crt_object._sigma_shape())) * len(crt_object._get_sigma_indicies()) + break_off_indicies[-1]]

        return sigma_objects, break_off_indicies

    def _delta_flatten(self, delta):
        return jnp.array([delta.real,delta.imag], dtype = jnp.float32)

    def _delta_unflatten(self, delta_arr):
        return delta_arr[0] + 1.j*delta_arr[1]

    def _calc_gap(self, g_r, f):
        gk = g_r @ f + f @ g_r._involution()
        integrand = (gk)._trace('-') 
        return -0.25*self.bcs_coupling * (self._integrate_over_omega((integrand),axis = 0)).real/(2 *jnp.pi)

    def _hr2gr(self, hr, indicies = None):
        return -1.j* hr/jnp.sqrt( hr._determinant()) 

    def _Doppler_w_r(self, Q:float,indices = None):
        # Uniform eta broadening across all frequencies
        return NambuTensor(self.w_arr + 1.j * self.eta, 3)
        
    def _get_hr(self, Q:float, delta, sigma_r = None, indices = None):
        h_r = self._Doppler_w_r(Q = Q) - (NambuTensor(1j *  jnp.real(delta) * np.ones_like(self.w_arr), 2) +  NambuTensor(1j*jnp.imag(delta) * np.ones_like(self.w_arr), 1)) 
        if sigma_r is not None:
            h_r -= sigma_r
        return h_r

    def _get_current(self, gr, f, Q):
        prefactor = 1j / self.current_maximum * (jnp.pi/4 / (2 * jnp.pi ))
        tau_3 = NambuTensor(jnp.ones_like(self.w_arr) * 1, 3)
        gk = gr @ f + f @ gr._involution()
        integrand = (gr @ tau_3._commutator(gk) - gk @ tau_3._commutator(gr._involution()))._trace(3)

        #* Taking the real part since numerical error can pile up and give finite imaginary part of the current! 
        return (prefactor * self._integrate_over_omega(integrand, axis = 0) * Q).real 

    def _self_consistent_gr(self, crt_gr, Q, f):

        sigmar = NambuTensor(jnp.zeros_like(self.w_arr), 0)
        if self.sigma_objects is not None and crt_gr is not None:
            for crt_obj_index in range(jnp.shape(self.sigma_objects)[0]):
                sigmar += self.sigma_objects[crt_obj_index]._sigma_r(crt_gr, f)
        
        delta = self._calc_gap(crt_gr, f)
        hr = self._get_hr(Q, delta, sigmar)
        tau_3 = NambuTensor(np.ones_like(self.w_arr), 3)
        root_function =  jnp.stack( [(-crt_gr._commutator(hr) + 1j * Q**2 * tau_3._commutator(crt_gr @ (tau_3._commutator(crt_gr))))._trace(1), crt_gr._determinant()[0,0] + 1], axis = 1).flatten()
        #print(root_function) 
        return jnp.append(root_function.real, root_function.imag)

    def _self_consistent_delta(self, crt_delta, crt_gr, crt_f):
        return self._delta_flatten(crt_delta - self._calc_gap(crt_gr, crt_f))

    def _calculate_self_consistent_state(self, f, Q, gr0 = None):

        if gr0 is None:
            gap_0 = (1.0 * 1.7 + 0.0j) 
            gr0 = self._hr2gr(self._get_hr(Q = Q, delta = gap_0, sigma_r = None))

        else:
            gap_0 = self._calc_gap(gr0, f) #+ 0.2 # added this to see gap revival actually
            

        solver_function = lambda gr: self._self_consistent_gr(NambuTensor._unflatten_nambu_object(gr, self.w_arr.shape, (2,3)), Q, f = f)
        
        gr = copt._iterative_solver(solver_function, gr0._flatten_nambu_object((2,3)), optimization_parameters = self.optimization_parameters)

        return NambuTensor._unflatten_nambu_object(gr, self.w_arr.shape, (2,3))


    def _run_temperature_computation(self,Q,T, gr0 = None):
        f_grid = self._thermal_occupation_numbers(temperature = T)
        f0 = NambuTensor(f_grid, 0)
        self_cons_gr = self._calculate_self_consistent_state(f0, Q, gr0)
        delta = self._calc_gap(self_cons_gr, f0)
        current = self._get_current(self_cons_gr, f0,Q)
        return self_cons_gr, delta, current

    def _thermal_occupation_numbers(self, temperature: float) -> jnp.ndarray:
        return jnp.tanh(0.5*self.w_arr/temperature)

    def _calc_dtQ_prefactor(self, gr, temperature):
        deps_f = NambuTensor(jnp.gradient(self._thermal_occupation_numbers(temperature = temperature), self.w_arr, axis = 0), 0)
        tau_3 = NambuTensor(np.ones_like(self.w_arr), 3)
        deps_f_comm = tau_3._anti_commutator(deps_f)/2

        return self._integrate_over_omega((deps_f_comm + gr @ deps_f_comm @ gr._involution())._trace(3), axis = 0).real/(2 * jnp.pi)

    def _calc_dtQ_prefactor_new(self, gr, temperature):
        f = NambuTensor(self._thermal_occupation_numbers(temperature = temperature),0)
        gk = gr @ f + f @ gr._involution()
        deps_gk = gk._nambu_gradient(included_indices=(2,3), axis = 0, x_vals = self.w_arr)
        tau_3 = NambuTensor(np.ones_like(self.w_arr), 3)

        res =  gr @ tau_3 @ deps_gk + gr._involution() @ tau_3 @ deps_gk

        return  jnp.pi/4 * self._integrate_over_omega((gr @ tau_3 @ deps_gk + gr._involution() @ tau_3 @ deps_gk)._trace(3), axis = 0).real/(2 * jnp.pi)

    def _test_jacobian(Q, T, gr0 = None):
        f = NambuTensor(self._thermal_occupation_numbers(temperature = T),0)

        if gr0 is None:
            gap_0 = (1.0 * 1.7 + 0.0j) 
            gr0 = self._hr2gr(self._get_hr(Q = Q, delta = gap_0, sigma_rs = None))

        else:
            gap_0 = self._calc_gap(gr0, f) + 0.5 # added this to see gap revival actually

        solver_function = lambda gr: self._self_consistent_gr(NambuTensor._unflatten_nambu_object(gr, self.w_arr.shape, (2,3)), Q, sigmas = None, f = f)
        
        true_jacobian = jax.jacfwd(solver_function)(gr0._flatten_nambu_object((2,3)))

        plt.imshow(true_jacobian)

        res_1 = solver_function(gr0._flatten_nambu_object((2,3)))

        gr = copt._iterative_solver(solver_function, gr0._flatten_nambu_object((2,3)), optimization_parameters = self.optimization_parameters)

        return NambuTensor._unflatten_nambu_object(gr, self.w_arr.shape, (2,3))
