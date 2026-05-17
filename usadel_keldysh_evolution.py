"""
Usadel-Keldysh evolution class for real-time dynamics of superconducting systems.
Handles time evolution of retarded Green's function g^R and distribution function f.
"""

import numpy as np
from tqdm import tqdm
from nambu_keldysh_class import NambuKeldyshTensor, get_pauli_matrix
from state_object_class import StateObject
from equilibrium_class import EquilibriumSolver
import self_energy_class

from matplotlib import pyplot as plt

class UsadelKeldyshEvolution:
    """
    Main evolution class for Usadel equation in Keldysh formalism.
    Computes equilibrium states and real-time dynamics.
    """

    def __init__(self, grid_parameters, system_parameters, optimization_parameters=None, sigma_scatterings=None):
        """
        Initialize Usadel evolution solver.

        Args:
            grid_parameters: dict with omega_sampling, cutoff, time_sampling, time_duration
            system_parameters: dict with critical_temperature, eta, etc.
            optimization_parameters: solver settings
            sigma_scatterings: dict of scattering mechanisms and rates
        """
        # Store parameters
        self.grid_parameters = grid_parameters
        self.system_parameters = system_parameters
        self.optimization_parameters = optimization_parameters
        self.sigma_scatterings = sigma_scatterings

        # Generate time grid
        self._generate_time_grid()

        # Generate omega grid from extended time domain
        self._generate_omega_grid()
        # Set eta with warning if too small
        self.eta = system_parameters['eta']
        recommended_eta = 5.0 / self.tmax
        if self.eta < recommended_eta:
            print(f"WARNING: eta = {self.eta:.4f} is smaller than recommended value {recommended_eta:.4f}")
            print(f"         Recommended: eta >= 5/T_max = 10/{self.tmax:.2f}")
            print(f"         Consider using larger T_max or increasing eta to avoid numerical issues.")

        self.critical_temperature = system_parameters['critical_temperature']
        self.temperature = system_parameters['temperature']


    # ========== Grid and Parameter Setup ==========

    @staticmethod
    def get_bcs_gap_constant() -> float:
        """Return BCS gap constant: 2*exp(gamma_E)/pi.

        Returns:
            float: BCS gap constant ≈ 1.134
        """
        return 2.0 * np.exp(np.euler_gamma) / np.pi

    @staticmethod
    def get_bcs_ratio() -> float:
        """Return Delta(0)/T_c ratio in BCS limit.

        Returns:
            float: BCS ratio ≈ 1.764
        """
        return 2.0 / UsadelKeldyshEvolution.get_bcs_gap_constant()

    def _get_BCS_coupling(self) -> float:
        """Compute BCS coupling constant from critical temperature.

        Uses the relation between BCS coupling λ and T_c for a fixed cutoff
        in the clean s-wave BCS equation.

        Returns:
            float: BCS coupling constant λ
        """
        return 1.0 / np.log(UsadelKeldyshEvolution.get_bcs_gap_constant() * self.energy_cutoff / self.critical_temperature) * (2.0 * np.pi)

    def _generate_time_grid(self):
        """
        Generate time grid from grid_parameters.

        Reads grid_parameters dict and creates time array for evolution.
        Stores time grid and integration weights.

        Called by: __init__
        """
        # Extract time grid parameters (support multiple naming conventions)
        if 'time_sampling' in self.grid_parameters:
            self.ntpoints = self.grid_parameters['time_sampling']
        elif 'n_tpoints' in self.grid_parameters:
            self.ntpoints = self.grid_parameters['n_tpoints']
        else:
            raise ValueError("grid_parameters must contain 'time_sampling' or 'n_tpoints'")

        if 'time_duration' in self.grid_parameters:
            self.tmax = self.grid_parameters['time_duration']
        elif 't_max' in self.grid_parameters:
            self.tmax = self.grid_parameters['t_max']
        else:
            raise ValueError("grid_parameters must contain 'time_duration' or 't_max'")

        # Compute time step
        self.delta_t = self.tmax / (self.ntpoints - 1)

        # Generate time grid
        self.time_grid = np.linspace(-self.tmax, 0, self.ntpoints)

    def _generate_omega_grid(self):
        """
        Generate angular frequency (omega) grid as exact Fourier dual of time grid.

        Uses the evolution time grid [-T_max, 0] with N points and spacing dt = T_max/(N-1).
        Creates frequency grid via FFT with the same number of points.

        The relationship between grids:
            - Time grid: N points with spacing dt = T_max / (N-1)
            - Frequency grid: N points from fftfreq
            - Fourier duality: d_omega * dt = 2π / N

        Stores:
            self.omega_grid: Angular frequency array (centered, sorted)
            self.energy_cutoff: Maximum omega value (Nyquist frequency)
            self.d_omega: Omega spacing
        """
        # Use same number of points as time grid
        n_points = self.ntpoints

        # Get frequency bins from FFT using the actual time grid spacing
        # np.fft.fftfreq gives frequencies f in cycles per unit time
        freq = np.fft.fftfreq(n_points, d=self.delta_t)

        # Convert to angular frequency: ω = 2π*f
        omega = 2 * np.pi * freq

        # Shift to center around 0 (zero frequency in middle)
        self.omega_grid = np.fft.fftshift(omega)

        # Energy cutoff is the maximum absolute omega value (Nyquist frequency)
        self.energy_cutoff = np.max(np.abs(self.omega_grid))

        # Store omega spacing and verify Fourier duality
        self.d_omega = self.omega_grid[1] - self.omega_grid[0]
        expected_product = 2 * np.pi / n_points
        actual_product = self.d_omega * self.delta_t
        if not np.allclose(actual_product, expected_product, rtol=1e-10):
            print(f"WARNING: Fourier duality check failed!")
            print(f"  Expected: {expected_product:.10f}, Actual: {actual_product:.10f}")
            print(f"  d_omega = {self.d_omega:.10e}")
            print(f"  delta_t = {self.delta_t:.10e}")
            print(f"  n_points = {n_points}")
            print(f"  Omega grid check: uniform spacing = {np.allclose(np.diff(self.omega_grid), self.d_omega)}")


    # ========== Initial State Generation ==========

    def generate_initial_state(self, Q=0.0, gr0=None):
        """
        Generate initial state from equilibrium for t,t' < 0.

        Steps:
        1. Create EquilibriumSolver object
        2. Compute equilibrium gr and gk in frequency domain
        3. Fourier transform to two-time representation (only t,t' < 0)
        4. Construct and return StateObject with equilibrium data

        Args:
            Q: Phase gradient (default 0)
            gr0: Initial guess for equilibrium gr (optional)

        Returns:
            StateObject with equilibrium data for t < 0

        Calls:
            - EquilibriumSolver.compute_equilibrium_gr()
            - EquilibriumSolver.fourier_transform_to_two_time()
        """
        # Create equilibrium solver with current grid parameters
        # Need to add omega_grid to grid_parameters for equilibrium solver
        grid_params_with_omega = self.grid_parameters.copy()
        grid_params_with_omega['omega_grid'] = self.omega_grid
        grid_params_with_omega['energy_cutoff'] = self.energy_cutoff

        equilibrium_solver = EquilibriumSolver(
            grid_params_with_omega,
            self.system_parameters,
            self.optimization_parameters,
            self.sigma_scatterings
        )

        # Compute equilibrium Green's functions in frequency domain
        gr_eq, gk_eq = equilibrium_solver.compute_equilibrium_gr(
            temperature=self.temperature,
            Q=Q,
            gr0=gr0,
            compute_gk=True
        )

        # Transform to two-time representation (returns only t,t' < 0)
        gr_two_time, gk_two_time, gr_tau, gk_tau = equilibrium_solver.fourier_transform_to_two_time(gr_eq, gk_eq)

        # Get BCS coupling constant for StateObject
        bcs_coupling = self._get_BCS_coupling()

        # Create and return StateObject
        initial_state = StateObject(
            gr=gr_two_time,
            gk=gk_two_time,
            bcs_coupling_constant=bcs_coupling,
            grid_params=self.grid_parameters
        )

        return initial_state, gr_tau, gk_tau

    # ========== Thermal Distributions ==========

    def get_thermal_occupation(self, temperature):
        """
        Generate thermal occupation function as a two-time object.

        Uses analytic form: f(τ) = -i T / sinh(π τ T)
        Returns f(t,t') = f(t-t') as a NambuKeldyshTensor with two time axes.

        Args:
            temperature: Temperature in energy units

        Returns:
            None (stores result in self.thermal_dist as (2, 2, ntpoints, ntpoints) tensor)
        """
        # Use time grid from -T_max to 0 (already defined in _generate_time_grid)
        time_grid = self.time_grid

        # Create meshgrid for all time pairs (t_i, t_j)
        t_i, t_j = np.meshgrid(time_grid, time_grid, indexing='ij')

        # Compute tau = t_i - t_j for all pairs
        tau_matrix = t_i - t_j

        # Initialize two-time thermal distribution
        f_two_time = np.zeros((self.ntpoints, self.ntpoints), dtype=complex)

        # Create mask to avoid division by zero where τ = 0 (diagonal and near-diagonal)
        mask = (np.abs(tau_matrix) > 1e-10)

        # Compute f(τ) = -i T / sinh(π τ T) for all non-zero tau values
        f_two_time[mask] = -1j * temperature / np.sinh(np.pi * tau_matrix[mask] * temperature)

        # Diagonal (τ=0) remains zero (already initialized to zero)

        # Store as NambuKeldyshTensor with two time axes (identity in Nambu space)
        self.thermal_dist = NambuKeldyshTensor(f_two_time, pauli_channel=0)

    def get_thermal_integral(self, temperature):
        """
        Compute cumulative integral of thermal distribution on finite time grid.

        For convolutions on finite grid [-T_max, 0], computes:
        F(t, t') = ∫_{-T_max - t'}^{t - t'} f(τ') dτ'
                 = F_full(t - t') - F_full(-T_max - t')

        where F_full(τ) = -i/π · ln(tanh(πτT/2)) is the infinite-domain integral
        and f(τ) = -i T / sinh(π τ T) is the thermal distribution.

        Args:
            temperature: Temperature in energy units

        Stores:
            self.thermal_integral: NambuKeldyshTensor of shape (2, 2, ntpoints, ntpoints)
                                  Identity in Nambu space
        """
        # Create meshgrid for all time pairs (t_i, t_j)
        t_i, t_j = np.meshgrid(self.time_grid, self.time_grid, indexing='ij')

        # Compute upper bound: tau = t_i - t_j
        tau_upper = t_i - t_j

        # Compute lower bound: -T_max - t_j
        tau_lower = -self.tmax - t_j

        # Helper function to compute F(τ) = -i/π · ln(tanh(πτT/2))
        def compute_F_full(tau_vals):
            """Compute analytical thermal integral from -∞ to τ (no constant)."""
            result = np.zeros_like(tau_vals, dtype=complex)

            # Mask to avoid singularity at τ = 0
            mask = (np.abs(tau_vals) > 1e-10)

            # Compute where τ ≠ 0
            x = np.pi * tau_vals[mask] * temperature / 2.0
            tanh_x = np.tanh(x)
            ln_tanh = np.log(tanh_x + 0j)
            result[mask] = -1j / np.pi * ln_tanh  # No +1 constant

            # At τ = 0, set to 0 (principal value)
            return result

        # Compute finite-domain integral: F(upper) - F(lower) + 1
        F_upper = compute_F_full(tau_upper)
        F_lower = compute_F_full(tau_lower)
        F_two_time = F_upper - F_lower #+ 1.0

        # Store as NambuKeldyshTensor (identity in Nambu space)
        self.thermal_integral = NambuKeldyshTensor(F_two_time, pauli_channel=0)


    # ========== Real-Time Evolution ==========

    def _compute_new_gr_row(self, state, external_field=None):
        """
        Evolve retarded Green's function gr by one timestep.

        Computes g^R(t_{time_index}, t_j) for all j < time_index using the
        discretized Usadel equation (without A(t) terms).

        Args:
            state: StateObject with current gr data
            time_index: New time index to compute (i+1 in equations)
            external_field: Optional external perturbation (not used)

        Returns:
            new_gr_row: List of NambuKeldyshTensor objects for g^R(t_new, t_j)

        Called by:
            - _evolve_state_by_one_timestep()
        """
        # Current time index (i in the equations)

        # Extract gap history
        gap_history = state.get_gap_history()
        gap_history = np.ones(np.size(gap_history)) * 1.5232319831848145
        #! overwrite gap update

        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) +  NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)
        # Create τ_3 Pauli matrix as NambuKeldyshTensor (identity in time)
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        #* Compute individual terms in the commutator
        
        gr_last_row = state.gr[-1:,:]  # Shape (2, 2, 1, Nt)

        gr_diagonal_new = -gap_tensor[-1] #- tau3 * gr_last_row[-1, -1] * tau3

        # evaluated at t
        left_matrix_evolution = (1j * tau3 - 1j *self.delta_t * gap_tensor[-1] + 1j * self.eta * self.delta_t * tau3) * expansion_tensor
        # evaluated at t' which will be varied anyway
        right_matrix_evolution = (-1j * tau3 - 1j * self.eta * self.delta_t * tau3) * expansion_tensor + 1j * self.delta_t * gap_tensor 

        rhs_vector_evolution = 1j * tau3 * gr_last_row #* missing the new updated term, but that will easily be added! 

        left_matrix_normalization = (tau3 + self.delta_t * gr_diagonal_new) * expansion_tensor
        right_matrix_normalization = tau3 * expansion_tensor
 
        rhs_vector_normalization = 0 * rhs_vector_evolution #* needs to be recomputed every timestep

        gr_new = self.gr_update_rule(left_matrix_evolution, left_matrix_normalization, right_matrix_evolution, right_matrix_normalization, rhs_vector_evolution, rhs_vector_normalization, gr_diagonal_new, state.gr)

        return gr_new, gr_diagonal_new 

    #* Rules for evaluating convolutions
    #* integral from t' to t, starts the sum from t' + dt until t 
    #* The indicies are described relative to old matrices -- new index is t + delta_t basically, but t' is same as before, of course!


    def gr_update_rule(self, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_matrix_1, rhs_matrix_2, diagonal_entry, full_g_matrix):
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        matrix_row_1 = (tau1 * left_matrix_1 + right_matrix_1 * tau1).matrix_to_vector()
        matrix_row_2 = (tau2 * left_matrix_1 + right_matrix_1 * tau2).matrix_to_vector()
        matrix_row_3 = (tau3 * left_matrix_2 + right_matrix_2 * tau3).matrix_to_vector()
        matrix_row_4 = (tau0 * left_matrix_2 + right_matrix_2 * tau0).matrix_to_vector()

        vector_row_1 = (rhs_matrix_1.trace(1)/2)[0]
        vector_row_2 = (rhs_matrix_1.trace(2)/2)[0]
        vector_row_3 = (rhs_matrix_2.trace(3)/2)[0]
        vector_row_4 = (rhs_matrix_2.trace(0)/2)[0]

        solution_tensor = diagonal_entry * NambuKeldyshTensor([1.0], pauli_channel=0)

        # Backward sweep in time
        for time in range(self.ntpoints-1, -1, -1):
            # Compute normalization convolution term
            if time == self.ntpoints - 1:
                norm_convolution = np.array([0, 0, 0, 0])
            else:
                # Convolution: solution_tensor (excluding diagonal) @ full_g_matrix[time+1:, time]
                #* summing over inner index. t' goes until t-dt, since diagonal is removed and it starts at t' + dt due to our summation rule
                #* full_g_matrix is evaluated at time and it starts at final t'. First index goes from t'+1 onwards as possible
                #* solution tensor is initiall the diagonal entry and the t' goes until all but t
                #* For first term the convolution is not used since, first index is in new row so new solution has to be used
                #* for second term, there is a solution corresponding to not final time
                norm_convolution = -(solution_tensor[:-1] @ full_g_matrix[time+1:, time]).matrix_to_vector() * self.delta_t
            total_matrix = np.array([matrix_row_1[:, time],matrix_row_2[:, time],matrix_row_3[:, time],matrix_row_4[:, time]])

            diagonal_components = solution_tensor[0].matrix_to_vector()

            total_vector = np.array([vector_row_1[time], vector_row_2[time],vector_row_3[time],vector_row_4[time]]) + np.array([diagonal_components[2], -diagonal_components[1], norm_convolution[3], norm_convolution[0] ])

            # Solve linear system for [g1, g2, g3, g0]
            g_components = np.linalg.solve(total_matrix, total_vector)

            # Append to solution (prepends to data)
            solution_tensor.append(g_components)

        return solution_tensor[:-1]

    def _compute_new_gk_row(self, state, external_field=None):
        
        gr = state.gr
        ga = state._r2a()
        #- 1j * tau3 * (term1 + term2 + term3 + term4 + term5 + term6 + term7) * self.delta_t
        # Current time index (i in the equations)

        # Extract gap history
        gap_history = state.get_gap_history()
        #! overwrite gap update
        gap_history = np.ones(np.size(gap_history)) * 1.5232319831848145
        
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) +  NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)
        # Create τ_3 Pauli matrix as NambuKeldyshTensor (identity in time)
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        #* Compute individual terms in the commutator
        gk_last_row = state.gk[-1:,:]  # Shape (2, 2, 1, Nt)

        gk_diagonal_new =  gk_last_row[-1,-1] 

        gr_diagonal_new = gr[-1,-1]
        ga_diagonal_new = ga[-1,-1]

        new_gr_row = gr[-1:,:]

        left_matrix_evolution = (1j * tau3 - 1j *self.delta_t * gap_tensor[-1] + 1j * self.eta * self.delta_t * tau3) * expansion_tensor
        right_matrix_evolution = (1j * tau3 + 1j * self.eta * self.delta_t * tau3) * expansion_tensor + 1j * self.delta_t * gap_tensor 
        
        #* removed last term since we now consider the shifted g^k equation
        rhs_vector_evolution = 1j * tau3 * gk_last_row -  2j * self.delta_t**2 * self.eta * (tau3 * (self.thermal_dist[-1:,:] @ (ga.shift(1,axis = 1))) - (gr[-1:,:] @ self.thermal_dist.shift(1, axis = 1)) * tau3) #+ 4j * self.eta * self.thermal_dist[-1:,:].shift(1, axis = 1) * self.delta_t
        rhs_vector_evolution += -2 * (-1j * self.delta_t * gap_tensor[-1] * tau3 * self.thermal_dist[-1:,:].shift(1, axis = 1) + 1j * self.delta_t * tau3 * self.thermal_dist[-1:,:].shift(1, axis = 1) * gap_tensor)
        
        #print('f is', self.thermal_dist[-1:,:].shift(1, axis = 1).trace(0))
        left_matrix_normalization = (tau3 + self.delta_t * gr_diagonal_new) * expansion_tensor 
        right_matrix_normalization = (-tau3 + self.delta_t * ga_diagonal_new) * expansion_tensor 
        
        rhs_vector_normalization = -2 *self.delta_t * (tau3 * (self.thermal_dist[-1:,:] @ (ga.shift(1,axis = 1))) + (gr[-1:,:] @ self.thermal_dist.shift(1, axis = 1)) * tau3)

        gk_new = self.gk_update_rule(left_matrix_evolution, left_matrix_normalization, right_matrix_evolution, right_matrix_normalization, rhs_vector_evolution, rhs_vector_normalization, new_gr_row, full_ga_matrix=ga, old_gk_matrix=state.gk)
        #print(gk_new.data.shape)
        #gk_new += 2 * tau3 * self.thermal_dist[-1,:].shift(1, axis = 0)
        #* removed last term since we now consider the  shifted g^k equation
        rhs_vector_evolution_diagonal =  1j * (gk_new.dagger())[-1:] * tau3  - 2j * self.delta_t**2 * self.eta * (tau3 * (self.thermal_dist[-1:,:] @ (ga[:,-1])) - (gr[-1:,:] @ self.thermal_dist[:,-1]) * tau3)# + 4j * self.eta * self.thermal_dist[-1] * self.delta_t  
        rhs_vector_evolution_diagonal += -2 * (-1j * self.delta_t * gap_tensor[-1] * tau3 * self.thermal_dist[-1:,-1] + 1j * self.delta_t * tau3 * self.thermal_dist[-1:,-1] * gap_tensor[-1])
        
        rhs_vector_normalization_diagonal = -2 * self.delta_t * (tau3 * (self.thermal_dist[-1:,:] @ (ga[:,-1])) + (gr[-1:,:] @ self.thermal_dist[:,-1]) * tau3)

        gk_diagonal_new = self.gk_diagonal_update_rule(left_matrix_evolution, left_matrix_normalization, right_matrix_evolution, right_matrix_normalization,rhs_vector_evolution_diagonal, rhs_vector_normalization_diagonal, new_gr_row, full_ga_matrix=ga, old_gk_matrix=state.gk, solution_tensor= gk_new)
        
        return gk_new, gk_diagonal_new

    def gk_update_rule(self, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_matrix_1, rhs_matrix_2, last_gr_row, full_ga_matrix, old_gk_matrix):
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        matrix_row_1 = (tau0 * left_matrix_1 + right_matrix_1 * tau0).matrix_to_vector()
        matrix_row_2 = (tau3 * left_matrix_1 + right_matrix_1 * tau3).matrix_to_vector()
        matrix_row_3 = (tau1 * left_matrix_2 + right_matrix_2 * tau1).matrix_to_vector()
        matrix_row_4 = (tau2 * left_matrix_2 + right_matrix_2 * tau2).matrix_to_vector()


        #print(tau1 * left_matrix_2 + right_matrix_2 * tau1)

        vector_row_1 = (rhs_matrix_1.trace(0)/2)[0]
        vector_row_2 = (rhs_matrix_1.trace(3)/2)[0]
        vector_row_3 = (rhs_matrix_2.trace(1)/2)[0]
        vector_row_4 = (rhs_matrix_2.trace(2)/2)[0]

        solution_tensor = NambuKeldyshTensor([0.0j], pauli_channel=0) # old_gk_matrix[-1, 0:1]

        for time in range(1, self.ntpoints):
            #* the start of the sum doesn't matter since its -infty somehow, so it should be zero anyway -- 1/2 may be important? 
            #* \sum_{t''} gr(t,t'') gk(t'',t') 0 --> t - dy since removed diagonals
            #* this means gr is new matrix evaluated at (t,t'') t'' sums all the elements except the last one there
            #* old_gk_matrix, no worries there we evaluate the gk(t'',t') at all entries up to t-dt, so we can just use the old matrix without a shift
            #* \sum{t''} gk(t,t'') ga(t'',t')  0 --> t' - dt since removed diagonals
            #* the gk matrix is evaluated in this row itself up to t'-dt and we are currently computing for t'
            #* 1. the first solution is the solution for t'' = -infty and should be set to 0 
            #* 2. the overall sum must go from 1 to self.ntpoints to generate a mesh of n entries with the first element fixed by boundary
            #* 3. the sum is present instantly since we have the first element alread, i.e. t > 0 always 
            #* 4. the convolution should go from -infty and should contain time-1 terms as expected 

            norm_convolution = -(last_gr_row[0,:-1] @ old_gk_matrix[1:,time]).matrix_to_vector() * self.delta_t #+ (last_gr_row[1:] @ full_gk_matrix[1:,0]).matrix_to_vector() * self.delta_t
            if time > 1:
                norm_convolution += -(solution_tensor[1:] @ full_ga_matrix[0:time-1, time-1]).matrix_to_vector() * self.delta_t #+ (solution_tensor[1:] @ full_ga_matrix[:time, 0]).matrix_to_vector() * self.delta_t 

            total_matrix = np.array([matrix_row_1[:, time],matrix_row_2[:, time],matrix_row_3[:, time],matrix_row_4[:, time]])
            diagonal_components = solution_tensor[-1].matrix_to_vector()
            total_vector = np.array([vector_row_1[time], vector_row_2[time],vector_row_3[time],vector_row_4[time]] ) + np.array([1j * diagonal_components[3], 1j * diagonal_components[0], norm_convolution[1], norm_convolution[2]])
            #if time == self.ntpoints -2:
                #print(total_matrix)
                #print(total_vector)

            g_components = np.linalg.solve(total_matrix, total_vector )

            # Append to solution (prepends to data)
            solution_tensor.append_right(g_components)

        return solution_tensor

    def gk_diagonal_update_rule(self, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_matrix_1, rhs_matrix_2,  last_gr_row, full_ga_matrix, old_gk_matrix, solution_tensor):
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        matrix_row_1 = (tau0 * left_matrix_1 + right_matrix_1 * tau0).matrix_to_vector()
        matrix_row_2 = (tau3 * left_matrix_1 + right_matrix_1 * tau3).matrix_to_vector()
        matrix_row_3 = (tau1 * left_matrix_2 + right_matrix_2 * tau1).matrix_to_vector()
        matrix_row_4 = (tau2 * left_matrix_2 + right_matrix_2 * tau2).matrix_to_vector()

        vector_row_1 = (rhs_matrix_1.trace(0)/2)[0]
        vector_row_2 = (rhs_matrix_1.trace(3)/2)[0]
        vector_row_3 = (rhs_matrix_2.trace(1)/2)[0]
        vector_row_4 = (rhs_matrix_2.trace(2)/2)[0]

        norm_convolution = -(last_gr_row[0,:-1] @ (solution_tensor[1:].involution())).matrix_to_vector() * self.delta_t -(solution_tensor[1:] @ full_ga_matrix[0:-1, -1]).matrix_to_vector() * self.delta_t
        total_matrix = np.array([matrix_row_1[:, -1],matrix_row_2[:, -1],matrix_row_3[:, -1],matrix_row_4[:, -1]])
        diagonal_components = solution_tensor[-1].matrix_to_vector()
        total_vector = np.array([vector_row_1, vector_row_2,vector_row_3,vector_row_4] ) +  np.array([1j * diagonal_components[3], 1j * diagonal_components[0], norm_convolution[1], norm_convolution[2]])

        g_components = np.linalg.solve(total_matrix, total_vector)

        return NambuKeldyshTensor.vector_to_matrix(g_components)[-1]

    def _evolve_state_by_one_timestep(self, state, time_index, external_field=None):
        """
        Evolve state by one timestep, generating new entries.

        Steps:
        1. Initialize thermal distribution if needed
        2. Call _compute_new_gr_row() to get new gr row and diagonal
        3. Call _compute_new_gk_row() to get new gk row and diagonal
        4. Compute gap from new gk diagonal element
        5. Update state using state.update_state_object()
        6. Return gap and current at new time

        Args:
            state: StateObject with current data
            time_index: Current time index (not used, kept for compatibility)
            external_field: Optional external perturbation

        Returns:
            gap_new: Gap value at new time t
            current_new: Current at new time t (zero for now)

        Calls:
            - _compute_new_gr_row(state, external_field)
            - _compute_new_gk_row(state, external_field)
            - state.update_state_object(new_gr_row, new_gr_diag, new_gk_row, new_gk_diag)
        """
        # Initialize thermal distribution if not already done
        if not hasattr(self, 'thermal_dist'):
            self.get_thermal_occupation(self.temperature)

        # Compute new gr and gk rows and diagonals
        new_gr_row, new_gr_diag = self._compute_new_gr_row(state, external_field)
        
        #* update state and then pass it to gk
        state.update_state_gr(new_gr_row, new_gr_diag)

        new_gk_row, new_gk_diag = self._compute_new_gk_row(state, external_field)

        # Update g^R first, then g^K
        state.update_state_gk(new_gk_row, new_gk_diag)

        # Compute gap at new time using state method
        gap_history = state.get_gap_history()
        gap_new = gap_history[-1]  # Extract gap at newest time

        # Current is zero for now (Stage 2 of project)
        current_new = 0.0

        return gap_new, current_new


    def real_time_evolution(self, initial_state, num_timesteps, external_field=None):
        """
        Main real-time evolution loop.

        Evolves state forward in time, extracting observables at each step.

        Steps:
        1. Initialize observables arrays
        2. For each timestep:
            a. Call _evolve_state_by_one_timestep()
            b. Store returned gap and current values
        3. Return evolved state and observable time series

        Args:
            initial_state: StateObject with equilibrium initial conditions
            num_timesteps: Number of time steps to evolve
            external_field: Optional time-dependent external field

        Returns:
            state: Final evolved StateObject
            gaps: Array of gap values at each timestep
            currents: Array of current values at each timestep

        Calls:
            - _evolve_state_by_one_timestep(state, time_index, external_field)
        """
        # Initialize arrays to track observables
        gaps = np.zeros(num_timesteps, dtype=complex)
        currents = np.zeros(num_timesteps, dtype=complex)

        # Start with initial state
        state = initial_state
        
        # Evolve over time with progress bar
        for time_index in tqdm(range(num_timesteps), desc="Real-time evolution"):
            # Evolve by one timestep and get observables
            gap_new, current_new = self._evolve_state_by_one_timestep(
                state, time_index, external_field
            )
            # Store observables
            gaps[time_index] = gap_new
            currents[time_index] = current_new

        return state, gaps, currents

