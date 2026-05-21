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
        #* the effective coupling passed to the time state has to be rescaled in the instananeous case!
        bcs_coupling = self._get_BCS_coupling() #* (1 + 0 * self._get_BCS_coupling()/4/np.pi * np.log(self.critical_temperature/self.temperature))

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
            mask = (np.abs(tau_vals) > 1e-6)

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
        F_two_time = F_upper - F_lower + 1.0

        # Set F(0) to BCS regularization: 1/λ + ln(T_c/T)
        bcs_coupling = self._get_BCS_coupling()

        F_zero_bcs = 2 * 1j * (1 + bcs_coupling/(2 * np.pi) * np.log(self.critical_temperature / temperature))

        # Replace diagonal (τ=0) with BCS value
        diagonal_mask = (np.abs(tau_upper) < 1e-6)
        F_two_time[diagonal_mask] = F_zero_bcs

        # Store as NambuKeldyshTensor (identity in Nambu space)
        self.thermal_integral = NambuKeldyshTensor(F_two_time, pauli_channel=0)

    # ========== Real-Time Evolution ==========

    def _compute_new_gr_row(self, state, A_history=None):
        """
        Evolve retarded Green's function gr by one timestep.

        Computes g^R(t_{time_index}, t_j) for all j < time_index using the
        discretized Usadel equation (without A(t) terms).

        Args:
            state: StateObject with current gr data
            time_index: New time index to compute (i+1 in equations)
            A_history: Optional external perturbation (not used)

        Returns:
            new_gr_row: List of NambuKeldyshTensor objects for g^R(t_new, t_j)

        Called by:
            - _evolve_state_by_one_timestep()
        """
        #========== Setup: Extract gap and define Pauli matrices ==========

        #Extract gap history Δ(t) from current state via gap equation
        gap_history = state.get_gap_history()
        #! overwrite gap 
        gap_history = np.ones(np.size(gap_history)) * 1.4524034261703491
        #Build gap tensor as Δ̂ = Re(Δ)τ₂ + Im(Δ)τ₁ (off-diagonal Nambu structure)
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) +  NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        #========== Phase 0: Vector Potential Initialization ==========
        #Initialize A_history if not provided (no vector potential case)
        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)

        #Convert to NambuKeldyshTensor (identity in Nambu space, pauli_channel=0)
        A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)

        #Extract A(t) at new time step (latest time point)
        A_t = A_tensor[-1]  # Shape: (2, 2, 1)

        #Compute A²(t) as scalar for use in operators
        A2_t = A_history[-1]**2

        #Define Pauli matrices as NambuKeldyshTensor objects (identity in time dimension)
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        #Expansion tensor for broadcasting Pauli matrices to full time grid
        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        #========== Discrete g^R Evolution Equation (tex file Eq. 62-64) ==========
        #Structure: L̂_R · ĝ'^R + ĝ'^R · R̂_R + [bilinear terms] = Ŝ_R
        #Also: Normalization constraint from Eq. 95

        #Extract g^R(t-δt, :) for time derivative term: ∂_{t₁}ĝ^R → [ĝ^R(t,t') - ĝ^R(t-δt,t')]/δt
        gr_last_row = state.gr[-1:,:]  # Shape (2, 2, 1, Nt)

        #Boundary condition: ĝ'^R(t,t) = -Δ̂(t) (tex Eq. 60)
        gr_diagonal_new = -gap_tensor[-1]

        #Left operator L̂_R = iτ̂₃ - iδt·Δ̂(t) + iη·δt·τ̂₃ + ie²D δt [A²(t)τ₃ - A(t)A(t')τ₃] - ie²D δt² A²(t)τ₃Δτ₃
        left_matrix_evolution = (
            (1j * tau3
            - 1j * self.delta_t * gap_tensor[-1]
            + 1j * self.eta * self.delta_t * tau3
            # Electromagnetic terms (e²D=1): +A²(t)τ₃ - A(t)A(t')τ₃ - A²(t)τ₃Δτ₃
            + 1j * self.delta_t * A2_t * tau3  # +iδt·A²(t)·τ₃
            - 1j * self.delta_t**2 * A2_t * tau3 * gap_tensor[-1] * tau3  # -iδt²·A²(t)·τ₃·Δ(t)·τ₃
            ) * expansion_tensor
            - 1j * self.delta_t * A_t * A_tensor * tau3  # -iδt·A(t)A(t')·τ₃ (CORRECTED SIGN)
        ) 
        #Right operator R̂_R = -iτ̂₃ + iδt·Δ̂(t') - iη·δt·τ̂₃ - [A terms] (tex Eq. 72)
        right_matrix_evolution = (
            (-1j * tau3 - 1j * self.eta * self.delta_t * tau3) * expansion_tensor
            + 1j * self.delta_t * gap_tensor
            # Phase 2: Vector potential terms (tex line 72, e²D=1)
            - 1j * self.delta_t * (A_tensor * A_tensor) * tau3  # A²(t') term
            - 1j * self.delta_t * A_t * A_tensor * tau3  # A(t)A(t') term
        )

        #Source term Ŝ_R = iτ̂₃·ĝ^R(t-δt,t') - iĝ^R(t,t'+δt)·τ̂₃ (tex Eq. 79)
        #The second term (forward-shifted in t') is handled implicitly in gr_update_rule's backward sweep
        rhs_vector_evolution = 1j * tau3 * gr_last_row

        # Phase 3: Sandwich terms L̂_R^(2) and R̂_R^(2) (tex Eq. 73)
        # Bilinear term: L̂_R^(2) · g'^R(t,t') · R̂_R^(2)
        left_sandwich = 1j * self.delta_t**2 * A_t * A_tensor * gap_tensor[-1] * tau3  # iδt²·A(t)A(t')·Δ(t)τ₃
        right_sandwich = tau3 * expansion_tensor  # τ₃
        g_sandwich_list = [(left_sandwich, right_sandwich)]

        #Normalization constraint operators (tex Eq. 95):
        #Left: τ̂₃ + δt·Δ̂(t) (from extracting t''=t term using boundary condition)
        left_matrix_normalization = (tau3 + self.delta_t * gr_diagonal_new) * expansion_tensor
        #Right: τ̂₃
        right_matrix_normalization = tau3 * expansion_tensor

        #Normalization RHS: convolution term Σ ĝ^R ĝ^R computed inside gr_update_rule
        rhs_vector_normalization = 0 * rhs_vector_evolution

        #Solve coupled evolution + normalization equations via backward sweep in t'
        # Phase 5: Pass A_history for source convolutions
        gr_new = self.gr_update_rule(left_matrix_evolution, left_matrix_normalization, right_matrix_evolution, right_matrix_normalization, rhs_vector_evolution, rhs_vector_normalization, gr_diagonal_new, state.gr, g_sandwich_list = g_sandwich_list, A_history = A_history)

        return gr_new, gr_diagonal_new 

    #* Rules for evaluating convolutions
    #* integral from t' to t, starts the sum from t' + dt until t 
    #* The indicies are described relative to old matrices -- new index is t + delta_t basically, but t' is same as before, of course!


    def gr_update_rule(self, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_matrix_1, rhs_matrix_2, diagonal_entry, full_g_matrix, g_sandwich_list = [], A_history = None):
        """
        Solve for g^R(t, t') at all t' using backward sweep (tex Eq. 62 + 95).

        The backward sweep from t'=0 down to t'=-T_max implicitly handles the forward
        derivative term -iĝ^R(t,t'+δt)τ̂₃ because each iteration uses the solution
        from the previous iteration (which computed the point at t'+δt).

        Pauli trace projection reduces the 2×2 matrix equation to 4 scalar equations.
        """
        #Define Pauli matrices for trace projection
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        #Build 4×4 system by projecting onto Pauli basis (exploits redundancy)
        #Evolution equation rows: τ₁, τ₂ traces
        matrix_row_1 = (tau1 * left_matrix_1 + right_matrix_1 * tau1).matrix_to_vector()
        matrix_row_2 = (tau2 * left_matrix_1 + right_matrix_1 * tau2).matrix_to_vector()
        #Normalization constraint rows: τ₃, τ₀ traces
        matrix_row_3 = (tau3 * left_matrix_2 + right_matrix_2 * tau3).matrix_to_vector()
        matrix_row_4 = (tau0 * left_matrix_2 + right_matrix_2 * tau0).matrix_to_vector()

        for terms in g_sandwich_list:
            left_term = terms[0]
            right_term = terms[1]
            matrix_row_1 += (right_term * tau1 * left_term).matrix_to_vector()
            matrix_row_2 += (right_term * tau2 * left_term).matrix_to_vector()

        #Extract RHS vectors for each Pauli component
        vector_row_1 = (rhs_matrix_1.trace(1)/2)[0]  # τ₁
        vector_row_2 = (rhs_matrix_1.trace(2)/2)[0]  # τ₂
        vector_row_3 = (rhs_matrix_2.trace(3)/2)[0]  # τ₃
        vector_row_4 = (rhs_matrix_2.trace(0)/2)[0]  # τ₀

        #Initialize with diagonal: g^R(t,t) = -Δ̂(t)
        solution_tensor = diagonal_entry * NambuKeldyshTensor([1.0], pauli_channel=0)

        #========== Backward sweep: t'=0 → -T_max ==========
        for time in range(self.ntpoints-1, -1, -1):
            #Normalization convolution: Σ_{t''=t'+δt}^{t-δt} ĝ^R(t,t'') ĝ^R(t'',t')
            if time == self.ntpoints - 1:
                #First iteration (t'=0): no intermediate points, convolution = 0
                norm_convolution = np.array([0, 0, 0, 0])
            else:
                #solution_tensor[:-1]: New ĝ^R(t,t'') for t'' from current t' to t-δt (excl. diagonal)
                #full_g_matrix[time+1:, time]: Old ĝ^R(t'',t') for t'' from t'+δt to latest
                norm_convolution = -(solution_tensor[:-1] @ full_g_matrix[time+1:, time]).matrix_to_vector() * self.delta_t

            # Phase 4: Source convolutions with vector potential (tex lines 80-81)
            # Compute only if time < ntpoints-1 and A_history is not all zeros
            if time < self.ntpoints - 1 and A_history is not None and not np.allclose(A_history, 0):
                # Need tau3 and A_tensor for convolutions
                tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
                A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)

                # Build weighted_gr = g^R(t,t'')·A(t'')·τ̂₃
                weighted_gr = solution_tensor[:-1] * A_tensor[time+1:] * tau3

                # First convolution: A(t)τ̂₃·[weighted_gr @ g^R(t'',t')]
                first_conv = A_tensor[-1] * tau3 * (weighted_gr @ full_g_matrix[time+1:, time])

                # Second convolution: [weighted_gr @ g^R(t'',t')]·A(t')τ̂₃
                second_conv = (weighted_gr @ full_g_matrix[time+1:, time]) * A_tensor[time] * tau3

                # Add to RHS: -iδt²[first - second]
                source_A_correction = -1j * self.delta_t**2 * (first_conv - second_conv)

                # Extract τ₁ and τ₂ components for evolution equation
                source_A_vector = np.array([
                    source_A_correction.trace(1)/ 2,# [0, time]   # τ₁ component
                    source_A_correction.trace(2) / 2 #[0, time]   # τ₂ component
                ])
            else:
                source_A_vector = np.array([0, 0])

            #Build 4×4 linear system at current t' point
            total_matrix = np.array([matrix_row_1[:, time],matrix_row_2[:, time],matrix_row_3[:, time],matrix_row_4[:, time]])

            #Diagonal coupling: {τ̂₃, ĝ^R(t,t)} appears in evolution equation
            diagonal_components = solution_tensor[0].matrix_to_vector()

            #Assemble RHS: [evolution RHS] + [diagonal coupling] + [normalization convolution] + [A source convolution]
            #Diagonal coupling adds [g₂_diag, -g₁_diag, 0, 0] from anticommutator
            #Normalization adds [0, 0, g₃_conv, g₀_conv] from convolution term
            #A source convolution adds [A_τ₁, A_τ₂, 0, 0] from electromagnetic terms
            total_vector = np.array([
                vector_row_1[time] + source_A_vector[0],  # τ₁ with A correction
                vector_row_2[time] + source_A_vector[1],  # τ₂ with A correction
                vector_row_3[time],  # τ₃ (normalization, no A correction)
                vector_row_4[time]   # τ₀ (normalization, no A correction)
            ]) + np.array([diagonal_components[2], -diagonal_components[1], norm_convolution[3], norm_convolution[0]])

            #Solve for Pauli components [g₁, g₂, g₃, g₀] at this t'
            g_components = np.linalg.solve(total_matrix, total_vector)

            #Prepend to solution (builds backward in time)
            solution_tensor.append(g_components)

        #Remove diagonal element (only needed for boundary condition)
        return solution_tensor[:-1]

    def _compute_new_gk_row(self, state, A_history=None):
        """Evolve g^K by one timestep (tex Eq. 142-194). Includes thermal collision integrals.

        Electromagnetic coupling signs corrected to match Chapter 9 analytics (lines 166-175, 180-193).
        R_K operator and bilinear sandwich terms updated based on boundary extraction derivations.
        """
        #Get Green's functions (g^A computed via involution from g^R)
        gr = state.gr
        ga = state._r2a()

        #Extract gap and build Pauli matrices
        gap_history = state.get_gap_history()
        #! overwrite gap 
        gap_history = np.ones(np.size(gap_history)) * 1.4524034261703491
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) +  NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)

        #Convert to NambuKeldyshTensor (identity in Nambu space, pauli_channel=0)
        A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)

        #Extract A(t) at new time step (latest time point)
        A_t = A_tensor[-1]  # Shape: (2, 2, 1)

        #Compute A²(t) as scalar for use in operators
        A2_t = A_history[-1]**2


        #Extract g^K(t-δt, :) for time derivative: ∂_{t₁}ĝ^K → [ĝ^K(t,t') - ĝ^K(t-δt,t')]/δt
        gk_last_row = state.gk[-1:,:]  # Shape (2, 2, 1, Nt)

        #Diagonal values for boundary conditions in normalization
        gk_diagonal_new =  gk_last_row[-1,-1]  # Placeholder (computed after off-diagonal)
        gr_diagonal_new = gr[-1,-1]
        ga_diagonal_new = ga[-1,-1]
        new_gr_row = gr[-1:,:]

        #Left operator L̂_K = iτ̂₃ - iδt·Δ̂(t) + iη·δt·τ̂₃ + [A terms] (tex Eq. 150-151)
        left_matrix_evolution = (
            (1j * tau3
            - 1j * self.delta_t * gap_tensor[-1]
            + 1j * self.eta * self.delta_t * tau3 + 1j * self.delta_t * A2_t * tau3  - 1j * self.delta_t**2 * A2_t * tau3 * gap_tensor[-1] * tau3   ) * expansion_tensor
            # Phase 1: Vector potential terms (tex lines 150-151, e²D=1)
            # A²(t) term
            - 1j * self.delta_t * A_t * A_tensor * tau3  # -A(t)A(t') term (minus sign!)
            # Higher-order A²·Δ term
        ) 

        #Right operator R̂_K = iτ̂₃ + iδt·Δ̂(t') + iη·δt·τ̂₃ + [A terms] (tex Eq. 169-175, corrected)
        right_matrix_evolution = (
            (1j * tau3 + 1j * self.eta * self.delta_t * tau3) * expansion_tensor
            + 1j * self.delta_t * gap_tensor
            # Electromagnetic terms (corrected to match tex lines 169-175)
            + 1j * self.delta_t * (A_tensor * A_tensor) * tau3  # +A²(t') term (CORRECTED)
            - 1j * self.delta_t * A_t * A_tensor * tau3  # -A(t)A(t') term
            + 1j * self.delta_t**2 * (A_tensor * A_tensor) * tau3 * gap_tensor * tau3  # +A²(t')·Δ(t') term
        )

        #Source term Ŝ_K (tex Eq. 165-177):
        #1. Time derivatives: iτ̂₃·ĝ^K(t-δt,t') + iĝ^K(t,t'-δt)·τ̂₃  (second term implicit in forward sweep)
        rhs_vector_evolution = 1j * tau3 * gk_last_row

        #2. Thermal collision integrals (tex lines 167-170): -2iη·δt² [Σ ĝ^R F τ̂₃ - Σ τ̂₃ F ĝ^A]
        #    precise_convolution includes δt factor, so total prefactor is -2iη·δt·δt = -2iη·δt²
        rhs_vector_evolution += -2j * self.delta_t * self.eta * (
            tau3 * ga.shift(1, axis=1).precise_convolution_right(self.thermal_dist[-1:,:],self.thermal_integral[-1:,:], self.delta_t,self_index=-1)
            - gr[-1:,:].precise_convolution_left(self.thermal_dist.shift(1, axis=1), self.thermal_integral[-1:,:].shift(1, axis=1), self.delta_t, other_index=-1) * tau3)

        #3. Gap-F coupling (tex line 166): -2iδt·F(t,t'-δt)[Δ̂(t)τ̂₃ - τ̂₃Δ̂(t')]
        rhs_vector_evolution += -2 * (-1j * self.delta_t * gap_tensor[-1] * tau3 * self.thermal_dist[-1:,:].shift(1, axis = 1) + 1j * self.delta_t * tau3 * self.thermal_dist[-1:,:].shift(1, axis = 1) * gap_tensor)

        # Phase 4: Direct F-electromagnetic coupling (tex line 167): -2iδt·F(t,t')[A(t) - A(t')]²
        A_diff_squared = (A_t * expansion_tensor - A_tensor) * (A_t * expansion_tensor - A_tensor)
        rhs_vector_evolution += -2j * self.delta_t * self.thermal_dist[-1:,:] * A_diff_squared

        # Phase 5: EM-thermal convolution (tex lines 170-171)
        # Term 1: -2iδt² [A(t)τ̂₃·Σ g^R·F·A(t'') - Σ g^R·A(t'')·F·A(t')τ̂₃]
        # First part: A(t)τ̂₃·[(g^R·A(t'')) @ F]
        weighted_gr_1 = gr[-1:,:] * A_tensor.shift(1, axis=0)  # g^R(t,t'')·A(t'')
        conv1_part1 = A_t * tau3 * weighted_gr_1.precise_convolution_left(
            self.thermal_dist.shift(1, axis=1), self.thermal_integral[-1:,:].shift(1, axis=1), self.delta_t, other_index=-1)
        # Second part: [(g^R·A(t'')) @ F]·A(t')τ̂₃
        conv1_part2 = weighted_gr_1.precise_convolution_left(
            self.thermal_dist.shift(1, axis=1), self.thermal_integral[-1:,:].shift(1, axis=1), self.delta_t, other_index=-1) * A_tensor * tau3
        rhs_vector_evolution += -2j * self.delta_t * (conv1_part1 - conv1_part2)

        # Term 2: +2iδt² [A(t)A(t'')τ̂₃·F·g^A - A(t'')A(t')τ̂₃·F·g^A]
        # First part: A(t)·[(A(t'')τ̂₃·g^A) @ F]  - but this needs precise_convolution_right
        weighted_ga_1 = A_tensor.shift(1, axis=0) * tau3 * ga.shift(1, axis=1)  # A(t'')τ̂₃·g^A(t'',t')
        conv2_part1 = A_t * weighted_ga_1.precise_convolution_right(
            self.thermal_dist[-1:,:], self.thermal_integral[-1:,:], self.delta_t, self_index=-1)
        # Second part: A(t')τ̂₃·[(A(t'')·g^A) @ F]
        weighted_ga_2 = A_tensor.shift(1, axis=0) * ga.shift(1, axis=1)  # A(t'')·g^A(t'',t')
        conv2_part2 = weighted_ga_2.precise_convolution_right(
            self.thermal_dist[-1:,:], self.thermal_integral[-1:,:], self.delta_t, self_index=-1) * A_tensor * tau3
        rhs_vector_evolution += 2j * self.delta_t * (conv2_part1 - conv2_part2)

        # Bilinear sandwich terms (tex lines 180-193, corrected based on boundary extraction)
        # Full bilinear: ie²D δt² A(t)A(t') [+Δ(t) τ₃ g'^K τ₃ - τ₃ g'^K τ₃ Δ(t')]
        # Sandwich term 1: +iδt²·A(t)A(t')·Δ̂(t)τ̂₃ · g^K · τ̂₃ (CORRECTED to match g^R structure)
        left_sandwich_1 = +1j * self.delta_t**2 * A_t * A_tensor * gap_tensor[-1] * tau3
        right_sandwich_1 = tau3 * expansion_tensor

        # Sandwich term 2: -iδt²·A(t)A(t')·τ̂₃ · g^K · τ̂₃·Δ̂(t') (opposite sign from term 1)
        left_sandwich_2 = -1j * self.delta_t**2 * A_t * A_tensor * tau3
        right_sandwich_2 = tau3 * gap_tensor

        g_sandwich_list = [(left_sandwich_1, right_sandwich_1), (left_sandwich_2, right_sandwich_2)]

        # Normalization constraint operators (tex Eq. 187-194):
        # Left: τ̂₃ + δt·ĝ^R(t,t)
        left_matrix_normalization = (tau3 + self.delta_t * gr_diagonal_new) * expansion_tensor
        # Right: -τ̂₃ + δt·ĝ^A(t',t')  (note minus sign)
        right_matrix_normalization = (-tau3 + self.delta_t * ga_diagonal_new) * expansion_tensor

        # Normalization RHS: -2[Σ ĝ^R F + Σ F ĝ^A] (thermal terms only)
        rhs_vector_normalization = -2 * (
            tau3 * ga.shift(1, axis=1).precise_convolution_right(self.thermal_dist[-1:,:],self.thermal_integral[-1:,:],self.delta_t,self_index=-1)
            + gr[-1:,:].precise_convolution_left(self.thermal_dist.shift(1, axis=1), self.thermal_integral[-1:,:].shift(1, axis=1), self.delta_t) * tau3)

        # Solve for off-diagonal elements via forward sweep in t'
        # Phase 6: Pass A_history for source convolutions
        gk_new = self.gk_update_rule(left_matrix_evolution, left_matrix_normalization, right_matrix_evolution, right_matrix_normalization, rhs_vector_evolution, rhs_vector_normalization, new_gr_row, full_ga_matrix=ga, old_gk_matrix=state.gk, g_sandwich_list=g_sandwich_list, A_history=A_history)

        # ========== Diagonal Element g^K(t,t) ==========
        # Same structure as off-diagonal but uses Keldysh symmetry: ĝ^K(t',t) = τ₃[ĝ^K(t,t')]†τ₃

        # Time derivative: iτ̂₃·ĝ^K(t-δt,t) where ĝ^K(t,t) uses gk_new via dagger()+involution
        rhs_vector_evolution_diagonal =  1j * (gk_new.dagger())[-1:] * tau3

        # Thermal collision integrals for diagonal
        rhs_vector_evolution_diagonal += -2j * self.delta_t * self.eta * (
            tau3 * ga.precise_convolution_right(self.thermal_dist[-1:,:],self.thermal_integral[-1:,:], self.delta_t,self_index=-1)[-1,-1:]
            - gr[-1:,:].precise_convolution_left(self.thermal_dist, self.thermal_integral, self.delta_t, other_index=-1)[-1,-1:] * tau3)

        # Gap-F coupling for diagonal
        rhs_vector_evolution_diagonal += -2 * (-1j * self.delta_t * gap_tensor[-1] * tau3 * self.thermal_dist[-1:,-1] + 1j * self.delta_t * tau3 * self.thermal_dist[-1:,-1] * gap_tensor[-1])

        # Normalization RHS for diagonal
        rhs_vector_normalization_diagonal = -2 * (
            tau3 * ga.precise_convolution_right(self.thermal_dist[-1:,:],self.thermal_integral[-1:,:],self.delta_t,self_index=-1)[-1,-1:]
            + gr[-1:,:].precise_convolution_left(self.thermal_dist, self.thermal_integral, self.delta_t, other_index=-1)[-1,-1:] * tau3)

        # Solve for diagonal element g^K(t,t)
        gk_diagonal_new = self.gk_diagonal_update_rule(left_matrix_evolution, left_matrix_normalization, right_matrix_evolution, right_matrix_normalization, rhs_vector_evolution_diagonal, rhs_vector_normalization_diagonal, new_gr_row, full_ga_matrix=ga, old_gk_matrix=state.gk, solution_tensor= gk_new,g_sandwich_list=g_sandwich_list, A_history=A_history)

        return gk_new, gk_diagonal_new

    def gk_update_rule(self, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_matrix_1, rhs_matrix_2, last_gr_row, full_ga_matrix, old_gk_matrix, g_sandwich_list = [], A_history = None):
        """
        Solve for g^K(t,t') via forward sweep in t' (tex Eq. 142 + 187-194).
        Forward sweep implicitly provides g^K(t,t'-δt) needed in backward derivative.
        Different Pauli projection than g^R: (τ₀,τ₃) for evolution, (τ₁,τ₂) for normalization.
        """
        # Define Pauli matrices
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        # Build 4×4 system: evolution (τ₀,τ₃), normalization (τ₁,τ₂)
        matrix_row_1 = (tau0 * left_matrix_1 + right_matrix_1 * tau0).matrix_to_vector()
        matrix_row_2 = (tau3 * left_matrix_1 + right_matrix_1 * tau3).matrix_to_vector()
        matrix_row_3 = (tau1 * left_matrix_2 + right_matrix_2 * tau1).matrix_to_vector()
        matrix_row_4 = (tau2 * left_matrix_2 + right_matrix_2 * tau2).matrix_to_vector()

        for terms in g_sandwich_list:
            left_term = terms[0]
            right_term = terms[1]
            matrix_row_1 += (right_term * tau0 * left_term).matrix_to_vector()
            matrix_row_2 += (right_term * tau3 * left_term).matrix_to_vector()

        # Extract RHS Pauli components
        vector_row_1 = (rhs_matrix_1.trace(0)/2)[0]  # τ₀
        vector_row_2 = (rhs_matrix_1.trace(3)/2)[0]  # τ₃
        vector_row_3 = (rhs_matrix_2.trace(1)/2)[0]  # τ₁
        vector_row_4 = (rhs_matrix_2.trace(2)/2)[0]  # τ₂

        # Initialize: g^K(t, t'=-T_max) from old state (thermal equilibrium boundary)
        solution_tensor = old_gk_matrix[-1, 0:1]

        # Forward sweep: t'=-T_max → 0
        for time in range(1, self.ntpoints):
            # Normalization convolution: Σ ĝ^R ĝ^K + Σ ĝ^K ĝ^A
            # First sum: Σ_{t''=-∞}^{t-δt} ĝ^R(t,t'') ĝ^K(t'',t')
            norm_convolution = -(last_gr_row[0,:-1] @ old_gk_matrix[1:,time]).matrix_to_vector() * self.delta_t

            # Second sum: Σ_{t''=-∞}^{t'-δt} ĝ^K(t,t'') ĝ^A(t'',t'-δt)
            # solution_tensor[1:] is new ĝ^K(t,t'') built so far (from -T_max up to current t'-δt)
            if time > 1:
                norm_convolution += -(solution_tensor[1:] @ full_ga_matrix[0:time-1, time-1]).matrix_to_vector() * self.delta_t

            # Phase 7: Electromagnetic self-convolution terms (tex lines 172-175)
            if time > 1 and A_history is not None and not np.allclose(A_history, 0):
                tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
                A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)
                A_t = A_tensor[-1]

                # Conv 1: A(t)τ̂₃·Σ g^R(t,t'')·A(t'')·τ̂₃·g^K(t'',t')
                weighted_gk_1 = old_gk_matrix[1:, time] * A_tensor[1:] * tau3
                conv1 = A_t * tau3 * (last_gr_row[0, :-1] @ weighted_gk_1)

                # Conv 2: -Σ g^R(t,t'')·A(t'')·τ̂₃·g^K(t'',t')·A(t')τ̂₃
                conv2 = (last_gr_row[0, :-1] @ weighted_gk_1) * A_tensor[time] * tau3

                # Conv 3: A(t)τ̂₃·Σ g^K(t,t'')·A(t'')·τ̂₃·g^A(t'',t')
                weighted_gk_3 = solution_tensor[1:] * A_tensor[1:time] * tau3
                conv3 = A_t * tau3 * (weighted_gk_3 @ full_ga_matrix[0:time-1, time])

                # Conv 4: -Σ g^K(t,t'')·A(t'')·τ̂₃·g^A(t'',t')·A(t')τ̂₃
                conv4 = (weighted_gk_3 @ full_ga_matrix[0:time-1, time]) * A_tensor[time] * tau3

                # Combine: -iδt²[conv1 - conv2 + conv3 - conv4]
                source_A_correction = -1j * self.delta_t**2 * (conv1 - conv2 + conv3 - conv4)

                # Extract τ₀ and τ₃ components for g^K evolution
                source_A_vector = np.array([
                    source_A_correction.trace(0)/2, #[0, time] / 2,  # τ₀ component
                    source_A_correction.trace(3)/2 #[0, time] / 2   # τ₃ component
                ])
            else:
                source_A_vector = np.array([0, 0])

            # Assemble 4×4 system at current t'
            total_matrix = np.array([matrix_row_1[:, time],matrix_row_2[:, time],matrix_row_3[:, time],matrix_row_4[:, time]])

            # Diagonal coupling: [τ̂₃, ĝ^K] from evolution equation
            diagonal_components = solution_tensor[-1].matrix_to_vector()

            # RHS = [evolution] + [diagonal coupling: i·g₃, i·g₀] + [normalization: g₁_conv, g₂_conv] + [A source: A_τ₀, A_τ₃]
            total_vector = np.array([
                vector_row_1[time] + source_A_vector[0],  # τ₀ with A correction
                vector_row_2[time] + source_A_vector[1],  # τ₃ with A correction
                vector_row_3[time],  # τ₁ (normalization)
                vector_row_4[time]   # τ₂ (normalization)
            ]) + np.array([1j * diagonal_components[3], 1j * diagonal_components[0], norm_convolution[1], norm_convolution[2]])

            # Solve for [g₀, g₃, g₁, g₂] at this t'
            g_components = np.linalg.solve(total_matrix, total_vector )

            # Append to solution (builds forward in time)
            solution_tensor.append_right(g_components)

        return solution_tensor

    def gk_diagonal_update_rule(self, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_matrix_1, rhs_matrix_2,  last_gr_row, full_ga_matrix, old_gk_matrix, solution_tensor, g_sandwich_list = [], A_history = None):
        """
        Solve for diagonal element g^K(t,t).
        Uses Keldysh symmetry: ĝ^K(t',t) = τ₃[ĝ^K(t,t')]†τ₃ (involution).
        """
        # Define Pauli matrices
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        # Build 4×4 system (same Pauli projection as off-diagonal)
        matrix_row_1 = (tau0 * left_matrix_1 + right_matrix_1 * tau0).matrix_to_vector()
        matrix_row_2 = (tau3 * left_matrix_1 + right_matrix_1 * tau3).matrix_to_vector()
        matrix_row_3 = (tau1 * left_matrix_2 + right_matrix_2 * tau1).matrix_to_vector()
        matrix_row_4 = (tau2 * left_matrix_2 + right_matrix_2 * tau2).matrix_to_vector()

        # Extract RHS Pauli components at diagonal (t'=t)
        vector_row_1 = (rhs_matrix_1.trace(0)/2)[0]
        vector_row_2 = (rhs_matrix_1.trace(3)/2)[0]
        vector_row_3 = (rhs_matrix_2.trace(1)/2)[0]
        vector_row_4 = (rhs_matrix_2.trace(2)/2)[0]

        # Normalization convolution for diagonal: Σ ĝ^R ĝ^K + Σ ĝ^K ĝ^A
        # First sum: Σ_{t''=-∞}^{t-δt} ĝ^R(t,t'') ĝ^K(t'',t)
        # Uses involution to get ĝ^K(t'',t) = τ₃[ĝ^K(t,t'')]†τ₃ from solution_tensor
        # Second sum: Σ_{t''=-∞}^{t-δt} ĝ^K(t,t'') ĝ^A(t'',t)
        norm_convolution = -(last_gr_row[0,:-1] @ (solution_tensor[1:].involution())).matrix_to_vector() * self.delta_t -(solution_tensor[1:] @ full_ga_matrix[0:-1, -1]).matrix_to_vector() * self.delta_t

        # Assemble 4×4 system at diagonal point (t'=t, i.e., time=-1)
        total_matrix = np.array([matrix_row_1[:, -1],matrix_row_2[:, -1],matrix_row_3[:, -1],matrix_row_4[:, -1]])

        # Diagonal coupling from [τ̂₃, ĝ^K(t,t)]
        diagonal_components = solution_tensor[-1].matrix_to_vector()

        # RHS = [evolution] + [diagonal coupling] + [normalization convolution]
        total_vector = np.array([vector_row_1, vector_row_2,vector_row_3,vector_row_4] ) +  np.array([1j * diagonal_components[3], 1j * diagonal_components[0], norm_convolution[1], norm_convolution[2]])

        # Solve for [g₀, g₃, g₁, g₂] at diagonal
        g_components = np.linalg.solve(total_matrix, total_vector)

        # Return as NambuKeldyshTensor (extract scalar)
        return NambuKeldyshTensor.vector_to_matrix(g_components)[-1]

    def _evolve_state_by_one_timestep(self, state, A_external=None):
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
            self.get_thermal_integral(self.temperature)

        # Compute new gr and gk rows and diagonals
        #* assuming the external vector potential just goes into the system -- will have to be regularized properly
        new_gr_row, new_gr_diag = self._compute_new_gr_row(state, A_history = A_external)
        
        #* update state and then pass it to gk
        state.update_state_gr(new_gr_row, new_gr_diag)

        new_gk_row, new_gk_diag = self._compute_new_gk_row(state, A_history = A_external)

        # Update g^R first, then g^K
        state.update_state_gk(new_gk_row, new_gk_diag)

        # Compute gap at new time using state method
        gap_history = state.get_gap_history()
        gap_new = gap_history[-1]  # Extract gap at newest time

        # Current is zero for now (Stage 2 of project)
        if A_external is None:
            vector_potential_new = 0.0
        else:
            vector_potential_new = A_external[-1]        
        
        #! some error here in computing the current, should be fixed
        current_new = 0 # state.get_current_at_time_t(A_external, self.thermal_dist, self.thermal_integral)

        return gap_new, current_new, vector_potential_new

    def real_time_evolution(self, initial_state, num_timesteps, A_external=None):
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
        gaps = []
        currents = []
        vector_potentials = []
        
        # Start with initial state
        state = initial_state
        
        # Evolve over time with progress bar
        for time_index in tqdm(range(num_timesteps), desc="Real-time evolution"):
            # Evolve by one timestep and get observables
            gap_new, current_new, vector_potential_new = self._evolve_state_by_one_timestep(
                state, A_external
            )
            # Store observables
            gaps += [gap_new]
            currents  += [current_new]
            vector_potentials += [vector_potential_new]

        return state, np.array(gaps), np.array(currents), np.array(vector_potentials)

