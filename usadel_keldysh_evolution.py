"""
Usadel-Keldysh evolution class for real-time dynamics of superconducting systems.
Handles time evolution of retarded Green's function g^R and distribution function f.
"""

import numpy as np
import os
import re
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
        #* overwrite system_parameters_eta
        grid_parameters['eta'] = system_parameters['eta']
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
        if 'dt' in self.grid_parameters:
            # Use provided dt if available (e.g., from saved state)
            self.delta_t = self.grid_parameters['dt']
        else:
            # Compute from grid size
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

    def generate_initial_state(self, Q=0.0, gr0=None):
        """
        Generate initial state from equilibrium for t,t' < 0.

        Steps:
        1. Create EquilibriumSolver object
        2. Compute equilibrium gr and gk in frequency domain
        3. Compute equilibrium current using old solver's _get_current
        4. Fourier transform to two-time representation (only t,t' < 0)
        5. Construct and return StateObject with equilibrium data

        Args:
            Q: Phase gradient (vector potential, default 0)
            gr0: Initial guess for equilibrium gr (optional)

        Returns:
            tuple: (initial_state, gr_tau, gk_tau, equilibrium_current)
                - initial_state: StateObject with equilibrium data for t < 0
                - gr_tau: Retarded Green's function in one-time form
                - gk_tau: Keldysh Green's function in one-time form
                - equilibrium_current: Equilibrium current at given Q

        Calls:
            - EquilibriumSolver.compute_equilibrium_gr()
            - EquilibriumSolver.fourier_transform_to_two_time()
            - UsadelEvolution._get_current() (via equilibrium_solver.usadel_solver)
        """

        # Create equilibrium solver with current grid parameters
        # Need to add omega_grid to grid_parameters for equilibrium solver
        grid_params_with_omega = self.grid_parameters.copy()
        grid_params_with_omega['omega_grid'] = self.omega_grid
        grid_params_with_omega['energy_cutoff'] = self.energy_cutoff

        # Print frequency information
        print(f"\n{'='*60}")
        print(f"Frequency Grid Information:")
        print(f"{'='*60}")
        print(f"  Maximum frequency (ω_max): {np.max(self.omega_grid):.6f}")
        print(f"  Minimum frequency (ω_min): {np.min(self.omega_grid):.6f}")
        print(f"  Energy cutoff (Nyquist):   {self.energy_cutoff:.6f}")
        print(f"  Frequency spacing (dω):    {self.d_omega:.6f}")
        print(f"  Number of frequency points: {len(self.omega_grid)}")
        print(f"{'='*60}\n")

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

        # Compute equilibrium current using old solver's _get_current method
        # Need thermal distribution in frequency domain from old solver
        f_eq = equilibrium_solver._get_thermal_occupation(self.temperature)
        equilibrium_current = equilibrium_solver.usadel_solver._get_current(gr_eq, f_eq, Q)

        print(f"\nEquilibrium observables:")
        print(f"  Gap (Δ₀): {equilibrium_solver.gap_0:.6f}")
        print(f"  Current (J₀): {equilibrium_current:.6f}")
        print(f"  Vector potential (Q): {Q:.6f}")

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

        return initial_state, gr_tau, gk_tau, equilibrium_current

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
            x = np.pi * tau_vals[mask] * temperature
            # F(τ) = -i/π · ln(tanh(πτT/2))
            # Must be purely imaginary since f(τ) is purely imaginary
            tanh_half = np.tanh(x / 2.0)
            result[mask] = -1j/np.pi * np.log(tanh_half + 0j)
            # At τ = 0, set to 0 (principal value)
            return result

        # Compute finite-domain integral
        # F must be purely imaginary since f(τ) is purely imaginary
        # Force this by taking only imaginary part
        F_upper = compute_F_full(tau_upper)
        F_lower = compute_F_full(tau_lower)

        # Take only imaginary part and multiply by 1j to ensure purely imaginary result
        F_two_time = 1j * np.imag(F_upper - F_lower)

        # Set F(0) to BCS regularization: 1/λ + ln(T_c/T)
        bcs_coupling = self._get_BCS_coupling()

        F_zero_bcs = 2j * (1/bcs_coupling + 1/(2 * np.pi) * np.log(self.critical_temperature / temperature))

        # Replace diagonal (τ=0) with BCS value
        diagonal_mask = (np.abs(tau_upper) < 1e-6)
        F_two_time[diagonal_mask] = F_zero_bcs

        # Store as NambuKeldyshTensor (identity in Nambu space)
        self.thermal_integral = NambuKeldyshTensor(F_two_time, pauli_channel=0)

        # Apply filter to suppress early-time numerical artifacts
        # Zero out first 1/5 of time points
        N_t = self.ntpoints
        filter_data = np.append(np.zeros(N_t//3), np.ones(N_t - N_t//3))
        filter_function = NambuKeldyshTensor(filter_data, pauli_channel=0)
        self.thermal_integral = self.thermal_integral #* filter_function

    def construct_discrete_operators(self, terms_dict, state, gap_tensor, g_type = 'r', additional_shift_index = 0):
        #* function assumes that all the terms depending on left time are computed for the current computation time (t,t)
        #* assumes g_matrix last element is (t-dt,t-dt)
        #* to capture the last element, we will have to add some corrections to the last element, so the output should also have an additional, diagonal_transpose output
        #* in the evolution the diagonal_transpose_output shuold be used with final solution. basically take all the terms that have gk_last_row (t-dt,t') that gets zeroed out 
        #* then the extra term should be computed using this. simplest to do using AI to generate the diagonal_gk_correction_terms
        
        """
        Construct discrete Crank-Nicolson operators from Type classifications.

        Each Type contributes to MULTIPLE output components:
        - Operator matrices (left_matrix, right_matrix)
        - Source terms from past times (rhs_vector = V_old)
        - Diagonal coupling terms (rhs_vector_factor_list = V_crt_diag)
        - Convolution terms (rhs_vector_history_list = V_crt_conv)
        - Bilinear sandwich terms (g_sandwich_matrices)

        Args:
            terms_dict: Dictionary with Type keys, e.g.:
                       {'type1': {'L': gap_tensor},
                        'type2': {'R': damping_tensor},
                        'type6_thermal': {'L': thermal_L, 'R': thermal_R},
                        'type6_em': {'L': em_L, 'R': em_R}}
                       Keys can be 'typeN' or 'typeN_description' for multiple terms
            state: StateObject containing g^R or g^K data for computing V_old
                   and extracting diagonal g(t,t) for Type 5
            gap_tensor: NambuKeldyshTensor for the gap function Δ(t)
            shift_index: +1 for g^R (forward shift), -1 for g^K (backward shift)

        Returns:
            8-tuple:
            (left_matrix, right_matrix, rhs_vector,
             rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices,
             diagonal_term_factor_list, diagonal_term_history_list)

            Where:
            - diagonal_term_factor_list: List of (left_op, right_op) tuples for diagonal
              corrections using simple multiplication pattern. Extracts operators from terms
              with g_last_row.shift(-1, axis=1) for separate diagonal element computation.
            - diagonal_term_history_list: List of (left_op, right_op) tuples for diagonal
              corrections using convolution pattern (currently empty, reserved for future).
        """

        #evolution term shapes: type 1: L -- (2,2,ntpoints)
        #                       type 2: R -- (2,2,ntpoints)
        #                       type 3: L -- (2,2,ntpoints,ntpoints)
        #                       type 4: R -- (2,2,ntpoints,ntpoints)
        #                       type 5: L,M,R -- (2,2,ntpoints) each
        #                       type 6: L -- (2,2,ntpoints,ntpoints) R -- (2,2,ntpoints)
        #                       type 7: L -- (2,2,ntpoints) R -- (2,2,ntpoints,ntpoints)
        #                       type 8: L -- (2,2,ntpoints) R -- (2,2,ntpoints)

        # ========== Shape Validation ==========
        # Validate all term shapes before processing
        if True:
            single_time_shape = (2, 2, self.ntpoints)
            two_time_shape = (2, 2, self.ntpoints, self.ntpoints)

            # Define expected shapes for each type
            type_shape_requirements = {
                1: {'L': single_time_shape},
                2: {'R': single_time_shape},
                3: {'L': two_time_shape},
                4: {'R': two_time_shape},
                5: {'L': single_time_shape, 'M': single_time_shape, 'R': single_time_shape},
                6: {'L': two_time_shape, 'R': single_time_shape},
                7: {'L': single_time_shape, 'R': two_time_shape},
                8: {'L': single_time_shape, 'R': single_time_shape}
            }

            for term_name, term_spec in terms_dict.items():
                # Extract type number from term name
                match = re.search(r'type(\d+)', term_name)
                if not match:
                    raise ValueError(f"Invalid term name: '{term_name}'. Expected format: 'typeN' or 'typeN_description'")
                type_num = int(match.group(1))

                # Check if type is valid
                if type_num not in type_shape_requirements:
                    raise ValueError(f"Unknown type number {type_num} in term '{term_name}'. Valid types: 1-8")

                # Get expected shapes for this type
                expected_shapes = type_shape_requirements[type_num]

                # Validate that all required operators are present
                for op_key in expected_shapes.keys():
                    if op_key not in term_spec:
                        raise ValueError(
                            f"Term '{term_name}' (Type {type_num}) is missing required operator '{op_key}'. "
                            f"Expected operators: {list(expected_shapes.keys())}"
                        )

                # Validate shape of each operator
                for op_key, expected_shape in expected_shapes.items():
                    operator = term_spec[op_key]

                    # Extract data shape from NambuKeldyshTensor
                    if isinstance(operator, NambuKeldyshTensor):
                        actual_shape = operator.data.shape
                    else:
                        raise TypeError(
                            f"Term '{term_name}' operator '{op_key}' must be a NambuKeldyshTensor, "
                            f"got {type(operator).__name__}"
                        )

                    # Check shape matches
                    if actual_shape != expected_shape:
                        shape_desc = "single-time (2,2,Nt)" if len(expected_shape) == 3 else "two-time (2,2,Nt,Nt)"
                        raise ValueError(
                            f"Term '{term_name}' operator '{op_key}' has wrong shape.\n"
                            f"  Expected: {expected_shape} ({shape_desc})\n"
                            f"  Got:      {actual_shape}\n"
                            f"  Type {type_num} requires {op_key} to be {shape_desc}"
                        )

        # ========== End Shape Validation ==========

        if g_type == 'r':
            g_matrix = state.gr
            shift_index = -1
            g_diagonal = -gap_tensor
            g_last_row = g_matrix[-1:, :]  # Shape: (2, 2, 1, Nt)
            g_diagonal_current = g_diagonal[-1]

        elif g_type == 'k':
            g_matrix = state.gk
            shift_index = 1
            g_diagonal = state.gk.diagonal_time()
            g_last_row = g_matrix[-1:, :]  # Shape: (2, 2, 1, Nt)
            g_diagonal_current = g_last_row[-1,-1]

        # Initialize outputs
        left_matrix = None
        right_matrix = None
        rhs_vector = None  # V_old: source terms from g(t-δt, ·)
        rhs_vector_history_list = []  # V_crt_conv: convolution sums
        rhs_vector_factor_list = []  # V_crt_diag: diagonal coupling g(t, t'±δt)
        g_sandwich_matrices = []  # Bilinear sandwich terms
        diagonal_term_factor_list = []  # Diagonal correction: simple multiplication
        diagonal_term_history_list = []  # Diagonal correction: convolution (reserved)

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        for type_key, term_spec in terms_dict.items():
            match = re.search(r'type(\d+)', type_key)
            if not match:
                raise ValueError(f"Invalid type key: {type_key}. Expected format: 'typeN' or 'typeN_description'")
            type_num = int(match.group(1))

            # ========== Type 1: Left operator multiplication L(t)·g(t,t') ==========
            if type_num == 1:
                #* F(t,t') = L(t)·g(t,t')
                l_operator = term_spec['L']
                cn_factor = 1.0 / 4.0 * self.delta_t

                #* Operator contribution: M_L = (1/4)·L(t)
                left_contribution = cn_factor * l_operator[-1] * expansion_tensor
                if left_matrix is None:
                    left_matrix = left_contribution
                else:
                    left_matrix = left_matrix + left_contribution

                #* V_old contribution: (1/4)·L(t-δt)·[g(t-δt,t') + g(t-δt,t'+δt)]
                #* g_last row is computed in the old time domain generally so we shift by -1 to bring it to the correct new t' basis, same for g_last_row.shift by correct index +-dt'
                
                v_old_contribution = cn_factor * l_operator[-2] * (g_last_row.shift(-1, axis=1) + g_last_row.shift(shift_index - 1, axis=1))
                
                #* diagonal gk contribution since reffering to g shifted by -1 
                diagonal_term_factor_list.append((cn_factor * l_operator[-2], tau0))

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                #* V_crt_diag contribution: (1/4)·L(t)·g(t,t'+δt)
                rhs_vector_factor_list.append((-cn_factor * l_operator[-1], tau0 * expansion_tensor))

            # ========== Type 2: Right operator multiplication g(t,t')·R(t') ==========
            elif type_num == 2:
                #* F(t,t') = g(t,t')·R(t')
                r_operator = term_spec['R']
                cn_factor = 1.0 / 4.0 * self.delta_t

                #* Operator contribution: M_R = (1/4)·R(t')
                right_contribution = cn_factor * r_operator
                if right_matrix is None:
                    right_matrix = right_contribution
                else:
                    right_matrix = right_matrix + right_contribution

                #* V_old contribution: (1/4)·[g(t-δt,t')·R(t') + g(t-δt,t'+δt)·R(t'+δt)]
                #* g_last row is computed in the old time domain obviously so this assigment is fine, same for g_last_row.shift by correct index +-dt'
                v_old_contribution = cn_factor * (g_last_row.shift(-1, axis=1) * r_operator + g_last_row.shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0))
                
                #* diagonal gk contribution since reffering to g shifted by -1
                diagonal_term_factor_list.append((cn_factor * tau0, r_operator[-1]))

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                #*  V_crt_diag contribution: (1/4)·g(t,t'+δt)·R(t'+δt)
                #* shifting here since we assume this. we will be reading matrix element at t' which corresponds to R(t'+dt')
                rhs_vector_factor_list.append((-tau0, cn_factor * r_operator.shift(shift_index, axis=0)))

            # ========== Type 3: Left convolution ∫_{-∞}^t L(t,t'')·g(t'',t') dt'' ==========
            elif type_num == 3:
                # F(t,t') = ∫_{-∞}^t L(t,t'')·g(t'',t') dt''
                l_operator = term_spec['L']  # Shape: (2, 2, Nt, Nt) with L(t_i, t_j)
                boundary_factor = self.delta_t / 8.0 * self.delta_t 
                interior_factor = self.delta_t / 4.0 * self.delta_t

                #* Operator contribution: M_L = (δt/8)·L(t,t)
                left_contribution = boundary_factor * l_operator[-1, -1] * expansion_tensor
                if left_matrix is None:
                    left_matrix = left_contribution
                else:
                    left_matrix = left_matrix + left_contribution

                #* Boundary terms: (δt/8)·L(t-δt, t-δt)·g(t-δt, t')
                v_old_boundary_1 = boundary_factor * l_operator[-2:-1, -2] * (g_last_row.shift(-1, axis=1) + g_last_row.shift(shift_index-1, axis=1))

                #* diagonal term in gk, since referencing to g_shift(-1)
                diagonal_term_factor_list.append((boundary_factor * l_operator[-2, -2], tau0))

                #* Interior sums from F(t-δt, ·): L(t-δt, :) @ g, sum to t-2δt 
                #* the g_matrix starts from 1 since we assume that the g_matrix corresponds to old, so last element is t-dt and last element of L operator is t and then we remove one extra last element due to trapezoid rule
                v_old_interior_1 = interior_factor * (l_operator[-2:-1, :-2] @ (g_matrix[1:-1, :].shift(-1, axis=1) + g_matrix[1:-1, :].shift(shift_index-1, axis=1)))

                #* diagonal term convolution since convolving with g_matrix which got shifted by -1
                diagonal_term_history_list += [(interior_factor * l_operator[-2:-1,:],tau0)]

                #* Interior sums from F(t, ·): L(t, :) @ g, sum to t-δt 
                #* this summation goes until t-dt so we sum over all elements since the last one corresponds to that time label, the g_matrix starts from 1 since we assume that the g_matrix corresponds to old
                v_old_interior_3 = interior_factor * (l_operator[-1:, :-1] @ (g_matrix[1:].shift(-1, axis=1) + g_matrix[1:].shift(shift_index-1, axis=1)))
                

                #* diagonal term convolution since convolving with g_matrix which got shifted by -1
                diagonal_term_history_list += [(interior_factor * l_operator[-1:,],tau0)]

                v_old_contribution = (v_old_boundary_1 + v_old_interior_1 + v_old_interior_3 )

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                #* V_crt_diag contribution: Boundary term only (δt/8)·L(t,t)·g(t, t'+shift)
                rhs_vector_factor_list.append((-boundary_factor * l_operator[-1, -1], tau0 * expansion_tensor))

            # ========== Type 4: Right convolution ∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt'' ==========
            elif type_num == 4:
                #* F(t,t') = ∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt''
                r_operator = term_spec['R']
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                #* Extract diagonal R(t', t') for boundary terms and M_R
                r_diagonal = r_operator.diagonal_time()
                
                #* Operator contribution: M_R = (δt/8)·R(t',t')
                right_contribution = boundary_factor * r_diagonal

                if right_matrix is None:
                    right_matrix = right_contribution
                else:
                    right_matrix = right_matrix + right_contribution

                #* V_old contribution: F(t-δt, t') and F(t-δt, t'+shift)
                #* the extra shift of -1 comes from the fact that g_last_row goes until t-dt, t-dt in both axis, so we need to shift by 1 to multiply g(t-dt, t-dt) with r(t-dt, t-dt)
                v_old_boundary_1 = boundary_factor * g_last_row.shift(-1, axis=1) * r_diagonal
                v_old_boundary_2 = boundary_factor * g_last_row.shift(shift_index-1, axis=1) * r_diagonal.shift(shift_index, axis=0)

                #* diagonal contribution to gk from since shifting g_last_row by -1
                diagonal_term_factor_list.append((boundary_factor * tau0, r_diagonal[-1]))

                #* Interior sums: (δt/4)·Σ_{t''} g(t-δt, t'')·R(t'', t')
                #*  since g is computed using old basis and r is in the new basis, they need to be relatively shifted w.r.t. eachother when summing, last element is t-dt in g and t,.. element is excluded from r term
                v_old_interior_1 = interior_factor * (g_last_row[:,1:] @ (r_operator[:-1,:] + r_operator[:-1,:].shift(shift_index, axis=1)))  

                v_old_contribution = v_old_boundary_1 + v_old_boundary_2 + v_old_interior_1

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                #* V_crt_diag contribution: (δt/8)·g(t, t'+shift)·R(t'+shift, t'+shift)
                #* here we shift since we are interested in previous term
                rhs_vector_factor_list.append((-tau0, boundary_factor * r_diagonal.shift(shift_index, axis=0)))

                #* V_crt interior contributions: (δt/4)·Σ_{t''} g(t, t'')·R(t'', t') and the shifted one 
                #* the convolution with this term should take approriately such that the last term here is going against the new 1 off diagonal element!
                rhs_vector_history_list.append((-interior_factor * tau0, r_operator + r_operator.shift(shift_index, axis=1)))

            # ========== Type 5: Bilinear convolution L·∫_{t'}^t g·M·g dt''·R ==========
            elif type_num == 5:
                #*F(t,t') = L(t)·∫_{t'}^t g(t,t'')·M(t'')·g(t'',t') dt''·R(t')
                #* Note, we assume that gk will never have type 5 terms, which makes sense given the causality structure, so we dont account for diagonal terms here
                
                l_operator = term_spec['L']
                m_operator = term_spec['M']
                r_operator = term_spec['R']
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                #* Boundary terms at t''=t-δt and t''=t'
                #*-δt/8 L(t-δt) Δ(t-δt)·M(t-δt)·g(t-δt,t')·R(t')
                #* g is shifted to make it appropriate in the new t' basis
                
                #* Sandwich terms from boundary extractions at current time t
                #* Term 1: Upper boundary (t''=t): -δt/8·L(t)·Δ(t)·M(t)·g(t,t')·R(t')

                left_sandwich_1 = boundary_factor * l_operator[-1] * (g_diagonal_current) * m_operator[-1]
                right_sandwich_1 = r_operator
                g_sandwich_matrices.append((left_sandwich_1, right_sandwich_1))

                #* Term 2: Lower boundary (t''=t'): -δt/8·L(t)·g(t,t')·M(t')·Δ(t')·R(t')
                left_sandwich_2 = boundary_factor * l_operator[-1]
                right_sandwich_2 = m_operator * (g_diagonal) * r_operator
                g_sandwich_matrices.append((left_sandwich_2, right_sandwich_2))

                v_old_upper_boundary = boundary_factor * l_operator[-2] * g_diagonal[-2] * m_operator[-2] * g_last_row.shift(-1, axis=1) * r_operator

                v_old_upper_boundary_shift = boundary_factor * l_operator[-2] * (g_diagonal[-2]) * m_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)

                v_old_lower_boundary = boundary_factor * l_operator[-2] * g_last_row.shift(-1, axis=1) * m_operator * (g_diagonal) * r_operator

                v_old_lower_boundary_shift = boundary_factor * l_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * m_operator.shift(shift_index, axis=0) * (g_diagonal.shift(shift_index, axis=0)) * r_operator.shift(shift_index, axis=0)

                #*δt/4 L(t-δt) Σ_{t''=t'+δt}^{t-2δt} g(t-δt,t'')·M(t'')·g(t'',t')·R(t')
                #* both g's here have the old g_matrix layout we skip first and last elements in g, first due to range and last due to sum going to t-2dt, last element is already included through trapezoidal rule
                v_old_interior_1 = interior_factor * l_operator[-2] * ((g_last_row[:,1:-1] * m_operator[:-2]) @ (g_matrix[1:-1].shift(-1, axis=1) * r_operator + g_matrix[1:-1,].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0))) 
                
                v_old_contribution = (v_old_upper_boundary + v_old_upper_boundary_shift + v_old_lower_boundary + v_old_lower_boundary_shift + v_old_interior_1)

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector -v_old_contribution

                #* Term 1: -δt/8 L(t)·g(t,t'+δt)·M(t'+δt)·Δ(t'+δt)·R(t'+δt)
                #* these terms convolution has to be controled appropriately in computational part!!!
                rhs_vector_factor_list.append((-boundary_factor * l_operator[-1], m_operator.shift(shift_index, axis=0) * (g_diagonal.shift(shift_index, axis=0) * r_operator.shift(shift_index, axis=0))))

                #* Term 2: -δt/8 L(t)·Δ(t)·M(t)·g(t,t'+δt)·R(t'+δt)
                rhs_vector_factor_list.append((-boundary_factor * l_operator[-1] * (g_diagonal_current) * m_operator[-1], r_operator.shift(shift_index, axis=0)))

               
                #* V_crt_conv: Interior bilinear sums
                #*Term 1: δt/4 L(t) Σ_{t''=t'+δt}^{t-δt} g(t,t'')·M(t'')·g(t'',t')·R(t')
                #* we need to shift the g_matrix operator because the time indicies are shifted by 1 in the passed g_matrix in t' we also shift the other axis since we want the operator to have the proper time indexing w.r.t. new time indices
                rhs_vector_history_list.append(( -interior_factor * l_operator[-1], m_operator * g_matrix.shift(-1, axis=0).shift(-1, axis=1) * r_operator + m_operator.shift(shift_index, axis=0) * g_matrix.shift(-1, axis=0).shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))

            # ========== Type 6: Mixed left-right ∫_{-∞}^t L(t,t'')·g(t'',t')·R(t') dt'' ==========
            elif type_num == 6:
                #* F(t,t') = ∫_{-∞}^t L(t,t'')·g(t'',t')·R(t') dt''
                l_operator = term_spec['L']  # Shape: (2, 2, Nt, Nt) for two-time
                r_operator = term_spec['R']  # Shape: (2, 2, Nt) for single-time
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                #* Sandwich term: (δt/8)·L(t,t) · g · R(t')
                left_sandwich = boundary_factor * l_operator[-1, -1]
                right_sandwich = r_operator
                g_sandwich_matrices.append((left_sandwich, right_sandwich))

                #* Extract diagonal correction term
                v_old_boundary_1 = boundary_factor * l_operator[-2, -2] * g_last_row.shift(-1, axis=1) * r_operator
                v_old_boundary_2 = boundary_factor * l_operator[-2, -2] * g_last_row.shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)

                #* diagonal gk contribution since referencing g shifted by -1 
                diagonal_term_factor_list.append((boundary_factor * l_operator[-2, -2], r_operator[-1]))

                #*From F(t-δt, ·): L(t-δt, t_init)·g(t_init, t')·R(t')
                v_old_interior_1 = interior_factor * (l_operator[-2:-1, :-2] @ (g_matrix[1:-1, :].shift(-1, axis=1) * r_operator + g_matrix[1:-1, :].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))
                
                diagonal_term_history_list += [(interior_factor * l_operator[-2:-1,:],r_operator[-1])]

                #* Interior sums from F(t, ·): L(t, :) @ (g·R), sum to t-δt (Eq. 509)
                v_old_interior_3 = interior_factor * (l_operator[-1:, :-1] @ (g_matrix[1:].shift(-1, axis=1) * r_operator + g_matrix[1:, :].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))
                diagonal_term_history_list += [(interior_factor * l_operator[-1:,:],r_operator[-1])]

                # Combine ALL V_old contributions
                v_old_contribution = (v_old_boundary_1 + v_old_boundary_2 + v_old_interior_1 + v_old_interior_3)

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector  - v_old_contribution

                #*V_crt_diag: Diagonal coupling term (Eq. 503)
                #*δt/8 L(t,t)·g(t,t'+δt)·R(t'+δt)
                rhs_vector_factor_list.append(( -boundary_factor * l_operator[-1, -1],r_operator.shift(shift_index, axis=0)))

            # ========== Type 7: Mixed right-left L(t)·∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt'' ==========
            elif type_num == 7:
                #* F(t,t') = L(t)·∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt''

                l_operator = term_spec['L']  # Shape: (2, 2, Nt) for single-time
                r_operator = term_spec['R']  # Shape: (2, 2, Nt, Nt) for two-time
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                #* Extract diagonal R(t', t') for boundary terms and sandwich
                r_diagonal = r_operator.diagonal_time()  

                #* Sandwich term: (δt/8)·L(t) · g · R(t',t')
                left_sandwich = boundary_factor * l_operator[-1]
                right_sandwich = r_diagonal
                g_sandwich_matrices.append((left_sandwich, right_sandwich))

                v_old_boundary_1 = boundary_factor * l_operator[-2] * g_last_row.shift(-1, axis=1) * r_diagonal
                v_old_boundary_2 = boundary_factor * l_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * r_diagonal.shift(shift_index, axis=0)

                #* diagonal gk contribution since referencing g shifted by -1
                diagonal_term_factor_list.append((boundary_factor * l_operator[-2], r_diagonal[-1]))
            
                #* Interior sums: (δt/4)·L(t-δt)·Σ_{t''=t_init+δt}^{t'} g(t-δt, t'')·R(t'', t') (Eq. 566-567)
                v_old_interior_1 = interior_factor * l_operator[-2] * (g_last_row[:,1:] @ (r_operator[:-1] +  r_operator[:-1].shift(shift_index, axis=1)))

                v_old_contribution = (v_old_boundary_1 + v_old_boundary_2 + v_old_interior_1)

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector -v_old_contribution

                #* δt/8 L(t)·g(t,t'+δt)·R(t'+δt,t'+δt)
                rhs_vector_factor_list.append((-boundary_factor * l_operator[-1], r_diagonal.shift(shift_index, axis=0)))

                #*Interior sums 
                #*δt/4 L(t) Σ_{t''=t_init+δt}^{t'-δt} g(t,t'')·R(t'',t')
                #*δt/4 L(t) Σ_{t''=t_init+δt}^{t'} g(t,t'')·R(t'',t'+δt)
                rhs_vector_history_list.append((-interior_factor * l_operator[-1], r_operator + r_operator.shift(shift_index, axis=1)))

            # ========== Type 8: Bilinear coupling L(t)·g(t,t')·R(t') ==========
            elif type_num == 8:
                #*F(t,t') = L(t)·g(t,t')·R(t')
                l_operator = term_spec['L']  # Shape: (2, 2, Nt) for single-time
                r_operator = term_spec['R']  # Shape: (2, 2, Nt) for single-time
                cn_factor = 1.0 / 4.0 * self.delta_t

                #*1. Sandwich matrix: (1/4)·L(t) acting on g·R(t')
                left_sandwich = cn_factor * l_operator[-1]
                right_sandwich = r_operator
                g_sandwich_matrices.append((left_sandwich, right_sandwich))

                #* V_crt_diag: (1/4)·L(t)·g(t,t'+δt)·R(t'+δt)
                rhs_vector_factor_list.append(( -cn_factor * l_operator[-1], r_operator.shift(shift_index, axis=0)))

                #* V_old: (1/4)·L(t-δt)·[g(t-δt,t')·R(t') + g(t-δt,t'+δt)·R(t'+δt)]
                v_old_contribution = cn_factor * l_operator[-2] * (g_last_row.shift(-1, axis=1) * r_operator + g_last_row.shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0) )

                #* Diagonal correction for g^K diagonal element since referencing g shifted by -1
                diagonal_term_factor_list.append((cn_factor * l_operator[-2], r_operator[-1]))

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

        #* Return tuple matching generalized_gr_update_rule signature
        # shape of tuples should be:
        # left_matrix: (2,2,1)
        # right_matrix: (2,2,N_t)
        # rhs_vector: (2,2,1,N_t)
        # rhs_vector_history_list: [(2,2), (2,2,N_t)]
        # rhs_vector_factor_list: [(2,2), (2,2,N_t)]
        # g_sandwich_matrices: [(2,2), (2,2,N_t)]
        # diagonal_term_factor_list: [(2,2), (2,2,N_t)]
        # diagonal_term_history_list: [(2,2), (2,2,N_t)]

        return (left_matrix, right_matrix, rhs_vector, rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices, diagonal_term_factor_list, diagonal_term_history_list)

    def get_gr_constraint(self, state, gap_tensor):
        """
        Construct operators for retarded normalization constraint using MIDPOINT RULE.

        Constraint equation (tex lines 1229-1231):
        δt·Σ(t''=t'+δt to t-δt) g'^R(t,t'')·g'^R(t'',t')
          + τ₃·g'^R(t,t') + g'^R(t,t')·τ₃
          - δt·Δ(t)·g'^R(t,t') - δt·g'^R(t,t')·Δ(t') = 0

        Args:
            state: StateObject containing g^R data
            gap_tensor: Gap function as NambuKeldyshTensor

        Returns:
            6-tuple: (left_matrix, right_matrix, rhs_vector,
                      rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices)
        """

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        gr = state.gr
        gr_last_row = gr[-1:, :]

        # Operator matrices (anticommutator structure: τ₃·g + g·τ₃)
        # Left: τ₃ - δt·Δ(t)
        left_matrix = tau3 * expansion_tensor - self.delta_t * gap_tensor[-1] * expansion_tensor/2

        # Right: τ₃ - δt·Δ(t')
        right_matrix = tau3 * expansion_tensor - self.delta_t * gap_tensor/2

        # RHS vector (V_old): no direct terms for gr constraint
        rhs_vector = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))

        # Convolution term: δt·Σ g'^R(t,t'')·g'^R(t'',t') where sum is from t'+δt to t-δt
        # This is handled via history list: (left_term * current_solution) @ right_term
        # For gr constraint: (tau0 * g_current) @ gr gives the convolution
        #* when implemented later, we automatically sum starting from t'+dt since this is previous solution and we eliminate last term by hand!
        rhs_vector_history_list = [(-tau0, gr.shift(-1, axis=1).shift(-1, axis=0) * self.delta_t)]

        # No diagonal coupling terms for this constraint
        rhs_vector_factor_list = []

        # No sandwich terms (convolution handled via history list)
        g_sandwich_matrices = []

        return (left_matrix, right_matrix, rhs_vector,
                rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices)

    def get_gk_constraint(self, state, gap_tensor):
        """
        Construct operators for Keldysh constraint equation using MIDPOINT RULE.

        Constraint equation (tex lines 1263-1268):
        δt·Σ(t''=-∞ to t-δt) g'^R(t,t'')·g'^K(t'',t')
          + δt·Σ(t''=-∞ to t'-δt) g'^K(t,t'')·g'^A(t'',t')
          + [τ₃, g'^K(t,t')] = δt·[Δ(t)·g'^K(t,t') + g'^K(t,t')·Δ(t')]

        Args:
            state: StateObject containing g^R and g^K data
            gap_tensor: Gap function as NambuKeldyshTensor

        Returns:
            6-tuple: (left_matrix, right_matrix, rhs_vector,
                      rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices)
        """
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        diag_g_factor_list = []
        diag_g_history_list = []

        gr = state.gr
        gk = state.gk
        ga = state._r2a()  # g^A = -(g^R)†

        left_matrix = tau3 * expansion_tensor + self.delta_t * gr[-1,-1] * expansion_tensor/2

        right_matrix = -tau3 * expansion_tensor + self.delta_t * ga.diagonal_time()/2

        # RHS vector (V_old): mixed convolution terms + gap sources

        # Term 1: δt·Σ g'^R(t,t'')·g'^K(t'',t') from t''=-∞ to t-δt
        gr_last_row = gr[-1:, :]
        gk_full = gk
        #* this assumes that gr_last_row has correct time index and not shifted t' index, midpoint rule automatically applied
        rhs_term_1 = -self.delta_t * (gr_last_row[:,:-1] @ gk_full[1:]).shift(-1, axis = 1) 
        diag_g_history_list += [(-gr_last_row * self.delta_t, tau0)]

        rhs_term_1 +=  -2 * (tau3 * ga.precise_convolution_right(self.thermal_dist[-1:,:],self.thermal_integral[-1:,:],self.delta_t,self_index=-1) + gr_last_row.precise_convolution_left(self.thermal_dist, self.thermal_integral[-1:,:], self.delta_t, other_index=-1) * tau3)
        rhs_vector = rhs_term_1 

        # Term 2: δt·Σ g'^K(t,t'')·g'^A(t'',t') from t''=-∞ to t'-δt
        # This convolution is handled via history list: (tau0 * gk_current) @ ga 
        #* this ga is now in good frame
        rhs_vector_history_list = [(-tau0 , ga * self.delta_t)]
        # No diagonal coupling terms
        rhs_vector_factor_list = []

        # No sandwich terms for this constraint
        g_sandwich_matrices = []

        return (left_matrix, right_matrix, rhs_vector, rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices, diag_g_factor_list, diag_g_history_list)

    # ========== Real-Time Evolution ==========

    def generalized_g_update_rule(self, g_type, diagonal_entry, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_vector_1, rhs_vector_2, rhs_vector_history_1_list, rhs_vector_history_2_list,rhs_vector_factor_1_list, rhs_vector_factor_2_list, g_sandwich_matrices = [], diagonal_term_factor_1_list=[], diagonal_term_factor_2_list=[], diagonal_term_history_1_list=[], diagonal_term_history_2_list=[]):
        if g_type == 'r':
            trace_index_list = [1,2,3,0]
            loop_start = self.ntpoints - 2
            loop_end = -1
            loop_step = -1
            solution_tensor_index = 0
            solution_tensor = diagonal_entry * NambuKeldyshTensor([1.0], pauli_channel=0)
        elif g_type == 'k':
            trace_index_list = [0,3,1,2]
            loop_start = 0
            loop_end = self.ntpoints
            loop_step = 1
            solution_tensor_index = -1
            solution_tensor = diagonal_entry * NambuKeldyshTensor([1.0], pauli_channel=0)

        #Define Pauli matrices for trace projection
        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        tau_vector = [tau0, tau1, tau2, tau3]

        matrix_row_1 = (tau_vector[trace_index_list[0]] * left_matrix_1 + right_matrix_1 * tau_vector[trace_index_list[0]])
        matrix_row_2 = (tau_vector[trace_index_list[1]] * left_matrix_1 + right_matrix_1 * tau_vector[trace_index_list[1]])
        #Normalization constraint rows: τ₃, τ₀ traces
        matrix_row_3 = (tau_vector[trace_index_list[2]] * left_matrix_2 + right_matrix_2 * tau_vector[trace_index_list[2]])
        matrix_row_4 = (tau_vector[trace_index_list[3]] * left_matrix_2 + right_matrix_2 * tau_vector[trace_index_list[3]])

        for terms in g_sandwich_matrices:
            left_term = terms[0]
            right_term = terms[1]
            matrix_row_1 += (right_term * tau_vector[trace_index_list[0]] * left_term)
            matrix_row_2 += (right_term * tau_vector[trace_index_list[1]] * left_term)

        vector_row_1 = (rhs_vector_1.trace(trace_index_list[0])/2)[0]  # τ₁
        vector_row_2 = (rhs_vector_1.trace(trace_index_list[1])/2)[0]  # τ₂
        vector_row_3 = (rhs_vector_2.trace(trace_index_list[2])/2)[0]  # τ₃
        vector_row_4 = (rhs_vector_2.trace(trace_index_list[3])/2)[0]  # τ₀

        matrix_row_1 = matrix_row_1.matrix_to_vector()
        matrix_row_2 = matrix_row_2.matrix_to_vector()
        matrix_row_3 = matrix_row_3.matrix_to_vector()
        matrix_row_4 = matrix_row_4.matrix_to_vector()

        #========== Backward sweep: t'=0 → -T_max ==========
        #* Main idea: we are actually computing a new element at time-1 and we should appropriately take into account of all the elements
        #* smallest error is given by a weird condition, this has to be understood better
        #* computing g(t,t') given all matrices and convolutions are defined w.r.t. (t,t') themselves.
        for time in range(loop_start, loop_end, loop_step):

            previous_solution = solution_tensor[solution_tensor_index]
            convolution_term_1 = NambuKeldyshTensor(np.zeros((2,2)), pauli_channel=None)
            convolution_term_2 = NambuKeldyshTensor(np.zeros((2,2)), pauli_channel=None)

            for terms in rhs_vector_factor_1_list:
                left_term = terms[0]
                right_term = terms[1]   
                convolution_term_1 += (left_term * previous_solution * right_term[time])
            
            for terms in rhs_vector_factor_2_list:
                left_term = terms[0]
                right_term = terms[1]
                convolution_term_2 += (left_term * previous_solution * right_term[time])
                
            if time != loop_start: 
                for terms in rhs_vector_history_1_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    if g_type == 'r':
                        #* last element in right_term corresponds to time t actually so it should be summed with solution tensor last element

                        convolution_term_1 += (left_term * solution_tensor[:-1]) @ right_term[time+1:-1, time] 
                    elif g_type == 'k':
                        #* note, last right term is actually time t as last time index and last solution tensor is that as well? 
                        #* the sum goes until time which means last element is t'-dt' which it should be summed fulled last one is giving the diagonal
                        if time != loop_end - loop_step:
                            #* in principle first solution corresponds to t, last time is n_points which ends with t-dt, as it should
                            convolution_term_1 += (left_term * solution_tensor[1:]) @ right_term[:time, time] 

                for terms in rhs_vector_history_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    if g_type == 'r':
                        convolution_term_2 += (left_term * solution_tensor[:-1]) @ right_term[time+1:-1, time]  
                    elif g_type == 'k':
                        if time != loop_end - loop_step:
                        #* in principle first solution corresponds to t, last time is n_points which ends with t-dt, as it should
                            convolution_term_2 += (left_term * solution_tensor[1:]) @ right_term[:time, time]

            # ========== Diagonal correction for g^K ==========
            # Apply boundary terms that were zeroed by g_last_row.shift(-1, axis=1)
            # Only applies when computing diagonal element: time == loop_end - loop_step
            if g_type == 'k' and time == loop_end - loop_step:
                # For g^K: loop_end=ntpoints, loop_step=1, so diagonal at time=ntpoints-1
                # diagonal_entry holds g^K(t-δt, t-δt) which is the needed boundary value
                for terms in diagonal_term_factor_1_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    # Pattern: left_term * g_diagonal_current * right_term[time]
                    convolution_term_1 += left_term *  previous_solution.involution() * right_term

                for terms in diagonal_term_factor_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    convolution_term_2 += left_term * previous_solution.involution() * right_term

                # NEW: Diagonal history convolution terms
                # These involve convolution with the diagonal element from g_matrix[0]
                for terms in diagonal_term_history_1_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    convolution_term_1 += left_term[-1,:-1] @ (solution_tensor[1:].involution() * right_term)

                for terms in diagonal_term_history_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    convolution_term_2 += left_term[-1,:-1] @ (solution_tensor[1:].involution()* right_term)

            total_matrix =  np.array([matrix_row_1[:, time],matrix_row_2[:, time],matrix_row_3[:, time],matrix_row_4[:, time]]) 
            #print('total_matrix',total_matrix.shape)
            convolution_term_1_vec = convolution_term_1.matrix_to_vector()
            convolution_term_2_vec = convolution_term_2.matrix_to_vector()
            total_vector = np.array([vector_row_1[time],vector_row_2[time],vector_row_3[time], vector_row_4[time]]) + np.array([convolution_term_1_vec[trace_index_list[0]],convolution_term_1_vec[trace_index_list[1]],convolution_term_2_vec[trace_index_list[2]],convolution_term_2_vec[trace_index_list[3]]])
            #print('total_vector',total_vector.shape)
            g_components = np.linalg.solve(total_matrix, total_vector)
            #Prepend to solution (builds backward in time)
            if g_type == 'r':
                solution_tensor.append(g_components)
            if g_type == 'k':
                solution_tensor.append_right(g_components)
        #Remove diagonal element (only needed for boundary condition)

        #* extra append for the element that will be removed anyways
        if g_type == 'r':
            solution_tensor.append([0,0,0,0])
            return solution_tensor[:-1]
        elif g_type == 'k':
            return solution_tensor

    def _compute_new_gr_row(self, state, A_history=None):
        """
        Evolve retarded Green's function gr by one timestep using Crank-Nicolson discretization.

        Computes g^R(t_{time_index}, t_j) for all j < time_index using the
        discretized Usadel equation with CN averaging over 4 time corners.

        Args:
            state: StateObject with current gr data
            A_history: Optional external vector potential history

        Returns:
            new_gr_row: NambuKeldyshTensor for g^R(t_new, :)
            gr_diagonal_new: Boundary condition g^R(t,t) = -Δ(t)

        Called by:
            - _evolve_state_by_one_timestep()
        """
        # ========== 1. Extract physics parameters ==========

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        
        gap_history = state.get_gap_history()
        #gap_history = np.ones(np.size(gap_history))  * 1.4563 # overwrite gap -- should be removed long term after debugging
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)
        A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)
        A_t = A_tensor[-1]
        A2_t = A_history[-1]**2

        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        # ========== 2. Boundary condition ==========
        gr_diagonal_new = -gap_tensor[-1]

        # ========== 3. Build evolution equation via Type classification ==========
        # NOTE: delta_t factor is applied later in construct_discrete_operators

        evolution_terms = {
            'type1_gap': {'L': -1j * gap_tensor},
            'type2_gap': {'R': 1j * gap_tensor},
            'type1_damping': {'L': 1j * self.eta * tau3 * expansion_tensor},
            'type2_damping': {'R': -1j * self.eta * tau3 * expansion_tensor},
            'type1_em_local': {'L': 1j * A_tensor * A_tensor * tau3 * expansion_tensor},
            'type2_em_local': {'R': -1j * (A_tensor * A_tensor) * tau3},
            'type8_em_cross': {'L': +1j * A_tensor * tau3, 'R': A_tensor},
            'type8_em_cross2': {'L': -1j * A_tensor, 'R': A_tensor * tau3},
            'type5_em_1': {'L': 1j * A_tensor * tau3,'M': A_tensor * tau3, 'R': expansion_tensor},
             'type5_em_2': {'L': -1j *expansion_tensor,'M': A_tensor * tau3,'R': A_tensor * tau3}
            }  

        L1, R1, V1, Vhist1, Vfact1, sandwich1, diag_factor_list, diag_hist_list =  self.construct_discrete_operators(evolution_terms, state, gap_tensor, g_type='r')

        # ========== 4. Add derivative corrections and source terms ==========
        gr_last_row = state.gr[-1:, :]

        L1 = L1 + (1j/2) * tau3 * expansion_tensor

        R1 = R1 - (1j/2) * tau3 * expansion_tensor

        Vfact1.append(((-1j/2) * tau3, expansion_tensor))
        Vfact1.append((tau0, -(1j/2) * tau3 * expansion_tensor))

        v_old_deriv = ((1j/2) * tau3 * gr_last_row.shift(-1, axis=1) + (1j/2) * gr_last_row.shift(-1, axis=1) * tau3
                       + (1j/2) * tau3 * gr_last_row.shift(-2, axis=1) - (1j/2) * gr_last_row.shift(-2, axis=1) * tau3) 

        boundary_correction =  +1j * tau3 * gr_diagonal_new * NambuKeldyshTensor([np.append(np.zeros(self.ntpoints-1),[1.0])], pauli_channel=0).shift(-1, axis=1) #* correction due to delta jump condition term
        V1 = V1 + v_old_deriv + boundary_correction

        # ========== 5. Build normalization constraint operators ==========
        L2, R2, V2, Vhist2, Vfact2, sandwich2 = self.get_gr_constraint(state, gap_tensor)

        # ========== 6. Call unified solver ==========
        gr_new = self.generalized_g_update_rule(
            g_type='r',
            diagonal_entry=gr_diagonal_new,
            left_matrix_1=L1,
            left_matrix_2=L2,
            right_matrix_1=R1,
            right_matrix_2=R2,
            rhs_vector_1=V1,
            rhs_vector_2=V2,
            rhs_vector_history_1_list=Vhist1,
            rhs_vector_history_2_list=Vhist2,
            rhs_vector_factor_1_list=Vfact1,
            rhs_vector_factor_2_list=Vfact2,
            g_sandwich_matrices=sandwich1,
            diagonal_term_factor_1_list=[],
            diagonal_term_factor_2_list=[],
            diagonal_term_history_1_list=[],
            diagonal_term_history_2_list=[]
        )
        return gr_new, gr_diagonal_new 

    def _compute_new_gk_complete(self, state, A_history=None):
        """
        Evolve Keldysh Green's function g^K by one timestep using Crank-Nicolson discretization.

        Computes g^K(t, t_j) for ALL j including diagonal using CN averaging over 4 time corners.
        Includes thermal collision integrals, electromagnetic coupling, and diagonal corrections.
        Now unified: computes both off-diagonal AND diagonal in single solver call.

        Args:
            state: StateObject with current gr and gk data
            A_history: Optional external vector potential history

        Returns:
            gk_new: Complete NambuKeldyshTensor for g^K(t_new, :) including diagonal
            gk_diagonal_new: Diagonal element g^K(t,t) (extracted for clarity)

        Called by:
            - _evolve_state_by_one_timestep()
        """
        # ========== 1. Extract physics parameters ==========
        gr = state.gr
        ga = state._r2a()

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        gk_last_row = state.gk[-1:, :]
        gr_last_row = state.gr[-1:, :]

        gap_history = state.get_gap_history()
        #gap_history = np.ones(np.size(gap_history))  * 1.4563  #overwrite gap
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)
        A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)
        A_t = A_tensor[-1]
        A2_t = A_history[-1]**2
        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        # ========== 1.5: Build Type 3 and Type 4 Electromagnetic Operators ==========
       
        # ========== 2. Build evolution equation via Type classification ==========
        # NOTE: delta_t factor is applied later in construct_discrete_operators
        evolution_terms = {
            'type1_gap': {'L': -1j * gap_tensor},
            'type2_gap': {'R': 1j * gap_tensor},
            'type1_damping': {'L': 1j * self.eta * tau3 * expansion_tensor},
            'type2_damping': {'R': 1j * self.eta * tau3 * expansion_tensor},
            'type1_em_local': {'L': 1j * A_tensor * A_tensor* tau3},
            'type2_em_local': {'R': 1j * (A_tensor * A_tensor) * tau3},
            'type8_em_cross1': {'L': -1j * A_tensor, 'R': A_tensor * tau3},
            'type8_em_cross2': {'L': -1j * A_tensor * tau3, 'R': A_tensor},
        }

        if A_history is not None and not np.allclose(A_history, 0):

            evolution_terms['type3_em1'] = {'L': 1j * A_tensor * tau3 * state.gr * A_tensor * tau3}
            evolution_terms['type6_em2'] = {'L': -1j * state.gr * A_tensor * tau3, 'R': A_tensor * tau3}

            evolution_terms['type4_em1'] = {'R': -1j * A_tensor * tau3 * ga * A_tensor * tau3}
            evolution_terms['type7_em2'] = {'L': 1j * A_tensor * tau3 , 'R': A_tensor * tau3 * ga}

        L1, R1, V1, Vhist1, Vfact1, sandwich1, diag_factor_list_1, diag_hist_list_1 = self.construct_discrete_operators(evolution_terms, state, gap_tensor, g_type='k')

        # ========== 3. Add derivative corrections ==========

        L1 = L1 + (1j/2) * tau3 * expansion_tensor
        R1 = R1 + (1j/2) * tau3 * expansion_tensor

        Vfact1.append((-(1j/2) * tau3, tau0 * expansion_tensor))
        Vfact1.append((tau0, (1j/2) * tau3 * expansion_tensor))

        v_old_deriv = ((1j/2) * tau3 * gk_last_row.shift(-1, axis=1) - (1j/2) * gk_last_row.shift(-1, axis=1) * tau3
                       + (1j/2) * tau3 * gk_last_row + (1j/2) * gk_last_row * tau3)

        diag_factor_list_1 += [((-1j/2) * tau3, tau0), (tau0, +(1j/2) * tau3)]

        V1 = V1 + v_old_deriv

        # ========== 4. Compute and add ALL source terms to V1 ==========

        # ---------- 4.1: Thermal collision integrals and  Gap-F coupling----------
        cn_factor = 1/4 * self.delta_t
        thermal_term = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))

        for dt_prime_shift in [0, 1]:
            for dt_shift in [0,1]:
                dt_end = None if dt_shift == 0 else -dt_shift
                thermal_term += cn_factor * -2j * self.eta * ( tau3 * ga.precise_convolution_right(self.thermal_dist[-1-dt_shift:dt_end,:],self.thermal_integral[-1-dt_shift:dt_end,:], self.delta_t,self_index=-1-dt_shift).shift(dt_prime_shift, axis=1)
                - gr[-1-dt_shift:dt_end,:].precise_convolution_left(self.thermal_dist, self.thermal_integral[-1-dt_shift:dt_end,:], self.delta_t, other_index=-1).shift(dt_prime_shift, axis=1) * tau3)
                thermal_term += cn_factor * -2 * (-1j  * gap_tensor[-1-dt_shift] * tau3 * self.thermal_dist[-1-dt_shift:dt_end,:].shift(dt_prime_shift, axis=1) + 1j * tau3 * self.thermal_dist[-1-dt_shift:dt_end,:].shift(dt_prime_shift, axis=1) * gap_tensor.shift(dt_prime_shift, axis=0))

        V1 = V1 + thermal_term

        # ---------- 4.3: EM-F direct coupling ----------
        if A_history is not None and not np.allclose(A_history, 0):
            em_f_direct = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))
            cn_factor = 1/4 * self.delta_t

            for dt_prime_shift in [0, 1]:
                for dt_shift in [0, 1]:  
                    dt_end = None if dt_shift == 0 else -dt_shift
                    A_past = A_tensor[-1-dt_shift]  # CHANGED: Use -1-dt_shift instead of -2
                    A_tprime = A_tensor.shift(dt_prime_shift, axis=0)
                    thermal_past = self.thermal_dist[-1-dt_shift:dt_end, :].shift(dt_prime_shift, axis=1)

                    A_diff = A_past * expansion_tensor - A_tprime
                    contribution = -2j  * thermal_past * (A_diff * A_diff)
                    em_f_direct += cn_factor * contribution

            V1 = V1 + em_f_direct

            # ---------- 4.4: EM-thermal convolution T1 ----------
            em_thermal_conv1 = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))
            cn_factor = 1/4 * self.delta_t

            for dt_prime_shift in [0, 1]:
                for dt_shift in [0, 1]:  # NEW: Add missing dt_shift loop
                    dt_end = None if dt_shift == 0 else -dt_shift

                    term1_left = -2j * (A_tensor[-1-dt_shift] * tau3 * gr[-1-dt_shift:dt_end, :] * A_tensor).precise_convolution_left(self.thermal_dist, self.thermal_integral[-1-dt_shift:dt_end,:], self.delta_t, other_index=-1).shift(dt_prime_shift, axis=1)
                    term2_left = +2j * (gr[-1-dt_shift:dt_end, :] * A_tensor * tau3).precise_convolution_left(self.thermal_dist, self.thermal_integral[-1-dt_shift:dt_end,:], self.delta_t, other_index=-1).shift(dt_prime_shift, axis=1) * A_tensor.shift(dt_prime_shift, axis=0)
                    term1and2_right = -2j * (A_tensor[-1-dt_shift] * tau3 * A_tensor * ga - A_tensor * ga * A_tensor * tau3).precise_convolution_right(self.thermal_dist[-1-dt_shift:dt_end,:],self.thermal_integral[-1-dt_shift:dt_end,:], self.delta_t,self_index=-1-dt_shift).shift(dt_prime_shift, axis=1)

                    em_thermal_conv1 += cn_factor * (term1_left + term2_left + term1and2_right)

            V1 = V1 + em_thermal_conv1

        # ========== 5. Build Keldysh constraint operators ==========
        L2, R2, V2, Vhist2, Vfact2, sandwich2, diag_factor_list_2, diag_hist_list_2 = self.get_gk_constraint(state, gap_tensor)

        # ========== 6. Call unified solver with diagonal corrections ==========
        gk_boundary = state.gk[-1, 0]  # g^K(t, -infty)

        gk_new = self.generalized_g_update_rule( g_type='k',diagonal_entry=gk_boundary, left_matrix_1=L1, left_matrix_2=L2,right_matrix_1=R1, right_matrix_2=R2,
            rhs_vector_1=V1, rhs_vector_2=V2, rhs_vector_history_1_list=Vhist1, rhs_vector_history_2_list=Vhist2, rhs_vector_factor_1_list=Vfact1,
            rhs_vector_factor_2_list=Vfact2, g_sandwich_matrices=sandwich1, diagonal_term_factor_1_list=diag_factor_list_1, diagonal_term_factor_2_list=diag_factor_list_2,
            diagonal_term_history_1_list=diag_hist_list_1, diagonal_term_history_2_list=diag_hist_list_2)

        # Extract diagonal element from unified result
        gk_diagonal_new = gk_new[-1]
        #gk_diagonal_new = state.gk[-1,-1]
        return gk_new[:-1], gk_diagonal_new

    def _evolve_state_by_one_timestep(self, state, A_external=None):
        """
        Evolve state by one timestep using Crank-Nicolson discretization.

        Steps:
        1. Initialize thermal distribution if needed
        2. Update retarded Green's function g^R
        3. Update complete Keldysh Green's function g^K (unified: both off-diagonal and diagonal)
        4. Extract observables (gap, current, vector potential)

        Args:
            state: StateObject with current data
            A_external: Optional external vector potential history

        Returns:
            gap_new: Gap value at new time t
            current_new: Current at new time t (zero for now)
            vector_potential_new: Vector potential at new time t

        Calls:
            - _compute_new_gr_row(state, A_external)
            - _compute_new_gk_complete(state, A_external)
            - state.update_state_gr(), state.update_state_gk()
        """
        # Initialize thermal distribution if not already done
        if not hasattr(self, 'thermal_dist'):
            self.get_thermal_occupation(self.temperature)
            self.get_thermal_integral(self.temperature)

        # Step 1: Update retarded Green's function (shifts gr matrix)
        new_gr_row, new_gr_diag = self._compute_new_gr_row(state, A_history=A_external)
        state.update_state_gr(new_gr_row, new_gr_diag)

        # Step 2: Compute and update complete Keldysh Green's function (unified: off-diagonal + diagonal)
        new_gk_row, new_gk_diag = self._compute_new_gk_complete(state, A_history=A_external)
        state.update_state_gk(new_gk_row, new_gk_diag)

        # Step 2.5: Update occupation function if tracking is enabled
        if state.occupation_function is not None:
            state.update_state_occupation(self.thermal_dist, self.thermal_integral)

        # Step 3: Extract observables
        gap_history = state.get_gap_history()
        gap_new = gap_history[-1]

        # Current is zero for now (Stage 2 of project)
        if A_external is None:
            vector_potential_new = 0.0
        else:
            vector_potential_new = A_external[-1]

        current_new = state.get_current_at_time_t(A_external, self.thermal_dist, self.thermal_integral)

        return gap_new, current_new, vector_potential_new

    def update_vector_potential(self, old_vector_potential, driving_field):
        """
        Update vector potential using sliding window approach.

        Appends new driving field value and removes oldest value to maintain
        constant history length N_t.

        Args:
            old_vector_potential: Current vector potential array (length N_t)
            driving_field: Time-dependent driving field array (length num_timesteps) or None
            time_index: Current timestep index

        Returns:
            new_vector_potential: Updated array (length N_t)
        """
        if driving_field is None:
            # No driving - return zeros
            return old_vector_potential

        # Get new field value at this timestep
        new_field_value = driving_field

        # Sliding window: remove first element, append new value
        new_vector_potential = np.append(old_vector_potential[1:], new_field_value)

        return new_vector_potential

    def real_time_evolution(self, initial_state, num_timesteps, driving_field=None, track_occupations=False):
        """
        Main real-time evolution loop.

        Evolves state forward in time using sliding window for vector potential.
        At each timestep, the driving field value is appended to the vector potential
        history and the oldest value is removed.

        Steps:
        1. Initialize A_external as zeros (size N_t from initial_state)
        2. For each timestep:
            a. Update A_external using sliding window with new driving_field value
            b. Call _evolve_state_by_one_timestep() with updated A_external
            c. Store returned gap and current values
            d. If track_occupations=True, compute and store energy-time representations
        3. Return evolved state and observable time series

        Args:
            initial_state: StateObject with equilibrium initial conditions
            num_timesteps: Number of time steps to evolve
            driving_field: Optional time-dependent driving field array (length num_timesteps)
                           If None, zero driving field is used. Can be:
                           - None: No driving (A_external remains zeros)
                           - 1D array (length num_timesteps): Time-dependent field values
            track_occupations: If True, compute and store energy-time representations of gr, gk, and f
                               at each timestep (default: False)

        Returns:
            Dictionary with keys:
            - 'final_state': Final evolved StateObject
            - 'gaps': Array of gap values at each timestep (length num_timesteps)
            - 'currents': Array of current values at each timestep (length num_timesteps)
            - 'vector_potentials': Array of vector potential values at each timestep (length num_timesteps)
            - 'gr_energy_time': (only if track_occupations=True) List of energy-time representations of gr
            - 'gk_energy_time': (only if track_occupations=True) List of energy-time representations of gk
            - 'f_energy_time': (only if track_occupations=True) List of energy-time representations of occupation function

        Calls:
            - update_vector_potential(A_external, driving_field, time_index)
            - _evolve_state_by_one_timestep(state, A_external)
        """
        # Initialize arrays to track observables
        gaps = []
        currents = []
        vector_potentials = []

        # Initialize lists to track energy-time representations if requested
        if track_occupations:
            gr_energy_time_list = []
            gk_energy_time_list = []
            f_energy_time_list = []

        # Initialize A_external as zeros (history window)
        # Size N_t from initial_state's time grid
        N_t = initial_state.gr.data.shape[2]
        if driving_field is None:
            A_external = np.zeros(N_t, dtype=complex)
        else:
            A_external = np.ones(N_t, dtype=complex) * driving_field[0]

        # Start with initial state
        state = initial_state

        # Initialize occupation function if tracking is enabled and not already initialized
        if track_occupations and state.occupation_function is None:
            state.occupation_function = 0 * state.gr

        # Disable progress bar on cluster (when SLURM_JOB_ID is set)
        disable_progress = 'SLURM_JOB_ID' in os.environ

        # Evolve over time with progress bar
        for time_index in tqdm(range(num_timesteps), desc="Real-time evolution", disable=disable_progress):

            A_external = self.update_vector_potential(A_external, driving_field[time_index])

            gap_new, current_new, vector_potential_new = self._evolve_state_by_one_timestep(state, A_external)

            gaps += [gap_new]
            currents  += [current_new]
            vector_potentials += [vector_potential_new]

            if track_occupations:
                gr_energy = state.energy_time_representation('gr')
                gk_energy = state.energy_time_representation('gk')
                f_energy = state.energy_time_representation('f')
                gr_energy_time_list.append(gr_energy)
                gk_energy_time_list.append(gk_energy)
                f_energy_time_list.append(f_energy)

        result = { 'final_state': state, 'gaps': np.array(gaps), 'currents': np.array(currents), 'vector_potentials': np.array(vector_potentials)}

        # Add energy-time data if tracking was enabled
        if track_occupations:
            result['gr_energy_time'] = gr_energy_time_list
            result['gk_energy_time'] = gk_energy_time_list
            result['f_energy_time'] = f_energy_time_list
        return result
