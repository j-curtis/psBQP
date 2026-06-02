"""
Usadel-Keldysh evolution class for real-time dynamics of superconducting systems.
Handles time evolution of retarded Green's function g^R and distribution function f.
"""

import numpy as np
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
                7: {'L': single_time_shape, 'R': two_time_shape}
            }

            for term_name, term_spec in terms_dict.items():
                # Extract type number from term name
                match = re.search(r'type(\d+)', term_name)
                if not match:
                    raise ValueError(f"Invalid term name: '{term_name}'. Expected format: 'typeN' or 'typeN_description'")
                type_num = int(match.group(1))

                # Check if type is valid
                if type_num not in type_shape_requirements:
                    raise ValueError(f"Unknown type number {type_num} in term '{term_name}'. Valid types: 1-7")

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
                #* Assumes that L(t) is computed in the new time domain
                # F(t,t') = L(t)·g(t,t')
                l_operator = term_spec['L']
                cn_factor = 1.0 / 4.0 * self.delta_t

                # 1. Operator contribution: M_L = (1/4)·L(t)
                left_contribution = cn_factor * l_operator[-1] * expansion_tensor
                if left_matrix is None:
                    left_matrix = left_contribution
                else:
                    left_matrix = left_matrix + left_contribution

                # 2. V_old contribution: (1/4)·L(t-δt)·[g(t-δt,t') + g(t-δt,t'+δt)]
                #* g_last row is computed in the old time domain generally so we shift by -1 to bring it to the correct new t' basis, same for g_last_row.shift by correct index +-dt'

                # Extract diagonal correction term (Type 1)
                diagonal_term_factor_list.append((cn_factor * l_operator[-2], tau0 * expansion_tensor))

                v_old_contribution = cn_factor * l_operator[-2] * (g_last_row.shift(-1, axis=1) + g_last_row.shift(shift_index-1, axis=1))

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                # 3. V_crt_diag contribution: (1/4)·L(t)·g(t,t'+δt)
                rhs_vector_factor_list.append((-cn_factor * l_operator[-1], tau0 * expansion_tensor))

            # ========== Type 2: Right operator multiplication g(t,t')·R(t') ==========
            elif type_num == 2:
                #* assumes R(t') is computed in old time domain and we only use it for elements that have t' < t. last element is t
                # F(t,t') = g(t,t')·R(t')
                r_operator = term_spec['R']
                cn_factor = 1.0 / 4.0 * self.delta_t

                # 1. Operator contribution: M_R = (1/4)·R(t')
                right_contribution = cn_factor * r_operator
                if right_matrix is None:
                    right_matrix = right_contribution
                else:
                    right_matrix = right_matrix + right_contribution

                # 2. V_old contribution: (1/4)·[g(t-δt,t')·R(t') + g(t-δt,t'+δt)·R(t'+δt)]
                #* g_last row is computed in the old time domain obviously so this assigment is fine, same for g_last_row.shift by correct index +-dt'

                # Extract diagonal correction term (Type 2)
                diagonal_term_factor_list.append((cn_factor * tau0, r_operator))

                v_old_contribution = cn_factor * (g_last_row.shift(-1, axis=1) * r_operator + g_last_row.shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0))

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                # 3. V_crt_diag contribution: (1/4)·g(t,t'+δt)·R(t'+δt)
                #* shifting here since we assume this. we will be reading matrix element at t' which corresponds to R(t'+dt')
                rhs_vector_factor_list.append((-tau0, cn_factor * r_operator.shift(shift_index, axis=0)))

            # ========== Type 3: Left convolution ∫_{-∞}^t L(t,t'')·g(t'',t') dt'' ==========
            elif type_num == 3:
                #* assumes that the L operator is computed such that last element corresponds to (t,t) --must be like this !!!
                #* assumes g_matrix last element is (t-dt,t-dt)
                # F(t,t') = ∫_{-∞}^t L(t,t'')·g(t'',t') dt''
                l_operator = term_spec['L']  # Shape: (2, 2, Nt, Nt) with L(t_i, t_j)
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                # 1. Operator contribution: M_L = (δt/8)·L(t,t)
                left_contribution = boundary_factor * l_operator[-1, -1]
                if left_matrix is None:
                    left_matrix = left_contribution
                else:
                    left_matrix = left_matrix + left_contribution

                # 2. V_old contribution: ALL interior sums (Eq. 295-300)
                # KEY: ALL 4 interior convolutions go to V_old because t'' < t (past times)

                # 2a. Boundary terms: (δt/8)·L(t-δt, t-δt)·g(t-δt, t')
                v_old_boundary_1 = boundary_factor * l_operator[-2:-1, -2] * (g_last_row.shift(-1, axis=1) + g_last_row.shift(shift_index-1, axis=1))

                # 2b. Interior sums from F(t-δt, ·): L(t-δt, :) @ g, sum to t-2δt (Eq. 298)
                #* the g_matrix starts from 1 since we assume that the g_matrix corresponds to old, so last element is t-dt and last element of L operator is t and then we remove one extra last element due to trapezoid rule
                #! check this if its correct!
                v_old_interior_1 = interior_factor * (l_operator[-2:-1, :-2] @ (g_matrix[1:-1, :].shift(-1, axis=1) + g_matrix[1:-1, :].shift(shift_index-1, axis=1)))

                # 2c. Interior sums from F(t, ·): L(t, :) @ g, sum to t-δt (Eq. 299-300)
                # These ALSO go to V_old because all involve g(t'', ·) with t'' < t
                #* this summation goes until t-dt so we sum over all elements since the last one corresponds to that time label, the g_matrix starts from 1 since we assume that the g_matrix corresponds to old
                v_old_interior_3 = interior_factor * (l_operator[-1:, :-1] @ (g_matrix[1:].shift(-1, axis=1) + g_matrix[1:].shift(shift_index-1, axis=1)))

                # Combine ALL V_old contributions (boundary + all 4 interior sums)
                v_old_contribution = (v_old_boundary_1 + v_old_interior_1 + v_old_interior_3 )

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                # 3. V_crt_diag contribution: Boundary term only (δt/8)·L(t,t)·g(t, t'+shift)
                rhs_vector_factor_list.append((-boundary_factor * l_operator[-1, -1], tau0 * expansion_tensor))

            # ========== Type 4: Right convolution ∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt'' ==========
            elif type_num == 4:
                #* again, assuming R(t'',t') last element corresponds to (t, t)
                # From LaTeX Eq. (Type 4, line ~337):
                # F(t,t') = ∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt''
                # R is a two-time object R(t'', t'), need to extract diagonal R(t', t')
                r_operator = term_spec['R']
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                # Extract diagonal R(t', t') for boundary terms and M_R
                # Check if R is two-time (4D) or single-time (3D)
                r_diagonal = r_operator.diagonal_time()  # Extract R(t', t')
                
                # 1. Operator contribution: M_R = (δt/8)·R(t',t')
                right_contribution = boundary_factor * r_diagonal
                if right_matrix is None:
                    right_matrix = right_contribution
                else:
                    right_matrix = right_matrix + right_contribution

                # 2. V_old contribution: F(t-δt, t') and F(t-δt, t'+shift)
                # 2a. Boundary terms: g(t-δt, t')·R(t', t')
                #* this is where the problem comes for the diagonal gk computation since g_last row now involves elements we dont know yet, but we will by symmetry! For diagonal element this has to be added by hand! into computation
                #* for t = t' this will automatically give zero since the shift will zero out the newly shifted element so this is wrong for t = t' but good otherwise
                #* the extra shift of -1 comes from the fact that g_last_row goes until t-dt, t-dt in both axis, so we need to shift by 1 to multiply g(t-dt, t-dt) with r(t-dt, t-dt)

                # Extract diagonal correction term (Type 4)
                diagonal_term_factor_list.append((boundary_factor * tau0, r_diagonal))

                v_old_boundary_1 = boundary_factor * g_last_row.shift(-1, axis=1) * r_diagonal
                v_old_boundary_2 = boundary_factor * g_last_row.shift(shift_index-1, axis=1) * r_diagonal.shift(shift_index, axis=0)

                # 2b. Interior sums: (δt/4)·Σ_{t''} g(t-δt, t'')·R(t'', t')
                # Four interior convolutions total (2 for V_old, 2 for V_crt):
                # V_old terms (use full R for convolution):
                #*  since g is computed using old basis and r is in the new basis, they need to be relatively shifted w.r.t. eachother when summing, last element is t-dt in g and t,.. element is excluded from r term
                v_old_interior_1 = interior_factor * (g_last_row[:,1:] @ (r_operator[:-1,:] + r_operator[:-1,:].shift(shift_index, axis=1)))  

                v_old_contribution = v_old_boundary_1 + v_old_boundary_2 + v_old_interior_1

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                # 3. V_crt_diag contribution: (δt/8)·g(t, t'+shift)·R(t'+shift, t'+shift)
                #* here we shift since we are interested in previous term
                rhs_vector_factor_list.append((-tau0, boundary_factor * r_diagonal.shift(shift_index, axis=0)))

                # 4. V_crt interior contributions: (δt/4)·Σ_{t''} g(t, t'')·R(t'', t') and the shifted one 
                # Two more interior convolutions for current time (use full R)
                #* the convolution with this term should take approriately such that the last term here is going against the new 1 off diagonal element!
                #TODO: check after implementation of convolution this is done appropriately
                rhs_vector_history_list.append((-interior_factor * tau0 * expansion_tensor, r_operator))
                rhs_vector_history_list.append((-interior_factor * tau0 * expansion_tensor, r_operator.shift(shift_index, axis=1)))

            # ========== Type 5: Bilinear convolution L·∫_{t'}^t g·M·g dt''·R ==========
            elif type_num == 5:
                #* assumes all elements are given in new time-basis
                # F(t,t') = L(t)·∫_{t'}^t g(t,t'')·M(t'')·g(t'',t') dt''·R(t')
                # Boundary extractions using g(t,t)=-Δ(t) create sandwich terms
                l_operator = term_spec['L']
                m_operator = term_spec['M']
                r_operator = term_spec['R']
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t


                # 2a. Boundary terms at t''=t-δt and t''=t'
                #-δt/8 L(t-δt) Δ(t-δt)·M(t-δt)·g(t-δt,t')·R(t')
                # Since Δ(t-δt) = -g(t-δt,t-δt) = -g_diagonal[-1]:
                #* g is shifted to make it appropriate in the new t' basis
                #* again missing a term here for the diagonal element of gk

                # Extract diagonal correction terms (Type 5 - upper boundary)
                left_op_upper = -boundary_factor * l_operator[-2] * (-g_diagonal[-2]) * m_operator[-2]
                diagonal_term_factor_list.append((left_op_upper, r_operator))

                v_old_upper_boundary = -boundary_factor * l_operator[-2] * (-g_diagonal[-2]) * m_operator[-2] * g_last_row.shift(-1, axis=1) * r_operator
                v_old_upper_boundary_shift = -boundary_factor * l_operator[-2] * (-g_diagonal[-2]) * m_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)

                # -δt/8 L(t-δt) g(t-δt,t')·M(t')·Δ(t')·R(t')
                # Since Δ(t') = -g(t',t') = -g_diagonal:

                # Extract diagonal correction terms (Type 5 - lower boundary)
                left_op_lower = -boundary_factor * l_operator[-2]
                right_op_lower = m_operator * (-g_diagonal) * r_operator
                diagonal_term_factor_list.append((left_op_lower, right_op_lower))

                v_old_lower_boundary = -boundary_factor * l_operator[-2] * g_last_row.shift(-1, axis=1) * m_operator * (-g_diagonal) * r_operator
                v_old_lower_boundary_shift = -boundary_factor * l_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * m_operator.shift(shift_index, axis=0) * (-g_diagonal.shift(shift_index, axis=0)) * r_operator.shift(shift_index, axis=0)

                # 2b. Interior bilinear convolutions from Eq. 457-458
                # Eq. 457: δt/4 L(t-δt) Σ_{t''=t'+δt}^{t-2δt} g(t-δt,t'')·M(t'')·g(t'',t')·R(t')
                #* both g's here have the old g_matrix layout we skip first and last elements in g, first due to range and last due to sum going to t-2dt, last element is already included through trapezoidal rule
                #* 
                v_old_interior_1 = interior_factor * l_operator[-2] * ((g_last_row[:,1:-1] * m_operator[:-2]) @ (g_matrix[1:-1].shift(-1, axis=1) * r_operator + g_matrix[1:-1,].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0))) 

    
                v_old_contribution = (v_old_upper_boundary + v_old_upper_boundary_shift + v_old_lower_boundary + v_old_lower_boundary_shift + v_old_interior_1)

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector -v_old_contribution

                # 3. V_crt_diag: Diagonal coupling terms (Eq. 449-450)
                # Term 1: -δt/8 L(t)·g(t,t'+δt)·M(t'+δt)·Δ(t'+δt)·R(t'+δt)
                # Since Δ(t') = -g(t',t') = -g_diagonal, we use -g_diagonal for Δ(t'+δt)
                #* these terms convolution has to be controled appropriately in computational part!!!
                rhs_vector_factor_list.append(
                    (-boundary_factor * l_operator[-1],
                    m_operator.shift(shift_index, axis=0) * (g_diagonal.shift(shift_index, axis=0) * r_operator.shift(shift_index, axis=0))
                ))

                # Term 2: -δt/8 L(t)·Δ(t)·M(t)·g(t,t'+δt)·R(t'+δt)
                # Since Δ(t) = -g(t,t) = -g_diagonal_current:
                rhs_vector_factor_list.append((
                    -boundary_factor * l_operator[-1] * (g_diagonal_current) * m_operator[-1],
                    r_operator.shift(shift_index, axis=0)
                ))

                # 4. Sandwich terms from boundary extractions at current time t
                #Bilinear contributions to sandwich term
                # Term 1: Upper boundary (t''=t): -δt/8·L(t)·Δ(t)·M(t)·g(t,t')·R(t')
                # Since Δ(t) = -g(t,t) = -g_diagonal_current:
                left_sandwich_1 = boundary_factor * l_operator[-1] * (g_diagonal_current) * m_operator[-1]
                right_sandwich_1 = r_operator
                g_sandwich_matrices.append((left_sandwich_1, right_sandwich_1))

                # Term 2: Lower boundary (t''=t'): -δt/8·L(t)·g(t,t')·M(t')·Δ(t')·R(t')
                # Since Δ(t') = -g(t',t') = -g_diagonal:
                left_sandwich_2 = boundary_factor * l_operator[-1]
                right_sandwich_2 = m_operator * (g_diagonal) * r_operator
                g_sandwich_matrices.append((left_sandwich_2, right_sandwich_2))

                # 5. V_crt_conv: Interior bilinear sums (Eq. 451-452)
                # Term 1: δt/4 L(t) Σ_{t''=t'+δt}^{t-δt} g(t,t'')·M(t'')·g(t'',t')·R(t')
                # Pre-compute effective operator: M(t'')·g(t'',t')·R(t')
                # Then convolution: g(t,t'') @ [M·g·R](t'',t')
                #* we need to shift the g_matrix operator because the time indicies are shifted by 1 in the passed g_matrix in t' we also shift the other axis since we want the operator to have the proper time indexing w.r.t. new time indices
                rhs_vector_history_list.append((
                    -interior_factor * l_operator[-1],
                    m_operator * g_matrix.shift(-1, axis=0).shift(-1, axis=1) * r_operator
                ))

                # Term 2: δt/4 L(t) Σ_{t''=t'+2δt}^{t-δt} g(t,t'')·M(t'')·g(t'',t'+δt)·R(t'+δt)
                # Pre-compute effective operator with shifted t': M(t'')·g(t'',t'+δt)·R(t'+δt)
                rhs_vector_history_list.append((
                    -interior_factor * l_operator[-1],
                    m_operator.shift(shift_index, axis=0) * g_matrix.shift(-1, axis=0).shift(-1, axis=1).shift(shift_index, axis=1) * r_operator.shift(shift_index, axis=0)
                ))

            # ========== Type 6: Mixed left-right ∫_{-∞}^t L(t,t'')·g(t'',t')·R(t') dt'' ==========
            elif type_num == 6:
                # From LaTeX Eq. (Type 6, line ~487):
                # F(t,t') = ∫_{-∞}^t L(t,t'')·g(t'',t')·R(t') dt''
                # L is two-time object L(t,t''), R is single-time R(t')
                l_operator = term_spec['L']  # Shape: (2, 2, Nt, Nt) for two-time
                r_operator = term_spec['R']  # Shape: (2, 2, Nt) for single-time
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                # 1. No M_L or M_R (operators inside integral or depend on t'')

                # 2. V_old contribution: ALL interior sums (Eq. 505, 508-509)
                # KEY: ALL 4 interior convolutions go to V_old (same as Type 3)

                # 2a. Diagonal boundary terms: L(t-δt, t-δt)·g(t-δt, t')·R(t') (Eq. 505)

                # Extract diagonal correction term (Type 6)
                diagonal_term_factor_list.append((boundary_factor * l_operator[-2, -2], r_operator))

                v_old_boundary_1 = boundary_factor * l_operator[-2, -2] * g_last_row.shift(-1, axis=1) * r_operator
                v_old_boundary_2 = boundary_factor * l_operator[-2, -2] * g_last_row.shift(-1, axis=1).shift(shift_index, axis=1) * r_operator.shift(shift_index, axis=0)

                # 2b. Initial time boundary terms (Eq. 506-507)
                # From F(t-δt, ·): L(t-δt, t_init)·g(t_init, t')·R(t')
                # 2c. Interior sums from F(t-δt, ·): L(t-δt, :) @ (g·R), sum to t-2δt (Eq. 508)
                v_old_interior_1 = interior_factor * (l_operator[-2:-1, :-2] @ (g_matrix[1:-1, :].shift(-1, axis=1) * r_operator + g_matrix[1:-1, :].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))

                # 2d. Interior sums from F(t, ·): L(t, :) @ (g·R), sum to t-δt (Eq. 509)
                # These ALSO go to V_old because g(t'', t') has t'' < t
                v_old_interior_3 = interior_factor * (l_operator[-1:, :-1] @ (g_matrix[1:].shift(-1, axis=1) * r_operator + g_matrix[1:, :].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))

                # Combine ALL V_old contributions
                v_old_contribution = (v_old_boundary_1 + v_old_boundary_2 +
                                     v_old_interior_1 +
                                     v_old_interior_3 + v_old_interior_4)

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector  - v_old_contribution

                # 3. V_crt_diag: Diagonal coupling term (Eq. 503)
                # δt/8 L(t,t)·g(t,t'+δt)·R(t'+δt)
                rhs_vector_factor_list.append((
                    -boundary_factor * l_operator[-1, -1],
                    r_operator.shift(shift_index, axis=0)
                ))

                # 4. Sandwich term: (δt/8)·L(t,t) · g · R(t')
                # Boundary extraction at current time (Eq. 502)
                left_sandwich = boundary_factor * l_operator[-1, -1]
                right_sandwich = r_operator
                g_sandwich_matrices.append((left_sandwich, right_sandwich))

                # 5. V_crt_conv = 0 (Eq. 504)
                # Per line 513: all interior sums have t'' < t, so no V_crt_conv

            # ========== Type 7: Mixed right-left L(t)·∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt'' ==========
            elif type_num == 7:
                # F(t,t') = L(t)·∫_{-∞}^{t'} g(t,t'')·R(t'',t') dt''
                # L is single-time L(t), R is two-time R(t'',t')
                l_operator = term_spec['L']  # Shape: (2, 2, Nt) for single-time
                r_operator = term_spec['R']  # Shape: (2, 2, Nt, Nt) for two-time
                boundary_factor = self.delta_t / 8.0 * self.delta_t
                interior_factor = self.delta_t / 4.0 * self.delta_t

                # Extract diagonal R(t', t') for boundary terms and sandwich
                if r_operator.data.ndim == 4:
                    r_diagonal = r_operator.diagonal_time()  # Extract R(t', t')
                else:
                    r_diagonal = r_operator

                # 1. No M_L or M_R
                # 2. V_old contribution
                # 2a. Boundary terms at diagonal: L(t-δt)·g(t-δt, t')·R(t', t') (Eq. 564)

                # Extract diagonal correction term (Type 7)
                diagonal_term_factor_list.append((boundary_factor * l_operator[-2], r_diagonal))

                v_old_boundary_1 = boundary_factor * l_operator[-2] * g_last_row.shift(-1, axis=1) * r_diagonal
                v_old_boundary_2 = boundary_factor * l_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * r_diagonal.shift(shift_index, axis=0)

                # 2b. Boundary terms at initial time: L(t-δt)·g(t-δt, t_init)·R(t_init, t') (Eq. 565)
            
                # 2c. Interior sums: (δt/4)·L(t-δt)·Σ_{t''=t_init+δt}^{t'} g(t-δt, t'')·R(t'', t') (Eq. 566-567)
                v_old_interior_1 = interior_factor * l_operator[-2] * (g_last_row[:,1:] @ r_operator[:-1])
                v_old_interior_2 = interior_factor * l_operator[-2] * (g_last_row[:,1:].shift(shift_index, axis=1) @ r_operator[:-1].shift(shift_index, axis=1))

                v_old_contribution = (v_old_boundary_1 + v_old_boundary_2 +
                                     v_old_interior_1 + v_old_interior_2)

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector -v_old_contribution

                # 3. V_crt_diag: Diagonal coupling term (Eq. 560)
                # δt/8 L(t)·g(t,t'+δt)·R(t'+δt,t'+δt)
                rhs_vector_factor_list.append((
                    -boundary_factor * l_operator[-1],
                    r_diagonal.shift(shift_index, axis=0)
                ))

                # 4. Sandwich term: (δt/8)·L(t) · g · R(t',t')
                # Boundary extraction at current time (Eq. 559)
                left_sandwich = boundary_factor * l_operator[-1]
                right_sandwich = r_diagonal
                g_sandwich_matrices.append((left_sandwich, right_sandwich))

                # 5. V_crt_conv: Initial boundary + interior sums (Eq. 561-563)
                # Initial time boundary terms (Eq. 561):
                # δt/8 L(t)·g(t,t_init)·R(t_init,t')
                rhs_vector_history_list.append((
                    -boundary_factor * l_operator[-1],
                    r_operator
                ))
                # δt/8 L(t)·g(t,t_init)·R(t_init,t'+δt)
                rhs_vector_history_list.append((
                    -boundary_factor * l_operator[-1],
                    r_operator.shift(shift_index, axis=1)
                ))

                # Interior sums (Eq. 562-563):
                # δt/4 L(t) Σ_{t''=t_init+δt}^{t'-δt} g(t,t'')·R(t'',t')
                rhs_vector_history_list.append((-interior_factor * l_operator[-1] * expansion_tensor, r_operator))
                # δt/4 L(t) Σ_{t''=t_init+δt}^{t'} g(t,t'')·R(t'',t'+δt)
                rhs_vector_history_list.append((-interior_factor * l_operator[-1] * expansion_tensor, r_operator.shift(shift_index, axis=1)))

        # Return tuple matching generalized_gr_update_rule signature
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

        gr = state.gr
        gk = state.gk
        ga = state._r2a()  # g^A = -(g^R)†

        left_matrix = tau3 * expansion_tensor + self.delta_t * gr[-1,-1] * expansion_tensor/2

        right_matrix = -tau3 * expansion_tensor + self.delta_t * ga[-1,-1] * expansion_tensor/2

        # RHS vector (V_old): mixed convolution terms + gap sources

        # Term 1: δt·Σ g'^R(t,t'')·g'^K(t'',t') from t''=-∞ to t-δt
        gr_last_row = gr[-1:, :]
        gk_full = gk
        #* this assumes that gr_last_row has correct time index and not shifted t' index, midpoint rule automatically applied
        rhs_term_1 = -self.delta_t * (gr_last_row[:,:-1] @ gk_full[1:]).shift(-1, axis = 1)
        #! should also have a diagonal contribution here! at the last element explicitly
        rhs_term_1 +=  -2 * (tau3 * ga.precise_convolution_right(self.thermal_dist[-1:,:],self.thermal_integral[-1:,:],self.delta_t,self_index=-1) + gr_last_row.precise_convolution_left(self.thermal_dist, self.thermal_integral[-1:,:], self.delta_t, other_index=-1) * tau3)
        rhs_vector = rhs_term_1 

        # Term 2: δt·Σ g'^K(t,t'')·g'^A(t'',t') from t''=-∞ to t'-δt
        # This convolution is handled via history list: (tau0 * gk_current) @ ga 
        #* actually this ga is now in good frame
        rhs_vector_history_list = [(-tau0 , ga * self.delta_t)]
        # No diagonal coupling terms
        rhs_vector_factor_list = []

        # No sandwich terms for this constraint
        g_sandwich_matrices = []

        return (left_matrix, right_matrix, rhs_vector,
                rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices)

    # ========== Real-Time Evolution ==========

    def generalized_g_update_rule(self, g_type, diagonal_entry, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_vector_1, rhs_vector_2, rhs_vector_history_1_list, rhs_vector_history_2_list,rhs_vector_factor_1_list, rhs_vector_factor_2_list, g_sandwich_matrices = [], diagonal_term_factor_1_list=[], diagonal_term_factor_2_list=[]):
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
            #print(solution_tensor.shape)
            previous_solution = solution_tensor[solution_tensor_index]
            #solution_string = solution_tensor[(self.ntpoints - solution_tensor_index) % self.ntpoints: self.ntpoints - (solution_tensor_index_2) % self.ntpoints]
            #print(solution_tensor.data.shape)
            #Normalization convolution: Σ_{t''=t'+δt}^{t-δt} ĝ^R(t,t'') ĝ^R(t'',t')
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
                        convolution_term_1 += (left_term * solution_tensor[1:]) @ right_term[:time, time] 

                for terms in rhs_vector_history_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    if g_type == 'r':
                        convolution_term_2 += (left_term * solution_tensor[:-1]) @ right_term[time+1:-1, time]  
                    elif g_type == 'k':
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
                    convolution_term_1 += left_term * diagonal_entry * right_term[time]

                for terms in diagonal_term_factor_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    convolution_term_2 += left_term * diagonal_entry * right_term[time]

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
            solution_tensor.append(g_components)
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
        gap_history = state.get_gap_history()
        # ! overwrite gap
        gap_history = np.ones(np.size(gap_history))  * 1.523294
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)
        A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)
        A_t = A_tensor[-1]
        A2_t = A_history[-1]**2

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
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
            'type1_em_local': {'L': 1j * A2_t * tau3 * expansion_tensor},
            'type2_em_local': {'R': -1j * (A_tensor * A_tensor) * tau3},
            'type1_em_cross': {'L': -1j * A_t * A_tensor * tau3},
            'type2_em_cross': {'R': -1j * A_t * A_tensor * tau3},
            'type5_em_1': {'L': 1j * A_t * tau3 * expansion_tensor,'M': A_tensor * tau3, 'R': expansion_tensor}, 
             'type5_em_2': {'L': expansion_tensor,'M': A_tensor * tau3,'R': -1j * A_tensor * tau3}
            }  

        L1, R1, V1, Vhist1, Vfact1, sandwich1, _, _ =  self.construct_discrete_operators(evolution_terms, state, gap_tensor, g_type='r')

        # ========== 4. Add derivative corrections and source terms ==========
        gr_last_row = state.gr[-1:, :]

        # M_L correction: +(i/2)τ₃
        L1 = L1 + (1j/2) * tau3 * expansion_tensor

        # M_R correction (opposite sign!): -(i/2)τ₃
        R1 = R1 - (1j/2) * tau3 * expansion_tensor

        # V_crt_diag correction (couples to g(t, t'+δt))
        Vfact1.append(((-1j/2) * tau3, expansion_tensor))
        Vfact1.append((tau0, -(1j/2) * tau3 * expansion_tensor))

        # V_old correction (from derivatives)
        # Retarded: -(i/2)τ₃·g(t-δt,t') - (i/2)g(t-δt,t')·τ₃
        #           -(i/2)τ₃·g(t-δt,t'+δt) + (i/2)g(t-δt,t'+δt)·τ₃  [+ sign on last term]
        
        v_old_deriv = ((1j/2) * tau3 * gr_last_row.shift(-1, axis=1) + (1j/2) * gr_last_row.shift(-1, axis=1) * tau3
                       + (1j/2) * tau3 * gr_last_row.shift(-2, axis=1)
                       - (1j/2) * gr_last_row.shift(-2, axis=1) * tau3) 

        # Add all source terms to V1
        #! problem is somehow double shifting!
        boundary_correction =  +1j * tau3 * state.gr[-1,-1] * NambuKeldyshTensor([np.append(np.zeros(self.ntpoints-1),[1.0])], pauli_channel=0).shift(-1, axis=1) #* correction due to delta jump condition term
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
            g_sandwich_matrices=sandwich1
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

        gap_history = state.get_gap_history()
        # ! overwrite gap
        gap_history = np.ones(np.size(gap_history))  * 1.523294
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)
        A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)
        A_t = A_tensor[-1]
        A2_t = A_history[-1]**2

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)

        # ========== 1.5: Build Type 3 and Type 4 Electromagnetic Operators ==========
        if A_history is not None and not np.allclose(A_history, 0):
            # Type 3A1: L(t,t') = 1j * A(t)·τ₃·g'^R(t,t')·A(t')·τ₃
            L_em_conv_A1 = 1j * A_tensor * tau3 * gr * A_tensor * tau3

            # Type 3A2: L(t,t') = 1j * g'^R(t,t')·A(t')·τ₃
            L_em_conv_A2 = 1j * gr * A_tensor * tau3

            # Type 4B1: R(t',t₂) = 1j * A(t')·τ₃·g'^A(t',t₂)
            R_em_conv_B1 = 1j * A_tensor * tau3 * ga

            # Type 4B2: R(t',t₂) = -1j * A(t')·τ₃·g'^A(t',t₂)·A(t₂)·τ₃
            R_em_conv_B2 = -1j * A_tensor * tau3 * ga * A_tensor * tau3
        else:
            # No electromagnetic field, create dummy zero operators
            L_em_conv_A1 = None
            L_em_conv_A2 = None
            R_em_conv_B1 = None
            R_em_conv_B2 = None

        # ========== 2. Build evolution equation via Type classification ==========
        # NOTE: delta_t factor is applied later in construct_discrete_operators
        evolution_terms = {
            'type1_gap': {'L': -1j * gap_tensor},
            'type2_gap': {'R': 1j * gap_tensor},  # SAME SIGN as Type 1
            'type1_damping': {'L': 1j * self.eta * tau3 * expansion_tensor},
            'type2_damping': {'R': 1j * self.eta * tau3 * expansion_tensor},  # SAME SIGN
            'type1_em_local': {'L': 1j * A2_t * tau3 * expansion_tensor},
            'type2_em_local': {'R': 1j * (A_tensor * A_tensor) * tau3},  # SAME SIGN
            'type1_em_cross': {'L': -1j * A_t * A_tensor * tau3},
            'type2_em_cross': {'R': -1j * A_t * A_tensor * tau3},
            'type5_em_1': {
                 'L': 1j * A_t * tau3 * expansion_tensor,
                 'M': A_tensor * tau3,
                 'R': expansion_tensor
             },
             'type5_em_2': {
                 'L': expansion_tensor,
                 'M': A_tensor * tau3,
                 'R': 1j *  A_tensor * tau3  # PLUS SIGN (not minus like gr)
            }
        }

        # Add Type 3 and Type 4 electromagnetic convolutions
        if L_em_conv_A1 is not None:
            evolution_terms['type3_em_A1'] = {'L': L_em_conv_A1}
            evolution_terms['type3_em_A2'] = {'L': L_em_conv_A2}
            evolution_terms['type4_em_B1'] = {'R': R_em_conv_B1}
            evolution_terms['type4_em_B2'] = {'R': R_em_conv_B2}

        L1, R1, V1, Vhist1, Vfact1, sandwich1, diag_factor_list, diag_hist_list = self.construct_discrete_operators(evolution_terms, state, gap_tensor, g_type='k')

        # ========== 3. Add derivative corrections ==========
        gk_last_row = state.gk[-1:, :]

        # M_L correction: +(i/2)τ₃
        L1 = L1 + (1j/2) * tau3 * expansion_tensor

        # M_R correction (SAME SIGN!): +(i/2)τ₃
        R1 = R1 + (1j/2) * tau3 * expansion_tensor

        # V_crt_diag correction (couples to g^K(t, t'-δt), BACKWARD!)
        Vfact1.append((-(1j/2) * tau3, tau0 * expansion_tensor))
        Vfact1.append((tau0, (1j/2) * tau3 * expansion_tensor))

        # V_old correction (both terms MINUS)
        # Keldysh: -(i/2)τ₃·g^K(t-δt,t') - (i/2)g^K(t-δt,t')·τ₃
        #          -(i/2)τ₃·g^K(t-δt,t'-δt) - (i/2)g^K(t-δt,t'-δt)·τ₃  [MINUS on both]
        v_old_deriv = ((1j/2) * tau3 * gk_last_row.shift(-1, axis=1) - (1j/2) * gk_last_row.shift(-1, axis=1) * tau3
                       + (1j/2) * tau3 * gk_last_row
                       + (1j/2) * gk_last_row * tau3)

        # ========== 4. Compute and add ALL source terms to V1 ==========
        # V1 already contains V_old from construct_discrete_operators
        # Add derivative corrections
        V1 = V1 + v_old_deriv

        # 4-corner CN: Only past time (t-δt) contributes to RHS
        # Current time (t) handled by operator matrices

        #* note, there were some errors regarding indexing
        #* check dt terms etc, this needs to be fixed in accordance to all the old terms
        #* check how the source terms should even be computed 
        #* write out the indexing structure for all the terms in the discrete operator assigment and source term computation
        #* -- each term should have explicit indexing comment written out! and explained what happens when something is passed in. 
        #* -- also minimize the amount of necessary convolutions 
        # ---------- 4.1: Thermal collision integrals ----------
        # ---------- 4.2: Gap-F coupling ----------

        #! need also the change of first index and second one, simply compute and average out with 1/4 factor
        #* basically we can do this for all the terms, simply compute and average over 4 sites 
        #* careful with the - signs since they should be added, overall indexing here should be trivial 
        #* to minimize the number of convolutions, we can also group together different convoluted terms maybe?

        cn_factor = 1/4 
        thermal_term = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))

        for dt_prime_shift in [0, 1]:
            for dt_shift in [0,1]:
                dt_end = None if dt_shift == 0 else -dt_shift
                thermal_term += cn_factor * -2j * self.delta_t * self.eta * ( tau3 * ga.precise_convolution_right(self.thermal_dist[-1-dt_shift:dt_end,:],self.thermal_integral[-1-dt_shift:dt_end,:], self.delta_t,self_index=-1).shift(dt_prime_shift, axis=1)
                - gr[-1-dt_shift:dt_end,:].precise_convolution_left(self.thermal_dist, self.thermal_integral[-1-dt_shift:dt_end,:], self.delta_t, other_index=-1).shift(dt_prime_shift, axis=1) * tau3)
                thermal_term += cn_factor * -2 * (-1j * self.delta_t * gap_tensor[-1-dt_shift] * tau3 * self.thermal_dist[-1-dt_shift:dt_end,:].shift(dt_prime_shift, axis=1) + 1j * self.delta_t * tau3 * self.thermal_dist[-1-dt_shift:dt_end,:].shift(dt_prime_shift, axis=1) * gap_tensor.shift(dt_prime_shift, axis=0))

        V1 = V1 + thermal_term

        # ---------- 4.3: EM-F direct coupling ----------
        if A_history is not None and not np.allclose(A_history, 0):
            em_f_direct = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))
            cn_factor = 1/4

            for dt_prime_shift in [0, 1]:
                for dt_shift in [0, 1]:  # NEW: Add missing dt_shift loop
                    dt_end = None if dt_shift == 0 else -dt_shift
                    A_past = A_tensor[-1-dt_shift]  # CHANGED: Use -1-dt_shift instead of -2
                    A_tprime = A_tensor.shift(dt_prime_shift, axis=0)
                    thermal_past = self.thermal_dist[-1-dt_shift:dt_end, :].shift(dt_prime_shift, axis=1)

                    A_diff = A_past * expansion_tensor - A_tprime
                    contribution = -2j * self.delta_t * thermal_past * (A_diff * A_diff)
                    em_f_direct += cn_factor * contribution

            V1 = V1 + em_f_direct

            # ---------- 4.4: EM-thermal convolution T1 ----------
            em_thermal_conv1 = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))
            cn_factor = 1/4

            for dt_prime_shift in [0, 1]:
                for dt_shift in [0, 1]:  # NEW: Add missing dt_shift loop
                    dt_end = None if dt_shift == 0 else -dt_shift

                    A_past = A_tensor[-1-dt_shift]  # CHANGED
                    A_tprime = A_tensor.shift(dt_prime_shift, axis=0)
                    thermal_past = self.thermal_dist[-1-dt_shift:dt_end, :].shift(dt_prime_shift, axis=1)
                    integral_past = self.thermal_integral[-1-dt_shift:dt_end, :].shift(dt_prime_shift, axis=1)

                    # Create A_past as row tensor for multiplication
                    A_past_row = NambuKeldyshTensor(
                        A_past * np.ones((1, self.ntpoints)),
                        pauli_channel=0
                    )

                    weighted_gr = gr[-1-dt_shift:dt_end, :] * A_past_row  # CHANGED

                    conv1 = A_past * tau3 * weighted_gr.precise_convolution_left(
                        thermal_past,
                        integral_past,
                        self.delta_t,
                        other_index=0
                    )

                    conv2 = weighted_gr.precise_convolution_left(
                        thermal_past,
                        integral_past,
                        self.delta_t,
                        other_index=0
                    ) * A_tprime * tau3

                    em_thermal_conv1 += cn_factor * (-2j * self.delta_t) * (conv1 - conv2)

            V1 = V1 + em_thermal_conv1

            # ---------- 4.5: EM-thermal convolution T2 ----------
            em_thermal_conv2 = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))
            cn_factor = 1/4

            for dt_prime_shift in [0, 1]:
                for dt_shift in [0, 1]:  # NEW: Add missing dt_shift loop
                    dt_end = None if dt_shift == 0 else -dt_shift

                    A_past = A_tensor[-1-dt_shift]  # CHANGED
                    A_tprime = A_tensor.shift(dt_prime_shift, axis=0)
                    ga_past_shifted = ga[-1-dt_shift:dt_end, :].shift(dt_prime_shift, axis=1)  # CHANGED

                    # Create A_past as row tensor
                    A_past_row = NambuKeldyshTensor(
                        A_past * np.ones((1, self.ntpoints)),
                        pauli_channel=0
                    )

                    weighted_ga1 = A_past_row * tau3 * ga_past_shifted
                    weighted_ga2 = A_past_row * ga_past_shifted

                    conv1 = A_past * weighted_ga1.precise_convolution_right(
                        self.thermal_dist[-1-dt_shift:dt_end, :],  # CHANGED
                        self.thermal_integral[-1-dt_shift:dt_end, :],  # CHANGED
                        self.delta_t,
                        self_index=0
                    )

                    conv2 = weighted_ga2.precise_convolution_right(
                        self.thermal_dist[-1-dt_shift:dt_end, :],  # CHANGED
                        self.thermal_integral[-1-dt_shift:dt_end, :],  # CHANGED
                        self.delta_t,
                        self_index=0
                    ) * A_tprime * tau3

                    em_thermal_conv2 += cn_factor * (2j * self.delta_t) * (conv1 - conv2)

            V1 = V1 + em_thermal_conv2
        # ========== 5. Build Keldysh constraint operators ==========
        L2, R2, V2, Vhist2, Vfact2, sandwich2 = self.get_gk_constraint(state, gap_tensor)

        # ========== 6. Call unified solver with diagonal corrections ==========
        # Boundary condition: g^K(t-δt, t-δt) for diagonal corrections
        gk_diagonal_boundary = state.gk[-1, -1]  # g^K(t-δt, t-δt)

        gk_new = self.generalized_g_update_rule(
            g_type='k',
            diagonal_entry=gk_diagonal_boundary,
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
            diagonal_term_factor_1_list=diag_factor_list,
            diagonal_term_factor_2_list=diag_hist_list
        )

        # Extract diagonal element from unified result
        gk_diagonal_new = gk_new[-1]
        gk_diagonal_new = gk_last_row[-1,-1]
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

        # Step 3: Extract observables
        gap_history = state.get_gap_history()
        gap_new = gap_history[-1]

        # Current is zero for now (Stage 2 of project)
        if A_external is None:
            vector_potential_new = 0.0
        else:
            vector_potential_new = A_external[-1]

        current_new = 0  # state.get_current_at_time_t(A_external, self.thermal_dist, self.thermal_integral)

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

        return state, np.array(gaps), np.array(currents), np.array(vector_potentials)#

    #! DEPRACATED
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
    #! DEPRACATED
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
    
    def gk_diagonal_update_rule(self, diagonal_entry, solution_tensor, left_matrix_1, left_matrix_2, right_matrix_1, right_matrix_2, rhs_vector_1, rhs_vector_2, rhs_vector_history_1_list, rhs_vector_history_2_list,rhs_vector_factor_1_list, rhs_vector_factor_2_list, g_sandwich_matrices = []):
        #* a function which updates the g based on the structure of the equation
        trace_index_list = [0,3,1,2]
        loop_start = 1
        loop_end = self.ntpoints
        loop_step = 1
        solution_tensor_index = -1

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

        #TODO: have to check ranges and indices here, something looks fishy!
        previous_solution = solution_tensor[solution_tensor_index]
        #solution_string = solution_tensor[(self.ntpoints - solution_tensor_index) % self.ntpoints: self.ntpoints - (solution_tensor_index_2) % self.ntpoints]
        #print(solution_tensor.data.shape)
        #Normalization convolution: Σ_{t''=t'+δt}^{t-δt} ĝ^R(t,t'') ĝ^R(t'',t')
        time = self.ntpoints
        if time == loop_start:
            #First two iterations (t'=0): no intermediate points, convolution = 0
            convolution_term_1 = NambuKeldyshTensor(np.zeros((2,2)), pauli_channel=None)
            convolution_term_2 = NambuKeldyshTensor(np.zeros((2,2)), pauli_channel=None)
        else:
            # Initialize to zero Nambu tensors
            convolution_term_1 = NambuKeldyshTensor(np.zeros((2,2)), pauli_channel=None)
            convolution_term_2 = NambuKeldyshTensor(np.zeros((2,2)), pauli_channel=None)

            for terms in rhs_vector_history_1_list:
                left_term = terms[0]
                right_term = terms[1]
                convolution_term_1 += (left_term * solution_tensor) @ right_term[:time, time] * self.delta_t

            for terms in rhs_vector_factor_1_list:
                left_term = terms[0]
                right_term = terms[1]
                convolution_term_1 += (left_term * previous_solution * right_term[time])

            for terms in rhs_vector_history_2_list:
                left_term = terms[0]
                right_term = terms[1]
                convolution_term_2 += (left_term * solution_tensor) @ right_term[:time, time] * self.delta_t

            for terms in rhs_vector_factor_2_list:
                left_term = terms[0]
                right_term = terms[1]
                convolution_term_2 += (left_term * previous_solution * right_term[time])


            total_matrix =  np.array([matrix_row_1[:, time],matrix_row_2[:, time],matrix_row_3[:, time],matrix_row_4[:, time]]) 
            #print('total_matrix',total_matrix.shape)
            convolution_term_1_vec = convolution_term_1.matrix_to_vector()
            convolution_term_2_vec = convolution_term_2.matrix_to_vector()
            total_vector = np.array([vector_row_1[time],vector_row_2[time],vector_row_3[time], vector_row_4[time]]) + np.array([convolution_term_1_vec[trace_index_list[0]],convolution_term_1_vec[trace_index_list[1]],convolution_term_2_vec[trace_index_list[2]],convolution_term_2_vec[trace_index_list[3]]])
            #print('total_vector',total_vector.shape)
            g_components = np.linalg.solve(total_matrix, total_vector)
            #Prepend to solution (builds backward in time)

            solution_tensor.append_right(g_components)

        #Remove diagonal element (only needed for boundary condition)
        

        return solution_tensor
    
    
    #! DEPRACATED
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
