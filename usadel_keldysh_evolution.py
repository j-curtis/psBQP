"""
Usadel-Keldysh evolution class for real-time dynamics of superconducting systems.
Handles time evolution of retarded Green's function g^R and distribution function f.
"""

import numpy as np
import os
import re
from tqdm import tqdm
from nambu_keldysh_class import NambuKeldyshTensor, get_pauli_matrix, get_expansion_tensor
from numerical_support import ThermalConvolutions
from state_object_class import StateObject
from equilibrium_class import EquilibriumSolver
#~ import self_energy_class
import self_energy_keldysh_class
from self_energy_keldysh_class import EM_selfenergy

from matplotlib import pyplot as plt

class UsadelKeldyshEvolution:
    """
    Main evolution class for Usadel equation in Keldysh formalism.
    Computes equilibrium states and real-time dynamics.
    """

    #~ def __init__(self, grid_parameters, system_parameters, optimization_parameters=None, sigma_scatterings=None):
    def __init__(self, grid_parameters, system_parameters, optimization_parameters=None, sigma_scatterings=None, self_energy_dictionary=None):
        """
        Initialize Usadel evolution solver.
        Args:
            grid_parameters: dict with omega_sampling, cutoff, time_sampling, time_duration
            system_parameters: dict with critical_temperature, eta, etc.
            optimization_parameters: solver settings
            sigma_scatterings: dict of scattering mechanisms and rates
            self_energy_dictionary: {self-energy name: scattering rate} of the self-energies entering the time evolution, e.g. {'em': 1.0}.
                                    None defaults to {'em': 1.0}, {} means no self-energy contributions
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

        #* thermal distribution, its integral, sums and convolutions
        self.thermal = ThermalConvolutions(self.temperature, self.grid_parameters)

        #* self-energy contributions to the gr and gk equations
        self.self_energy_dictionary = self_energy_dictionary
        self.self_energy_contributions_list = []
        self.initialize_self_energies()

    # ========== Self-energy contributions ==========

    def initialize_self_energies(self):
        """Build self_energy_contributions_list from self_energy_dictionary ({name: scattering_rate})."""
        self_energy_registry = {'em': EM_selfenergy}

        self.self_energy_contributions_list = []
        for name, scattering_rate in self.self_energy_dictionary.items():
            if name not in self_energy_registry:
                raise ValueError(f"Unknown self-energy '{name}'. Available: {list(self_energy_registry.keys())}")
            self.self_energy_contributions_list.append(self_energy_registry[name](scattering_rate=scattering_rate))

    def _add_self_energy_terms(self, evolution_terms, g_type, state, params):
        """Add the equation coefficients of every self-energy to evolution_terms (g_type 'r' or 'k')."""
        for self_energy in self.self_energy_contributions_list:
            if g_type == 'r':
                terms = self_energy.r_equation_coefficients(state, params)
            else:
                terms = self_energy.k_equation_coefficients(state, params)

            for term_name, term_spec in terms.items():
                if term_name in evolution_terms:
                    raise ValueError(f"Self-energy '{self_energy.name}' term '{term_name}' already present in evolution_terms")
                evolution_terms[term_name] = term_spec

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
        Generate angular frequency (omega) grid as exact Fourier dual of EXTENDED time grid.

        For two-time functions g(t,t') with t,t' in [-T_max, 0], we need tau = t-t'
        in range [-T_max, T_max]. To avoid interpolation errors, we use an extended
        time grid [-T_max, T_max] for FFT, then extract [-T_max, 0] for evolution.

        Extended grid properties:
            - Evolution grid: [-T_max, 0] with N_t points, spacing dt = T_max/(N_t-1)
            - Extended FFT grid: [-T_max, T_max] with 2*N_t-1 points, SAME spacing dt
            - After inverse FFT, tau naturally ranges from -T_max to T_max
            - No interpolation needed - direct array indexing

        Stores:
            self.omega_grid: Angular frequency array (symmetric, -E_max to E_max)
            self.energy_cutoff: Maximum omega value (Nyquist frequency)
            self.d_omega: Omega spacing
            self.n_extended: Number of points in extended grid (2*N_t - 1)
        """

        # Extended grid for FFT: [-T_max, T_max] with same delta_t
        # Number of points: 2*T_max / delta_t + 1 = 2*(N_t-1) + 1 = 2*N_t - 1
        self.n_extended = 2 * self.ntpoints - 1

        # Get frequency bins from FFT using EXTENDED grid with same delta_t
        # np.fft.fftfreq gives frequencies f in cycles per unit time
        freq = np.fft.fftfreq(self.n_extended, d=self.delta_t)

        # Convert to angular frequency: ω = 2π*f
        omega = 2 * np.pi * freq

        # Shift to center around 0 (zero frequency in middle)
        # This gives symmetric grid: -E_max to E_max
        self.omega_grid = np.fft.fftshift(omega)

        # Energy cutoff is the maximum absolute omega value (Nyquist frequency)
        self.energy_cutoff = np.max(np.abs(self.omega_grid))

        # Store omega spacing and verify Fourier duality
        self.d_omega = self.omega_grid[1] - self.omega_grid[0]
        expected_product = 2 * np.pi / self.n_extended
        actual_product = self.d_omega * self.delta_t

        if not np.allclose(actual_product, expected_product, rtol=1e-10):
            print(f"WARNING: Fourier duality check failed!")
            print(f"  Expected: {expected_product:.10f}, Actual: {actual_product:.10f}")
            print(f"  d_omega = {self.d_omega:.10e}")
            print(f"  delta_t = {self.delta_t:.10e}")
            print(f"  n_extended = {self.n_extended}")

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

        #* gap of the frequency-domain solution (before the FFT to the two-time state), the gap read from the two-time gk differs from it at O(dt)
        self.equilibrium_gap = equilibrium_solver.gap_0

        # Compute equilibrium current using old solver's _get_current method
        # Need thermal distribution in frequency domain from old solver
        f_eq = equilibrium_solver._get_thermal_occupation(self.temperature)
        equilibrium_current = equilibrium_solver.usadel_solver._get_current(gr_eq, f_eq, Q)

        # Transform to two-time representation (returns only t,t' < 0)
        gr_two_time, gk_two_time, gr_tau, gk_tau = equilibrium_solver.fourier_transform_to_two_time(gr_eq, gk_eq)

        # Get BCS coupling constant for StateObject
        bcs_coupling = self._get_BCS_coupling() 
        bcs_coupling_prime = -1 / (1/2/np.pi * np.log(self.critical_temperature/self.temperature)) 

        # Create and return StateObject
        initial_state = StateObject(
            gr=gr_two_time,
            gk=gk_two_time,
            bcs_coupling_constant=bcs_coupling_prime,
            temperature=self.temperature,
            grid_params=self.grid_parameters
        )

        return initial_state, gr_tau, gk_tau, equilibrium_current

    # ---------- CN averaging functions (general, previously nested inside _compute_new_gk_complete) ----------

    def cn_derivative1_average(self, tensor, shift_index = 1):
        cn_derivative_factor = 1/2
        return cn_derivative_factor * (tensor[-1:] + tensor[-1:].shift(shift_index, axis=1)) - cn_derivative_factor * (tensor[-2:-1] + tensor[-2:-1].shift(shift_index, axis=1))
    def cn_derivative2_average(self, tensor, shift_index = 1):
        cn_derivative_factor = 1/2
        return cn_derivative_factor * (tensor[-1:] + tensor[-2:-1]) - cn_derivative_factor * ( tensor[-1:].shift(shift_index, axis=1) + tensor[-2:-1].shift(shift_index, axis=1))

    def cn_plaquette_average(self, tensor, shift_index = 1):
        """
        Computes CN average of a tensor over the four corners (t, t'), (t, t'+shift), (t-dt, t'), (t-dt, t'+shift),
        multiplied by delta_t. Note tensor should be of size 2 or more times N_t.
        shift_index: +1 for g^K, -1 for g^R (same convention as construct_discrete_operators)
        """
        cn_factor = 1/4 * self.delta_t
        result = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))
        for dt_prime_shift in [0, 1]:
            for dt_shift in [0,1]:
                dt_end = None if dt_shift == 0 else -dt_shift
                result += cn_factor * tensor[-1-dt_shift:dt_end,:].shift(dt_prime_shift * shift_index, axis=1)

        return result

    def two_row_source(self, row_function):
        """
        Stack row_function(dt_shift) for dt_shift = 1 (row t-dt) and dt_shift = 0 (row t) into a (2,2,2,Nt) tensor.
        This is the form a type0 term S needs: only the last two rows enter the CN average.
        """
        rows = [row_function(dt_shift) for dt_shift in (1, 0)]
        return NambuKeldyshTensor(np.concatenate([row.data for row in rows], axis=2))

    # ========== CN implementation of the operators from continous to discrete space ========== 

    def construct_discrete_operators(self, terms_dict, state, gap_tensor, g_type = 'r', additional_shift_index = 0):
        #* function assumes that all the terms are computed for the current computation time (t,t)
        #* assumes state last element is (t-dt,t-dt), which is true by default
        #* to capture the last element, we will have to add some corrections to the last element, so the output should also have an additional, diagonal_transpose output
        #* in the evolution the diagonal_transpose_output shuold be used with final solution. basically take all the terms that have gk_last_row (t-dt,t') that gets zeroed out 
        
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

        #evolution term shapes: type 0: S -- (2,2,k,ntpoints), k>=2, last two rows used (no g, CN averaged, rhs gets -CN(S))
        #                       type 1: L -- (2,2,ntpoints)
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
                0: {'S': two_time_shape},
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
                    raise ValueError(f"Unknown type number {type_num} in term '{term_name}'. Valid types: 0-8")

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
                    #~ if actual_shape != expected_shape:
                    #* type 0 only needs the last two rows (t-dt, t): (2,2,k,Nt) with k >= 2 is accepted
                    if type_num == 0:
                        shape_ok = (len(actual_shape) == 4 and actual_shape[:2] == (2, 2) and actual_shape[2] >= 2 and actual_shape[3] == self.ntpoints)
                    else:
                        shape_ok = (actual_shape == expected_shape)
                    if not shape_ok:
                        shape_desc = "single-time (2,2,Nt)" if len(expected_shape) == 3 else "two-time (2,2,Nt,Nt)"
                        if type_num == 0:
                            shape_desc = "(2,2,k,Nt) with k >= 2 (last two rows t-dt, t are used)"
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
        #~ expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)
        expansion_tensor = get_expansion_tensor(self.ntpoints)

        for type_key, term_spec in terms_dict.items():
            match = re.search(r'type(\d+)', type_key)
            if not match:
                raise ValueError(f"Invalid type key: {type_key}. Expected format: 'typeN' or 'typeN_description'")
            type_num = int(match.group(1))

            # ========== Type 0: Term without g, S(t,t') ==========
            if type_num == 0:
                #* F(t,t') = S(t,t'), CN averaged over the four blocks; same side as the other types so rhs gets the minus sign
                s_operator = term_spec['S']
                v_contribution = self.cn_plaquette_average(s_operator, shift_index=shift_index)

                if rhs_vector is None:
                    rhs_vector = -v_contribution
                else:
                    rhs_vector = rhs_vector - v_contribution

            # ========== Type 1: Left operator multiplication L(t)·g(t,t') ==========
            #~ if type_num == 1:
            elif type_num == 1:
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
                diagonal_term_factor_list.append((-cn_factor * l_operator[-2], tau0))

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
                diagonal_term_factor_list.append((-cn_factor * tau0, r_operator[-1]))

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
                #* the diagonal term history below (l_operator[-2:-1,:]) runs up to t'' = t-dt, which is the end point of F(t-δt, ·) and enters there with the full weight (δt/4),
                #* so this boundary correction must be -δt/8 to leave the end point weight 1/2
                #* the sign is positive here because the later history term is double counting the boundary already!
                #~ diagonal_term_factor_list.append((-boundary_factor * l_operator[-2, -2], tau0))
                diagonal_term_factor_list.append((boundary_factor * l_operator[-2, -2], tau0))

                #* Interior sums from F(t-δt, ·): L(t-δt, :) @ g, sum to t-2δt
                #* the g_matrix starts from 1 since we assume that the g_matrix corresponds to old, so last element is t-dt and last element of L operator is t and then we remove one extra last element due to trapezoid rule
                #* no midpoint 1/2 factors because the last element contributes to the diagonal itself and that element has 1/2

                v_old_interior_1 = interior_factor * (l_operator[-2:-1, :-2] @ (g_matrix[1:-1, :].shift(-1, axis=1) + g_matrix[1:-1, :].shift(shift_index-1, axis=1)))

                #* diagonal term convolution since convolving with g_matrix which got shifted by -1
                #~ diagonal_term_history_list += [(interior_factor * l_operator[-2:-1,:],tau0)]
                diagonal_term_history_list += [(-interior_factor * l_operator[-2:-1,:],tau0)]

                #* Interior sums from F(t, ·): L(t, :) @ g, sum to t-δt
                #* this summation goes until t-dt so we sum over all elements since the last one corresponds to that time label, the g_matrix starts from 1 since we assume that the g_matrix corresponds to old
                v_old_interior_3 = interior_factor * (l_operator[-1:, :-1] @ (g_matrix[1:].shift(-1, axis=1) + g_matrix[1:].shift(shift_index-1, axis=1)))
                
                #* diagonal term convolution since convolving with g_matrix which got shifted by -1
                diagonal_term_history_list += [(-interior_factor * l_operator[-1:,],tau0)]

                v_old_contribution = (v_old_boundary_1 + v_old_interior_1 + v_old_interior_3)

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
                #* the interior sum below already contains the diagonal R(t',t') with the full weight (δt/4), so the boundary term must be -δt/8 to leave the end point weight 1/2
                #~ v_old_boundary_1 = boundary_factor * g_last_row.shift(-1, axis=1) * r_diagonal
                #~ v_old_boundary_2 = boundary_factor * g_last_row.shift(shift_index-1, axis=1) * r_diagonal.shift(shift_index, axis=0)
                v_old_boundary_1 = -boundary_factor * g_last_row.shift(-1, axis=1) * r_diagonal
                v_old_boundary_2 = -boundary_factor * g_last_row.shift(shift_index-1, axis=1) * r_diagonal.shift(shift_index, axis=0)

                #* diagonal contribution to gk from since shifting g_last_row by -1
                diagonal_term_factor_list.append((-boundary_factor * tau0, r_diagonal[-1]))

                #* Interior sums: (δt/4)·Σ_{t''} g(t-δt, t'')·R(t'', t')
                #*  since g is computed using old basis and r is in the new basis, they need to be relatively shifted w.r.t. eachother when summing, last element is t-dt in g and t,.. element is excluded from r term
                #* 1/2 weight removed because the diagonal term is used anyway.
                v_old_interior_1 = interior_factor * (g_last_row[:,1:] @ (r_operator[:-1,:] + r_operator[:-1,:].shift(shift_index, axis=1)))  
                
                v_old_contribution = v_old_boundary_1 + v_old_boundary_2 + v_old_interior_1

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector - v_old_contribution

                #* V_crt_diag contribution: (δt/8)·g(t, t'+shift)·R(t'+shift, t'+shift)
                #* here we shift since we are interested in previous term
                #* the history convolution below includes the diagonal of the shifted column R(t'+shift, t'+shift) with the full weight, so this term gets the opposite sign
                #~ rhs_vector_factor_list.append((-tau0, boundary_factor * r_diagonal.shift(shift_index, axis=0)))
                rhs_vector_factor_list.append((tau0, boundary_factor * r_diagonal.shift(shift_index, axis=0)))

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

                #* the lower end point t''=t' is also contained in v_old_interior_1 below: g(t'',t') is retarded, so the product picks up t''=t' (g(t',t') = -Δ(t')) with the full weight (δt/4).
                #* The lower boundary terms are therefore -δt/8 to leave the end point weight 1/2. For t' = t-dt both end points coincide (zero length interval) and the two boundary terms cancel
                
                v_old_lower_boundary = -boundary_factor * l_operator[-2] * g_last_row.shift(-1, axis=1) * m_operator * (g_diagonal) * r_operator

                v_old_lower_boundary_shift = -boundary_factor * l_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * m_operator.shift(shift_index, axis=0) * (g_diagonal.shift(shift_index, axis=0)) * r_operator.shift(shift_index, axis=0)

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
                #* lower end point of F(t, t'+δt): the history list below starts at t'' = t'+δt and contains it with the full weight, so this term gets the opposite sign
                rhs_vector_factor_list.append((boundary_factor * l_operator[-1], m_operator.shift(shift_index, axis=0) * (g_diagonal.shift(shift_index, axis=0) * r_operator.shift(shift_index, axis=0))))

                #* Term 2: -δt/8 L(t)·Δ(t)·M(t)·g(t,t'+δt)·R(t'+δt)
                rhs_vector_factor_list.append((-boundary_factor * l_operator[-1] * (g_diagonal_current) * m_operator[-1], r_operator.shift(shift_index, axis=0)))
               
                #* V_crt_conv: Interior bilinear sums
                #*Term 1: δt/4 L(t) Σ_{t''=t'+δt}^{t-δt} g(t,t'')·M(t'')·g(t'',t')·R(t')
                #* we need to shift the g_matrix operator because the time indicies are shifted by 1 in the passed g_matrix in t' we also shift the other axis since we want the operator to have the proper time indexing w.r.t. new time indices
                #* M(t'') sits at the integration variable t'' (the row of the history), not at the column t'+δt: it must not be shifted in the second term
                rhs_vector_history_list.append(( -interior_factor * l_operator[-1], m_operator * g_matrix.shift(-1, axis=0).shift(-1, axis=1) * r_operator + m_operator * g_matrix.shift(-1, axis=0).shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))

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
                #~ diagonal_term_factor_list.append((boundary_factor * l_operator[-2, -2], r_operator[-1]))
                #~ diagonal_term_factor_list.append((-boundary_factor * l_operator[-2, -2], r_operator[-1]))
                #* the diagonal term history below (l_operator[-2:-1,:]) runs up to t'' = t-dt, the end point of F(t-δt, ·), with the full weight (δt/4): the boundary correction must be -δt/8
                diagonal_term_factor_list.append((boundary_factor * l_operator[-2, -2], r_operator[-1]))

                #*From F(t-δt, ·): L(t-δt, t_init)·g(t_init, t')·R(t')

                v_old_interior_1 = interior_factor * (l_operator[-2:-1, :-2] @ (g_matrix[1:-1, :].shift(-1, axis=1) * r_operator + g_matrix[1:-1, :].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))

                diagonal_term_history_list += [(-interior_factor * l_operator[-2:-1,:],r_operator[-1])]

                #* Interior sums from F(t, ·): L(t, :) @ (g·R), sum to t-δt (Eq. 509)

                v_old_interior_3 = interior_factor * (l_operator[-1:, :-1] @ (g_matrix[1:].shift(-1, axis=1) * r_operator + g_matrix[1:, :].shift(shift_index-1, axis=1) * r_operator.shift(shift_index, axis=0)))

                diagonal_term_history_list += [(-interior_factor * l_operator[-1:,:],r_operator[-1])]

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

                #* same as type 4: the interior sum contains the diagonal R(t',t') with the full weight (δt/4), so the boundary terms are -δt/8
                v_old_boundary_1 = -boundary_factor * l_operator[-2] * g_last_row.shift(-1, axis=1) * r_diagonal
                v_old_boundary_2 = -boundary_factor * l_operator[-2] * g_last_row.shift(shift_index-1, axis=1) * r_diagonal.shift(shift_index, axis=0)

                #* diagonal gk contribution since referencing g shifted by -1
                diagonal_term_factor_list.append((-boundary_factor * l_operator[-2], r_diagonal[-1]))
            
                #* Interior sums: (δt/4)·L(t-δt)·Σ_{t''=t_init+δt}^{t'} g(t-δt, t'')·R(t'', t') (Eq. 566-567)
                v_old_interior_1 = interior_factor * l_operator[-2] * (g_last_row[:,1:] @ (r_operator[:-1] +  r_operator[:-1].shift(shift_index, axis=1)))

                v_old_contribution = (v_old_boundary_1 + v_old_boundary_2 + v_old_interior_1)

                if rhs_vector is None:
                    rhs_vector = -v_old_contribution
                else:
                    rhs_vector = rhs_vector -v_old_contribution

                #* δt/8 L(t)·g(t,t'+δt)·R(t'+δt,t'+δt)
                #* the history list below contains the diagonal of the shifted column with the full weight, so this term gets the opposite sign (same as type 4)
                #~ rhs_vector_factor_list.append((-boundary_factor * l_operator[-1], r_diagonal.shift(shift_index, axis=0)))
                rhs_vector_factor_list.append((boundary_factor * l_operator[-1], r_diagonal.shift(shift_index, axis=0)))

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
                diagonal_term_factor_list.append((-cn_factor * l_operator[-2], r_operator[-1]))

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

    # ========== Constraint equations  ==========  

    def get_gr_constraint(self, state, gap_tensor):
        """
        Construct operators for retarded normalization constraint using MIDPOINT RULE.
        Args:
            state: StateObject containing g^R data
            gap_tensor: Gap function as NambuKeldyshTensor

        Returns:
            6-tuple: (left_matrix, right_matrix, rhs_vector,
                      rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices)
        """

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        #~ expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)
        expansion_tensor = get_expansion_tensor(self.ntpoints)

        gr = state.gr
        gr_last_row = gr[-1:, :]

        # Operator matrices (anticommutator structure: τ₃·g + g·τ₃)
        # using the fact that gr diagonal is the  -gap 
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
        # Missing far history endpoint corrections -- ignored in most places in the code, since zero anyway        

        rhs_vector_history_list = [(-tau0, gr.shift(-1, axis=1).shift(-1, axis=0) * self.delta_t)]

        # No diagonal coupling terms for this constraint
        rhs_vector_factor_list = []

        # No sandwich terms (convolution handled via history list)
        g_sandwich_matrices = []

        return (left_matrix, right_matrix, rhs_vector, rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices)

    def get_gk_constraint(self, state, gap_tensor):
        """
        Construct operators for Keldysh constraint equation using midpoint rule.
        Args:
            state: StateObject containing g^R and g^K data
            gap_tensor: Gap function as NambuKeldyshTensor

        Returns:
            6-tuple: (left_matrix, right_matrix, rhs_vector,
                      rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices)
        """

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        #~ expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)
        expansion_tensor = get_expansion_tensor(self.ntpoints)
        #~ full_expansion_tensor = NambuKeldyshTensor(np.ones((1,self.ntpoints)), pauli_channel=0)
        full_expansion_tensor = get_expansion_tensor((1, self.ntpoints))

        diag_g_factor_list = []
        diag_g_history_list = []

        gr = state.gr
        gk = state.gk
        ga = state._r2a()  # g^A = -(g^R)†

        # by this point gr and ga are already updated, so we can use them directly, we dont need to use the gap values
        left_matrix = tau3 * expansion_tensor + self.delta_t * gr[-1,-1] * expansion_tensor/2

        right_matrix = -tau3 * expansion_tensor + self.delta_t * ga.diagonal_time()/2

        thermal_gap_term = self.thermal.thermal_gap_term(gap_tensor)


        gr_last_row = gr[-1:, :]
        gk_full = gk

        #* this assumes that gr_last_row has correct time index and not shifted t' index, midpoint rule automatically applied since its the other edge term which has gap/2
        #* shift of gk because its corresponding to old timestamps
        rhs_term_1 =  -self.delta_t * (gr_last_row[:,:-1] @ gk_full[1:]).shift(-1, axis = 1) #+ self.delta_t * 0.5 * gr_last_row[:,0:1] * gk_full[0:1,:].shift(-1, axis = 1)

        diag_g_history_list += [(-gr_last_row * self.delta_t, tau0)] 

        #* f- and F-parts (with 2*tau3 and log regularization) from ThermalConvolutions; usadel RHS carries the opposite sign
        rhs_term_1 += -(self.thermal.convolve_thermal_full_left(gr_last_row, gap_tensor, -1) + self.thermal.convolve_thermal_full_right(ga, gap_tensor, -1))

        # new term coming from the commutator
        rhs_term_1 += -(tau3 * thermal_gap_term[-1:,:] - thermal_gap_term[-1:,:] * tau3)  
        
        rhs_vector = rhs_term_1 

        # This convolution is handled via history list: (tau0 * gk_current) @ ga
        #* this ga is now in good frame
        rhs_vector_history_list = [(-tau0 , ga * self.delta_t)]
        # No diagonal coupling terms
        rhs_vector_factor_list = []

        # No sandwich terms for this constraint
        g_sandwich_matrices = []

        return (left_matrix, right_matrix, rhs_vector, rhs_vector_history_list, rhs_vector_factor_list, g_sandwich_matrices, diag_g_factor_list, diag_g_history_list)

    # ========== Computation of new gr by matrix inversion ========== 

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

        # Using cyclicity of trace
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
                        #* first element is skipped because in that case the convolution is over the empty list
                        #* last element in right_term corresponds to time t actually so it should be summed with solution tensor last element
                        #* First element of the solution tensor corresponds to time+1 index actually, that is why the sum is correct
                        convolution_term_1 += (left_term * solution_tensor[:-1]) @ right_term[time+1:-1, time]
                    elif g_type == 'k':
                        #* note, last right term is actually time t as last time index and last solution tensor is that as well?
                        #* the sum goes until time which means last element is t'-dt' which it should be summed fulled last one is giving the diagonal
                        #if time != loop_end - loop_step:
                        #* in principle first solution corresponds to t, last time is n_points which ends with t-dt, as it should
                        #* solution tensor last element is time-1, as is in the right term
                        convolution_term_1 += (left_term * solution_tensor[1:]) @ right_term[:time, time] 

                for terms in rhs_vector_history_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    if g_type == 'r':
                        convolution_term_2 += (left_term * solution_tensor[:-1]) @ right_term[time+1:-1, time]
                    elif g_type == 'k':
                        #if time != loop_end - loop_step:
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
                    convolution_term_1 += left_term * previous_solution.involution() * right_term

                for terms in diagonal_term_factor_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    convolution_term_2 += left_term * previous_solution.involution() * right_term

                for terms in diagonal_term_history_1_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    #* last term in solution tensor is t-1 same as in left term
                    convolution_term_1 += left_term[-1,:-1] @ (solution_tensor[1:].involution() * right_term)

                for terms in diagonal_term_history_2_list:
                    left_term = terms[0]
                    right_term = terms[1]
                    convolution_term_2 += left_term[-1,:-1] @ (solution_tensor[1:].involution() * right_term)

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
            #* diagonal is handled separately for legacy reasons
            return solution_tensor[:-1]
        elif g_type == 'k':
            return solution_tensor

    # ========== Update rules for gr and gk ==========

    #TODO: Remains to be fully checked 
    def _compute_new_gr_row(self, state, gap_history, A_history=None):
        """
        Evolve retarded Green's function gr by one timestep using Crank-Nicolson discretization.

        Computes g^R(t_{time_index}, t_j) for all j < time_index using the
        discretized Usadel equation with CN averaging over 4 time corners.

        Args:
            state: StateObject with current gr data
            gap_history: Gap Δ on the NEW time grid, last entry is Δ(t) at the new time
            A_history: Optional external vector potential history

        Returns:
            new_gr_row: NambuKeldyshTensor for g^R(t_new, :)
            gr_diagonal_new: Boundary condition g^R(t,t) = -Δ(t)

        Called by:
            - _evolve_state_by_one_timestep()
        """
        # ========== 1. Extract parameters ==========

        tau0 = NambuKeldyshTensor(1.0, pauli_channel=0)
        tau1 = NambuKeldyshTensor(1.0, pauli_channel=1)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)

        #~ expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)
        expansion_tensor = get_expansion_tensor(self.ntpoints)

        # ========== 2. Impose boundary condition for gr ==========
        gr_diagonal_new = -gap_tensor[-1]

        # ========== 3. Build evolution equation via Type classification ==========
        # NOTE: delta_t factor is applied later in construct_discrete_operators

        evolution_terms = {
            'type1_gap': {'L': -1j * gap_tensor},
            'type2_gap': {'R': 1j * gap_tensor},
            'type1_damping': {'L': 1j * self.eta * tau3 * expansion_tensor},
            'type2_damping': {'R': -1j * self.eta * tau3 * expansion_tensor},
        }  

        #* EM and other self-energy terms come from self.self_energy_contributions_list
        self_energy_params = {'A_history': A_history, 'thermal': self.thermal, 'gap_tensor': gap_tensor}
        self._add_self_energy_terms(evolution_terms, 'r', state, self_energy_params)

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

        #* the plaquette of the entry t'=t-dt has its centre on the jump of g^R at t=t': the corners (t,t) and (t-dt,t-dt) of the local terms (types 1, 2, 8) enter with theta(0)=1/2,
        #* construct_discrete_operators weights them with 1, so 1/2 of their (1/4 dt) contribution is added back here
        #* gap terms cancel since [Delta, g(t,t)] = 0, the damping and EM local terms do not
        gr_diagonal_old = state.gr[-1, -1]
        diagonal_corner_sum = None
        for term_name, term_spec in evolution_terms.items():
            term_type = int(re.search(r'type(\d+)', term_name).group(1))
            if term_type == 1:
                left_op = term_spec['L']
                corner = left_op[-1] * gr_diagonal_new + left_op[-2] * gr_diagonal_old
            elif term_type == 2:
                right_op = term_spec['R']
                corner = gr_diagonal_new * right_op[-1] + gr_diagonal_old * right_op[-2]
            elif term_type == 8:
                left_op, right_op = term_spec['L'], term_spec['R']
                corner = left_op[-1] * gr_diagonal_new * right_op[-1] + left_op[-2] * gr_diagonal_old * right_op[-2]
            else:
                continue
            diagonal_corner_sum = corner if diagonal_corner_sum is None else diagonal_corner_sum + corner

        if diagonal_corner_sum is not None:
            first_entry = NambuKeldyshTensor([np.append(np.zeros(self.ntpoints-1),[1.0])], pauli_channel=0).shift(-1, axis=1)
            V1 = V1 + (self.delta_t / 8.0) * diagonal_corner_sum * first_entry

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

    #TODO: Remains to be fully checked 
    def _compute_new_gk_complete(self, state, gap_history, A_history=None):
        """
        Evolve Keldysh Green's function g^K by one timestep using Crank-Nicolson discretization.

        Computes g^K(t, t_j) for ALL j including diagonal using CN averaging over 4 time corners.
        Includes thermal collision integrals, electromagnetic coupling, and diagonal corrections.
        Now unified: computes both off-diagonal AND diagonal in single solver call.

        Args:
            state: StateObject with current gr and gk data
            gap_history: Gap Δ on the NEW time grid, last entry is Δ(t) at the new time
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

        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        if A_history is None:
            A_history = np.zeros(len(gap_history), dtype=complex)
       
        #~ expansion_tensor = NambuKeldyshTensor(np.ones(self.ntpoints), pauli_channel=0)
        expansion_tensor = get_expansion_tensor(self.ntpoints)

        # ========== 1.5: Build Type 3 and Type 4 Electromagnetic Operators ==========
        
        # ========== 2. Build evolution equation via Type classification ==========
        # NOTE: delta_t factor is applied later in construct_discrete_operators
        evolution_terms = {
            'type1_gap': {'L': -1j * gap_tensor},
            'type2_gap': {'R': 1j * gap_tensor},
            'type1_damping': {'L': 1j * self.eta * tau3 * expansion_tensor},
            'type2_damping': {'R': 1j * self.eta * tau3 * expansion_tensor},
        }
    
        # ---------- Source terms without g, passed as type0 (rows t-dt and t only) ----------
        #* type0 puts -CN_plaquette(S) on the rhs, the old code added +source to V1, hence S = -source
        def dynes_row(dt_shift):
            dt_end = None if dt_shift == 0 else -dt_shift
            return -2j * self.eta * (tau3 * self.thermal.convolve_thermal_dist_right(ga, time_index=-1-dt_shift) - self.thermal.convolve_thermal_dist_left(gr[-1-dt_shift:dt_end,:], time_index=-1-dt_shift) * tau3)

        def gap_f_row(dt_shift):
            dt_end = None if dt_shift == 0 else -dt_shift
            f_row = self.thermal.thermal_dist[-1-dt_shift:dt_end,:]
            return -2 * (-1j * gap_tensor[-1-dt_shift] * tau3 * f_row + 1j * tau3 * f_row * gap_tensor)

        thermal_integral_rows = self.thermal.thermal_integral[-2:, :]
        new_source_rows = -1j * (gap_tensor[-2:] * gap_tensor[-2:] * thermal_integral_rows - thermal_integral_rows * gap_tensor * gap_tensor)

        evolution_terms['type0_dynes'] = {'S': -self.two_row_source(dynes_row)}
        evolution_terms['type0_gap_f'] = {'S': -self.two_row_source(gap_f_row)}
        evolution_terms['type0_new_source'] = {'S': -new_source_rows}

        #* EM and other self-energy terms (including the g-independent type0 sources) come from self.self_energy_contributions_list
        self_energy_params = {'A_history': A_history, 'thermal': self.thermal, 'gap_tensor': gap_tensor}
        self._add_self_energy_terms(evolution_terms, 'k', state, self_energy_params)

        L1, R1, V1, Vhist1, Vfact1, sandwich1, diag_factor_list_1, diag_hist_list_1 = self.construct_discrete_operators(evolution_terms, state, gap_tensor, g_type='k')

        # ========== 3. Add derivative corrections ==========

        L1 = L1 + (1j/2) * tau3 * expansion_tensor
        R1 = R1 + (1j/2) * tau3 * expansion_tensor

        Vfact1.append((-(1j/2) * tau3, tau0 * expansion_tensor))
        Vfact1.append((tau0, (1j/2) * tau3 * expansion_tensor))

        # Every term has to be shifted by -1, but then the ones from t'-dt dont get shifted    
        v_old_deriv = ((1j/2) * tau3 * gk_last_row.shift(-1, axis=1) - (1j/2) * gk_last_row.shift(-1, axis=1) * tau3
                       + (1j/2) * tau3 * gk_last_row + (1j/2) * gk_last_row * tau3)

        # It should mimic the v_old_deriv not the vfact1! the one that got shifted
        diag_factor_list_1 += [((1j/2) * tau3, tau0), (tau0, -(1j/2) * tau3)]

        V1 = V1 + v_old_deriv

        thermal_term = NambuKeldyshTensor(np.zeros((2, 2, 1, self.ntpoints), dtype=complex))

        shifted_derivative_term = (- gap_tensor * self.thermal.thermal_integral - self.thermal.thermal_integral * gap_tensor) 

        thermal_term += -(1j * tau3 * self.cn_derivative1_average(shifted_derivative_term) + self.cn_derivative2_average(shifted_derivative_term) * tau3 * 1j)

        V1 = V1 + thermal_term

        L2, R2, V2, Vhist2, Vfact2, sandwich2, diag_factor_list_2, diag_hist_list_2 = self.get_gk_constraint(state, gap_tensor)

        # ========== 6. Call unified solver with diagonal corrections ==========

        gk_boundary = ( 2 * state.gk[-1, 0] - state.gk[-2, 0]) # g^K(t, -infty)

        gk_new = self.generalized_g_update_rule( g_type='k',diagonal_entry=gk_boundary, left_matrix_1=L1, left_matrix_2=L2,right_matrix_1=R1, right_matrix_2=R2,
            rhs_vector_1=V1, rhs_vector_2=V2, rhs_vector_history_1_list=Vhist1, rhs_vector_history_2_list=Vhist2, rhs_vector_factor_1_list=Vfact1,
            rhs_vector_factor_2_list=Vfact2, g_sandwich_matrices=sandwich1, diagonal_term_factor_1_list=diag_factor_list_1, diagonal_term_factor_2_list=diag_factor_list_2,
            diagonal_term_history_1_list=diag_hist_list_1, diagonal_term_history_2_list=diag_hist_list_2)

        # Extract diagonal element from unified result
        gk_diagonal_new = gk_new[-1]
        #gk_diagonal_new = state.gk[-1,-1]
        return gk_new[:-1], gk_diagonal_new

    # ========== Updating the state ==========

    def _evolve_trial_state(self, state, gap_history, A_external):
        """
        Evolve a copy of the state by one timestep for a given gap history on the new time grid.
        The original state is left untouched.

        Returns:
            trial_state: StateObject with updated g^R and g^K
        """
        trial_state = state.copy()
        new_gr_row, new_gr_diag = self._compute_new_gr_row(trial_state, gap_history, A_history=A_external)
        trial_state.update_state_gr(new_gr_row, new_gr_diag)
        new_gk_row, new_gk_diag = self._compute_new_gk_complete(trial_state, gap_history, A_history=A_external)
        trial_state.update_state_gk(new_gk_row, new_gk_diag)
        return trial_state

    #~ def _evolve_state_by_one_timestep(self, state, A_external=None, gap_tolerance=1e-5, max_gap_iterations=10):
    def _evolve_state_by_one_timestep(self, state, A_external=None, gap_tolerance=1e-5, max_gap_iterations=25, circuit_params=None, driving_field_current=None, driving_field_next=None, current_no_gradient=None, vector_potential_tolerance=1e-3):
        
        """
        Evolve state by one timestep using Crank-Nicolson discretization, with a self-consistent gap Δ(t).

        The new gap Δ(t) enters its own update (thermal_gap_term, g^R diagonal, gap terms), with a gain > 1,
        so using the old gap explicitly is unstable. Instead Δ(t) is iterated until the gap put into the
        step equals the gap read off the new g^K diagonal:
        1. Guess Δ(t) by linear extrapolation of the previous gaps
        2. Evolve a trial copy of the state with the gap history on the new grid (old gaps shifted, guess last)
        3. Compare the resulting gap with the guess; update the guess with the secant method
        4. Once converged, copy the trial g^R and g^K into the state

        Args:
            state: StateObject with current data (modified in place)
            A_external: Optional external vector potential history
            gap_tolerance: Convergence tolerance on |Δ_out - Δ_guess|
            max_gap_iterations: Maximum number of trial steps
            circuit_params: If given (mode 2), gap and vector potential are solved self-consistently at the new time
                            (see _evolve_state_gap_and_vector_potential); the arguments below are only used then
            driving_field_current, driving_field_next: I_in at the current and next timestep
            current_no_gradient: J(t-dt) without gradient term, computed before the step
            vector_potential_tolerance: Convergence tolerance on |A_out - A_guess|

        Returns:
            gap_new: Gap value at new time t
            current_new: Current at the evolved time (mode 1: J(t-dt) before the step, mode 2: J(t) after the step)
            vector_potential_new: Vector potential (mode 1: A_external[-1]; mode 2: converged A(t))

        Calls:
            - _evolve_trial_state(state, gap_history, A_external)
        """
        #* mode 2: gap and vector potential are solved together at the new time
        if circuit_params is not None:
            return self._evolve_state_gap_and_vector_potential(state, A_external, circuit_params, driving_field_current, driving_field_next, current_no_gradient, gap_tolerance, vector_potential_tolerance, max_gap_iterations)

        current_new = state.get_current_at_time_t(A_external, self.thermal)

        # Step 1: guess Δ(t) by linear extrapolation, old_gap is on the old grid (last entry is Δ(t-dt))
        old_gap = state.get_gap_history()
        gap_guess = 2 * old_gap[-1] - old_gap[-2]
        previous_guess, previous_residual = None, None

        # Step 2: iterate trial steps until the gap put in equals the gap coming out
        for iteration in range(max_gap_iterations):
            #* gap on the new grid: old gaps shifted by one index, guess as Δ(t)
            gap_history = np.append(old_gap[1:], gap_guess)
            trial_state = self._evolve_trial_state(state, gap_history, A_external)
            residual = trial_state.get_gap_history()[-1] - gap_guess

            if np.abs(residual) < gap_tolerance:
                break

            #* first iteration: take the output gap as second point; afterwards secant (plain fixed point diverges, gain > 1)
            if previous_residual is None:
                next_guess = gap_guess + residual
            else:
                next_guess = gap_guess - residual * (gap_guess - previous_guess) / (residual - previous_residual)
            previous_guess, previous_residual = gap_guess, residual
            gap_guess = next_guess
        else:
            print(f"WARNING: gap not self-consistent after {max_gap_iterations} iterations, |residual| = {np.abs(residual):.2e}")

        # Step 3: accept the trial Green's functions (state object is updated in place)
        state.gr = trial_state.gr
        state.gk = trial_state.gk

        # Step 2.5: Update occupation function if tracking is enabled
        if state.occupation_function is not None:
            state.update_state_occupation(self.thermal)

        # Step 3: Extract observables
        gap_history = state.get_gap_history()
        gap_new = gap_history[-1]

        # Current is zero for now (Stage 2 of project)
        if A_external is None:
            vector_potential_new = 0.0
        else:
            vector_potential_new = A_external[-1]

        return gap_new, current_new, vector_potential_new

    #TODO: Remains to be fully checked 
    def _evolve_state_gap_and_vector_potential(self, state, A_external, circuit_params, driving_field_current, driving_field_next,
                                               current_no_gradient, gap_tolerance, vector_potential_tolerance, max_iterations):
        """
        Evolve state by one timestep (mode 2) solving Δ(t) and A(t) self-consistently.

        Each pass evolves a trial copy with the guesses (Δ_guess, A_guess) on the new grid, then reads
        Δ_out from the new g^K diagonal and A_out from the Crank-Nicolson update evaluated on the trial state.
        Both guesses start from linear extrapolation. A is real, so the unknowns x = (Re Δ, Im Δ, A) are updated jointly with a
        Broyden secant step on the residual r = out - guess (first pass: fixed-point step, Jacobian of r taken as -1);
        for the gap alone this is the secant step of _evolve_state_by_one_timestep.

        Returns:
            gap_new, current_new (J(t) after the step), vector_potential_new (converged A(t))
        """
        old_gap = state.get_gap_history()
        gap_guess = 2 * old_gap[-1] - old_gap[-2]
        A_guess = 2 * np.real(A_external[-1]) - np.real(A_external[-2])

        unknowns = np.array([np.real(gap_guess), np.imag(gap_guess), A_guess])
        jacobian = -np.eye(3)
        previous_unknowns, previous_residual = None, None

        for iteration in range(max_iterations):
            gap_guess = complex(unknowns[0], unknowns[1])
            A_guess = unknowns[2]

            gap_history = np.append(old_gap[1:], gap_guess)
            A_trial = np.append(A_external[1:], A_guess)
            trial_state = self._evolve_trial_state(state, gap_history, A_trial)

            gap_residual = trial_state.get_gap_history()[-1] - gap_guess
            A_out = np.real(self.update_vector_potential(old_vector_potential=A_external, driving_field_current=driving_field_current, driving_field_next=driving_field_next, current=current_no_gradient, circuit_params=circuit_params, state=trial_state)[-1])
            A_residual = A_out - A_guess

            if np.abs(gap_residual) < gap_tolerance and np.abs(A_residual) < vector_potential_tolerance:
                break

            residual = np.array([np.real(gap_residual), np.imag(gap_residual), A_residual])

            if previous_residual is not None:
                step = unknowns - previous_unknowns
                step_norm = step @ step
                if step_norm > 0:
                    jacobian = jacobian + np.outer((residual - previous_residual) - jacobian @ step, step) / step_norm

            previous_unknowns, previous_residual = unknowns.copy(), residual
            try:
                unknowns = unknowns - np.linalg.solve(jacobian, residual)
            except np.linalg.LinAlgError:
                unknowns = unknowns + residual
        else:
            print(f"WARNING: gap and vector potential not self-consistent after {max_iterations} iterations, "
                  f"|gap residual| = {np.abs(gap_residual):.2e}, |A residual| = {np.abs(A_residual):.2e}")

        state.gr = trial_state.gr
        state.gk = trial_state.gk

        if state.occupation_function is not None:
            state.update_state_occupation(self.thermal)

        gap_new = state.get_gap_history()[-1]
        vector_potential_new = A_guess
        current_new = state.get_current_at_time_t(np.append(A_external[1:], vector_potential_new), self.thermal)

        return gap_new, current_new, vector_potential_new

    #TODO: Remains to be fully checked
    def update_vector_potential(self, old_vector_potential, driving_field_current=None, driving_field_next=None, current=None, kernel_prefactor=None, circuit_params=None, state=None):
        """
        Update vector potential using sliding window or Crank-Nicolson scheme.

        Two modes:
        1. Simple sliding window (circuit_params=None):
           Append new driving_field_current value (interpreted as A(t))

        2. Crank-Nicolson update (circuit_params provided):
           Solve Eq. (262) from effective_circuit_model.tex for A'_(n+1)
           Uses input current I_in(t) from driving_field_current and driving_field_next

        Args:
            old_vector_potential: Current A_external array (length N_t)
            driving_field_current: Field value at current timestep
                                   - If circuit_params=None: interpreted as A(t)
                                   - If circuit_params provided: interpreted as I_in(n)
            driving_field_next: Field value at next timestep (only used if circuit_params provided)
                                Interpreted as I_in(n+1)
            current: Current J(t) at this timestep (required if circuit_params provided)
            kernel_prefactor: Kernel trace tr(K_(t,t)) (required if circuit_params provided)
            circuit_params: Dict with Z_T, R_n, J_DP_prime, I_DP (optional)
            state: StateObject (required if circuit_params provided, for kernel and current computation)

        Returns:
            new_vector_potential: Updated array (length N_t)
        """

        # Mode 1: Simple sliding window (current behavior)
        if circuit_params is None:
            if driving_field_current is None:
                return old_vector_potential

            new_field_value = driving_field_current
            new_vector_potential = np.append(old_vector_potential[1:], new_field_value)
            return new_vector_potential

        # Mode 2: Crank-Nicolson update for current-driven evolution
        else:
            # Extract circuit parameters
            z_t = circuit_params['Z_T']
            r_n = circuit_params['R_n']
            j_dp_prime = circuit_params['J_DP_prime']
            r_contact = circuit_params['R_c'] if 'R_c' in circuit_params else 0.0

            #~ kernel_prefactor = state.get_current_kernel_prefactor(self.thermal_dist)
            kernel_prefactor = state.get_current_kernel_prefactor(self.thermal)
            A_external_newest = np.append(old_vector_potential[1:], 0.0)

            #~ current_newest = state.get_current_at_time_t(A_external_newest, self.thermal_dist, self.thermal_integral,
                                                   #~ thermal_sum_left=self.thermal_sum_left,
                                                   #~ thermal_sum_right=self.thermal_sum_right, include_derivative = False)
            current_newest = state.get_current_at_time_t(A_external_newest, self.thermal, include_derivative = False)

            # Get current I_in values at current and next timesteps (for Crank-Nicolson midpoint)
            i_in_current = driving_field_current if driving_field_current is not None else 0.0
            i_in_next = driving_field_next if driving_field_next is not None else i_in_current

            # Get current J'(t) - this is passed from the evolution
            j_prime_current = current if current is not None else 0.0

            j_prime_hist = j_prime_current

            dt = self.delta_t
            
            lhs_coeff = 1.0  - np.real(  ((2.0 * z_t + r_contact) / (2.0 * r_n)) * (dt**2) * kernel_prefactor)/(1 + (2.0 * z_t + r_contact)/ (r_n))

            # Right-hand side
            a_n = old_vector_potential[-1]  # Current value A'_n
            #* the last division term comes from gradient exclusion in current expression
            current_term = np.real(dt *j_dp_prime * ( ((2.0 * z_t + r_contact)/ r_n) *(j_prime_hist + current_newest)/2.0/j_dp_prime - (2.0 * z_t / r_n) *(i_in_current + i_in_next)/2.0 ))/(1 + (2.0 * z_t  + r_contact)/ (r_n))
            rhs = (a_n + current_term)/ lhs_coeff
            #TODO: scale the output of the current by critical current! and check all the factors!!
            # Update vector potential history using sliding window
            new_vector_potential = np.append(old_vector_potential[1:], rhs)

            return new_vector_potential

    # ========== Real-time evolution ==========

    def real_time_evolution(self, initial_state, num_timesteps, driving_field=None, circuit_params=None, track_occupations=False, track_state=False, track_state_every_n=1):
        """
        Main real-time evolution loop.

        Evolves state forward in time using sliding window for vector potential.
        At each timestep, the driving field value is appended to the vector potential
        history and the oldest value is removed.

        Steps:
        1. Initialize A_external as zeros (size N_t from initial_state)
        2. For each timestep:
            a. Update A_external using sliding window with new driving_field value
               (or Crank-Nicolson update if circuit_params provided)
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
                           - If circuit_params=None: interpreted as A(t)
                           - If circuit_params provided: interpreted as I_in(t)
            circuit_params: Optional dict with circuit parameters for current-driven evolution
                           Required fields: Z_T, R_n, J_DP_prime, I_DP
                           If provided, enables Crank-Nicolson vector potential update
            track_occupations: If True, compute and store energy-time representations of gr, gk, and f
                               at each timestep (default: False)
            track_state: If True, the g^R and g^K normalization errors of the newest row (t = -1) at t' = t are stored every
                         track_state_every_n timesteps, while the row still has the full history (default: False)
            track_state_every_n: Stride of the normalization tracking (default 1)

        Returns:
            Dictionary with keys:
            - 'final_state': Final evolved StateObject
            - 'gaps': Array of gap values at each timestep (length num_timesteps)
            - 'currents': Array of current values at each timestep (length num_timesteps)
            - 'vector_potentials': Array of vector potential values at each timestep (length num_timesteps)
              (circuit mode: gaps, currents and vector_potentials are all evaluated at the new time t_(n+1))
            - 'gr_energy_time': (only if track_occupations=True) List of energy-time representations of gr
            - 'gk_energy_time': (only if track_occupations=True) List of energy-time representations of gk
            - 'f_energy_time': (only if track_occupations=True) List of energy-time representations of occupation function
            - 'gr_normalization', 'gk_normalization': (only if track_state=True) (4, n_tracked) complex Pauli components of the g^R and
              g^K normalization errors at (t, t') = (t_n, t_n); 'normalization_indices': (n_tracked,) timestep indices n

        Calls:
            - update_vector_potential(A_external, driving_field, state, circuit_params, ...)
            - _evolve_state_by_one_timestep(state, A_external)
        """
    
        gaps = []
        currents = []
        vector_potentials = []

        # Initialize lists to track energy-time representations if requested
        if track_occupations:
            gr_energy_time_list = []
            gk_energy_time_list = []
            f_energy_time_list = []

        #* normalization of the newest row, evaluated while it still has the full history (the window slides afterwards)
        if track_state:
            gr_normalization_list = []
            gk_normalization_list = []
            normalization_indices = []

        # Initialize A_external as zeros (history window)
        # Size N_t from initial_state's time grid
        N_t = initial_state.gr.data.shape[2]
        if driving_field is None or circuit_params is not None:
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

            # Mode-dependent order of operations
            if circuit_params is None:
                # Mode 1: Simple voltage-driven (original behavior)

                # Evolve state using updated A_external
                gap_new, current_new, vector_potential_new = self._evolve_state_by_one_timestep(state, A_external)

                A_external = self.update_vector_potential(old_vector_potential=A_external, driving_field_current=driving_field[time_index] if driving_field is not None else None, driving_field_next=None, current=None, kernel_prefactor=None, circuit_params=None)

            #? this part remains to be checked!
            else:
                # Mode 2: Current-driven with Crank-Nicolson
                # Evolve state FIRST to get current and kernel
                # Compute kernel prefactor from evolved state

                current_no_gradient = state.get_current_at_time_t(A_external, self.thermal, include_derivative = False)

                driving_field_current = driving_field[time_index] 
                driving_field_next = driving_field[time_index + 1] if time_index < num_timesteps - 1 else driving_field[time_index] 

                #* gap and vector potential are solved self-consistently at the new time; all returned values refer to t_(n+1)
                gap_new, current_new, vector_potential_new = self._evolve_state_by_one_timestep(
                    state, A_external, max_gap_iterations=30, circuit_params=circuit_params, driving_field_current=driving_field_current,
                    driving_field_next=driving_field_next, current_no_gradient=current_no_gradient)

                A_external = np.append(A_external[1:], vector_potential_new)
            
            # Store observables
            gaps += [gap_new]
            currents  += [current_new]
            vector_potentials += [vector_potential_new]

            if track_state and time_index % track_state_every_n == 0:
                _, gr_totals = state.check_gr_normalization(t1_idx=-1)
                _, gk_totals, _ = state.check_keldysh_normalization(-1, self.thermal)
                gr_normalization_list.append(gr_totals[:, -1])
                gk_normalization_list.append(gk_totals[:, -1])
                normalization_indices.append(time_index)

            if track_occupations:
                gr_energy =state.energy_time_representation('gr', eta=self.eta)
                gk_energy = state.energy_time_representation('gk', eta=self.eta)
                f_energy = state.energy_time_representation('f', eta=self.eta)
                gr_energy_time_list.append(gr_energy)
                gk_energy_time_list.append(gk_energy)
                f_energy_time_list.append(f_energy)

        critical_current_factor = circuit_params['J_DP_prime'] if circuit_params is not None else 1.0

        result = { 'final_state': state, 'gaps': np.array(gaps), 'currents': np.array(currents)/critical_current_factor, 'vector_potentials': np.array(vector_potentials)}

        if track_state:
            result['gr_normalization'] = np.array(gr_normalization_list).T
            result['gk_normalization'] = np.array(gk_normalization_list).T
            result['normalization_indices'] = np.array(normalization_indices)

        # Add energy-time data if tracking was enabled
        if track_occupations:
            result['gr_energy_time'] = gr_energy_time_list
            result['gk_energy_time'] = gk_energy_time_list
            result['f_energy_time'] = f_energy_time_list
        return result
