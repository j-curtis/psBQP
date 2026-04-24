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
        print(self.omega_grid)
        # Set eta with warning if too small
        self.eta = system_parameters['eta']
        recommended_eta = 5.0 / self.tmax
        if self.eta < recommended_eta:
            print(f"WARNING: eta = {self.eta:.4f} is smaller than recommended value {recommended_eta:.4f}")
            print(f"         Recommended: eta >= 10/T_max = 10/{self.tmax:.2f}")
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
        return 1.0 / np.log(UsadelKeldyshEvolution.get_bcs_gap_constant() * self.energy_cutoff / self.critical_temperature)

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
        Generate angular frequency (omega) grid from Fourier transform of extended time grid.

        Extends current time grid [-T_max, 0] to [-T_max, +T_max], then computes
        angular frequency grid using FFT convention e^{i omega t}.

        Stores:
            self.omega_grid: Angular frequency array
            self.energy_cutoff: Maximum omega value
        """
        # Extend time grid from [-T_max, 0] to [-T_max, +T_max]
        # Use 2*ntpoints - 1 to avoid duplicating t=0
        n_extended = 2 * self.ntpoints - 1
        extended_time_grid = np.linspace(-self.tmax, self.tmax, n_extended)
        dt_extended = extended_time_grid[1] - extended_time_grid[0]

        # Get frequency bins from FFT
        # np.fft.fftfreq gives frequencies f in cycles per unit time
        # Convention: F(f) = Σ f(t) * e^{-2πi f t}
        freq = np.fft.fftfreq(n_extended, d=dt_extended)

        # Convert to angular frequency: ω = 2π*f
        # This gives us the correct convention e^{i ω t}
        omega = 2 * np.pi * freq

        # Shift to center around 0 (zero frequency in middle)
        self.omega_grid = np.fft.fftshift(omega)

        # Energy cutoff is the maximum absolute omega value (Nyquist frequency)
        self.energy_cutoff = np.max(np.abs(self.omega_grid))


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
        gr_two_time, gk_two_time = equilibrium_solver.fourier_transform_to_two_time(gr_eq, gk_eq)

        # Get BCS coupling constant for StateObject
        bcs_coupling = self._get_BCS_coupling()

        # Create and return StateObject
        initial_state = StateObject(
            gr=gr_two_time,
            gk=gk_two_time,
            bcs_coupling_constant=bcs_coupling,
            grid_params=self.grid_parameters
        )

        return initial_state


    # ========== Thermal Distributions ==========

    def get_thermal_occupation(self, temperature):
        """
        Generate thermal occupation function in two-time representation.

        Uses analytic form: f(τ) = 2πi T / sinh(π τ T)
        Converts from f(τ = t-t') to f(t,t') and stores in self.thermal_dist

        Args:
            temperature: Temperature in energy units

        Returns:
            None (stores result in self.thermal_dist)
        """
        # Create tau grid spanning from -tmax to tmax
        # For times t,t' in [-tmax, 0], tau = t - t' ranges from -tmax to tmax
        n_tau = 2 * self.ntpoints - 1
        tau_grid = np.linspace(-self.tmax, self.tmax, n_tau)
        dtau = tau_grid[1] - tau_grid[0]

        # Compute f(τ) = 2πi T / sinh(π τ T) for all tau values
        # Add small regularization to avoid division by zero at tau=0
        eps = 1e-12
        f_tau = 2.0 * np.pi * 1j * temperature / np.sinh(np.pi * (tau_grid + eps) * temperature) #! This will create numerical problems, should find a way to deal with it.

        # Convert from f(τ) to f(t,t') for times t,t' < 0
        # Create meshgrid of time values (only for negative times)
        t_i, t_j = np.meshgrid(self.time_grid, self.time_grid, indexing='ij')

        # Compute tau = t_i - t_j for all pairs
        tau_matrix = t_i - t_j

        # Find closest tau index for each (t,t') pair
        # tau_grid starts at -tmax, so index 0 corresponds to tau = -tmax
        tau_idx_matrix = np.round((tau_matrix + self.tmax) / dtau).astype(int)

        # Create mask for valid indices
        valid_mask = (tau_idx_matrix >= 0) & (tau_idx_matrix < n_tau)

        # Initialize f(t,t') array
        f_tt = np.zeros((self.ntpoints, self.ntpoints), dtype=complex)

        # Fill f(t,t') using advanced indexing (vectorized)
        f_tt[valid_mask] = f_tau[tau_idx_matrix[valid_mask]]

        # Store in self.thermal_dist
        self.thermal_dist = f_tt

    # ========== Real-Time Evolution ==========

    def _compute_new_gr_column(self, state, external_field=None):
        """
        Evolve retarded Green's function gr by one timestep.

        Computes g^R(t_{time_index}, t_j) for all j < time_index using the
        discretized Usadel equation (without A(t) terms).

        Update equation:
        g^R_{i+1,j} = g^R_{ij} + τ_3 Δt Δ_i g^R_{ij} + (η Δt / 2) τ_3² g^R_{ij}
                      + τ_3 (g^R_{ij} - g^R_{i,j-1}) τ_3
                      + τ_3 Δt g^R_{ij} Δ_j + (η Δt / 2) τ_3 g^R_{ij} τ_3

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

        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=1) +  NambuKeldyshTensor(np.imag(gap_history), pauli_channel=2)
        # Create τ_3 Pauli matrix as NambuKeldyshTensor (identity in time)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        # Extract g^R_{ij} as (2, 2) matrix and wrap as NambuKeldyshTensor
        gr_last_column = state.gr[-2:-1]  # Shape (2, 2, 1, Nt)
        gr_difference = state.gr.gradient(axis=1)

        # Compute update equation terms
        # Term 1: τ_3 Δt Δ_i g^R_{ij}
        term1 = -1j * gap_tensor[-2:-1] * gr_last_column

        # Term 2: (η Δt / 2) τ_3² g^R_{ij} = (η Δt / 2) g^R_{ij}
        # Since τ_3² = I
        term2 = gr_last_column * (self.eta) * 1j

        # Term 3: τ_3 (g^R_{ij} - g^R_{i,j-1}) τ_3
        term3 = (gr_difference[-2:-1] / self.delta_t) * tau3 * 1j

        # Term 4: τ_3 Δt g^R_{ij} Δ_j
        term4 = gr_last_column * gap_tensor * 1j

        # Term 5: (η Δt / 2) τ_3 g^R_{ij} τ_3
        term5 = -gr_last_column * tau3 * (self.eta) * 1j

        # Combine all terms: g^R_{i+1,j} = g^R_{ij} + sum of terms
        gr_new = gr_last_column + (term1 + term2 + term3 + term4 + term5) * self.delta_t

        # Diagonal element: basically stays constant for tau_3 only Hamiltonian
        gr_diagonal_new = tau3 * gr_last_column[:, :, :, -1] * tau3

        return gr_new, gr_diagonal_new

    def _compute_new_gk_column(self, state, external_field=None):
        """
        Evolve Keldysh Green's function gk by one timestep.

        Computes g^K(t_{i+1}, t_j) using the discretized Usadel equation
        with Dynes self-energy (without A(t) terms).

        Update equation (using η instead of η/2):
        g^K_{i+1,j} = g^K_{ij} - i τ_3 Δt [RHS terms including thermal self-energy]

        Args:
            state: StateObject with current gk and gr data
            external_field: Optional external perturbation (not used)

        Returns:
            gk_new: New row of g^K
            gk_diagonal_new: New diagonal element

        Called by:
            - _evolve_state_by_one_timestep()
        """
        # Extract gap history and create gap tensor
        gap_history = state.get_gap_history()
        gap_tensor = (NambuKeldyshTensor(np.real(gap_history), pauli_channel=1) +
                     NambuKeldyshTensor(np.imag(gap_history), pauli_channel=2))

        # Create τ_3 Pauli matrix
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        # Extract last row of g^K
        gk_last_column = state.gk[-2:-1]  # Shape (2, 2, 1, Nt)

        # Compute gradient in t' direction
        gk_difference = state.gk.gradient(axis=1)

        # Compute g^A = -involution(g^R)
        ga = -state.gr.involution()

        gr_last_column = state.gr[-2:-1]
        ga_last_column = ga[-2:-1]  # Shape (2, 2, 1, Nt)

        # Term 1: i Δ_i g^K_{ij}
        term1 = -1j * gap_tensor[-2:-1] * gk_last_column

        # Term 2: -iη τ_3 g^K_{ij}
        term2 = 1j * self.eta * tau3 * gk_last_column

        # Term 3: -i (g^K_{ij} - g^K_{i,j-1})/Δt τ_3
        term3 = 1j * (gk_difference[-2:-1] / self.delta_t) * tau3

        # Term 4: -i g^K_{ij} Δ_j
        term4 = 1j * gk_last_column * gap_tensor

        # Term 5: -iη g^K_{ij} τ_3
        term5 = 1j * self.eta * gk_last_column * tau3

        # Term 6: 2iη Δt Σ_k F_{i,k} τ_3 g^A_{k,j}
        # Convolution: sum over k from 0 to j of F[i,k] * g^A[k,j]
        # F is thermal_dist (Nt x Nt), extract F[i, :] and convolve with g^A[:, j]
        # TODO: Implement proper convolution with thermal distribution
        term6 = NambuKeldyshTensor(np.zeros_like(gk_last_column.data))

        # Term 7: -2iη Δt Σ_k g^R_{i,k} F_{k,j} τ_3
        # Convolution: sum over k from 0 to i of g^R[i,k] * F[k,j]
        # TODO: Implement proper convolution with thermal distribution
        term7 = NambuKeldyshTensor(np.zeros_like(gk_last_column.data))

        # Combine all terms
        gk_new = gk_last_column + (term1 + term2 + term3 + term4 + term5 + term6 + term7) * self.delta_t

        # Diagonal element
        gk_diagonal_new = gk_last_column[:, :, :, -1]
        return gk_new, gk_diagonal_new

    def _evolve_state_by_one_timestep(self, state, time_index, external_field=None):
        """
        Evolve state by one timestep, generating new entries.

        Steps:
        1. Initialize thermal distribution if needed
        2. Call _compute_new_gr_column() to get new gr row and diagonal
        3. Call _compute_new_gk_column() to get new gk row and diagonal
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
            - _compute_new_gr_column(state, external_field)
            - _compute_new_gk_column(state, external_field)
            - state.update_state_object(new_gr_row, new_gr_diag, new_gk_row, new_gk_diag)
        """
        # Initialize thermal distribution if not already done
        if not hasattr(self, 'thermal_dist'):
            self.get_thermal_occupation(self.temperature)

        # Compute new gr and gk rows and diagonals
        new_gr_row, new_gr_diag = self._compute_new_gr_column(state, external_field)
        new_gk_row, new_gk_diag = self._compute_new_gk_column(state, external_field)

        # Update state with new row, column, diagonal
        state.update_state_object(new_gr_row, new_gr_diag, new_gk_row, new_gk_diag)

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

