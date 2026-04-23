"""
Usadel-Keldysh evolution class for real-time dynamics of superconducting systems.
Handles time evolution of retarded Green's function g^R and distribution function f.
"""

import numpy as np
from nambu_keldysh_class import NambuKeldyshTensor
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

        # Generate time grid
        self._generate_time_grid()
        
        # Generate omega grid from extended time domain
        self._generate_omega_grid()



        # Set eta with warning if too small
        self.eta = system_parameters['eta']
        recommended_eta = 10.0 / self.tmax
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
            self.N_t = self.grid_parameters['time_sampling']
        elif 'N_t' in self.grid_parameters:
            self.N_t = self.grid_parameters['N_t']
        else:
            raise ValueError("grid_parameters must contain 'time_sampling' or 'N_t'")

        if 'time_duration' in self.grid_parameters:
            self.t_max = self.grid_parameters['time_duration']
        elif 't_max' in self.grid_parameters:
            self.t_max = self.grid_parameters['t_max']
        else:
            raise ValueError("grid_parameters must contain 'time_duration' or 't_max'")

        # Compute time step
        self.delta_t = self.t_max / (self.N_t - 1)

        # Generate time grid
        self.time_grid = np.linspace(-self.t_max, 0, self.N_t)


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

    def _generate_initial_state(self):
        """
        Generate initial state from equilibrium and truncate for t,t' < 0.

        Steps:
        1. Call equilibrium_solver.compute_self_cons_equilibrium() to get equilibrium gr, gk
        2. Truncate to only t,t' < 0 time indices
        3. Construct StateObject with initial time slices

        Returns:
            StateObject with equilibrium data for t < 0

        Calls:
            - Equilibrium_class.compute_self_cons_equilibrium()
        """
        pass

    # ========== Hamiltonian Construction ==========

    def _construct_hr(self, Q: float, delta, sigma_r=None):
        """
        Construct retarded Hamiltonian h^R.

        Args:
            Q: Supercurrent phase gradient
            delta: Superconducting gap (complex scalar)
            sigma_r: Retarded self-energy (NambuKeldyshTensor or None)

        Returns:
            h^R = (omega - Q)*tau_3 - Delta*tau_y - Sigma^R + i*eta
        """
        pass

    # ========== Thermal Distributions ==========

    def _get_thermal_occupation(self, temperature):
        """
        Generate equilibrium Fermi-Dirac distribution function.

        Args:
            temperature: Temperature in energy units

        Returns:
            f(omega) = tanh(omega / 2T) as NambuKeldyshTensor
        """
        pass

    # ========== Real-Time Evolution ==========

    def _evolve_gr_by_one_timestep(self, state, time_index, external_field=None):
        """
        Evolve retarded Green's function gr by one timestep.

        Computes gr at new timestep from current state data.

        Args:
            state: StateObject with current gr data
            time_index: Current time index
            external_field: Optional external perturbation

        Returns:
            new_gr: Updated retarded Green's function at next timestep

        Called by:
            - _evolve_state_by_one_timestep()
        """
        pass

    def _evolve_gk_by_one_timestep(self, state, time_index, external_field=None):
        """
        Evolve Keldysh Green's function gk by one timestep.

        Computes gk at new timestep from current state data.

        Args:
            state: StateObject with current gk data
            time_index: Current time index
            external_field: Optional external perturbation

        Returns:
            new_gk: Updated Keldysh Green's function at next timestep

        Called by:
            - _evolve_state_by_one_timestep()
        """
        pass

    def _evolve_state_by_one_timestep(self, state, time_index, external_field=None):
        """
        Evolve state by one timestep, generating new entries.

        Steps:
        1. Extract current time slice from state
        2. Call _evolve_gr_by_one_timestep() to get new gr
        3. Call _evolve_gk_by_one_timestep() to get new gk
        4. Update state using state._update_state_object()

        Args:
            state: StateObject with current data
            time_index: Current time index
            external_field: Optional external perturbation

        Returns:
            Updated StateObject

        Calls:
            - _evolve_gr_by_one_timestep(state, time_index, external_field)
            - _evolve_gk_by_one_timestep(state, time_index, external_field)
            - state._update_state_object(new_gr, new_gk, time_index)
        """
        pass

    def real_time_evolution(self, initial_state, num_timesteps, external_field=None):
        """
        Main real-time evolution loop.

        Evolves state forward in time, extracting observables at each step.

        Steps:
        1. Initialize state from initial_state
        2. For each timestep:
            a. Call _evolve_state_by_one_timestep()
            b. Extract gap using state.get_gap()
            c. Extract current using state.get_current()
            d. Store gap and current in arrays
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
            - state.get_gap()
            - state.get_current()
        """
        pass

