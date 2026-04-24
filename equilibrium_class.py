"""
Equilibrium solver class for computing self-consistent equilibrium states.
Handles all equilibrium calculations including gap equations and self-consistency.
Wrapper around the old equilibrium code.
"""

import numpy as np
import jax
import jax.numpy as jnp

from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject

# Import old equilibrium solver classes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'equilibrium_python'))

from Usadel_methods import UsadelEvolution
from nambu_class import NambuTensor
from system_state import SupercondctingState


class EquilibriumSolver:
    """
    Handles equilibrium self-consistency calculations.
    Wrapper around the old UsadelEvolution class to compute equilibrium gr, gk.
    """

    def __init__(self, grid_parameters, system_parameters, optimization_parameters=None, sigma_scatterings=None):
        """
        Initialize equilibrium solver.

        Args:
            grid_parameters: dict with omega grid, cutoff, eta
            system_parameters: dict with critical_temperature, temperature
            optimization_parameters: dict with solver settings
            sigma_scatterings: dict with scattering mechanisms and rates
        """
        # Store parameters
        self.grid_parameters = grid_parameters
        self.system_parameters = system_parameters
        self.optimization_parameters = optimization_parameters
        self.sigma_scatterings = sigma_scatterings

        # Create the old Usadel solver object
        self.usadel_solver = UsadelEvolution(
            grid_parameters,
            system_parameters,
            optimization_parameters,
            sigma_scatterings
        )


    def compute_equilibrium_gr(self, temperature, Q=0.0, gr0=None, compute_gk=False):
        """
        Compute equilibrium retarded Green's function.

        Uses the old Usadel solver to compute self-consistent equilibrium state.
        Optionally computes Keldysh Green's function from thermal distribution.

        Args:
            temperature: Temperature in energy units
            Q: Phase gradient (default 0)
            gr0: Initial guess for gr (NambuTensor from old code)
            compute_gk: If True, also compute and return g^K (default False)

        Returns:
            If compute_gk is False:
                gr_eq: Equilibrium retarded Green's function (old NambuTensor)
            If compute_gk is True:
                gr_eq: Equilibrium retarded Green's function (old NambuTensor)
                gk_eq: Equilibrium Keldysh Green's function (old NambuTensor)
        """
        #TODO: later we can load the solution, no need to find it self-consistently
        # Call the old equilibrium solver
        gr_eq, gap, current = self.usadel_solver._run_temperature_computation(Q=Q, T=temperature, gr0=gr0)

        if not compute_gk:
            return gr_eq

        # Compute equilibrium Keldysh Green's function
        # g^K = g^R @ f - f @ g^A
        # where f(ω) = tanh(ω/2T) is the thermal distribution

        # Get thermal occupation function
        f_eq = self._get_thermal_occupation(temperature)

        # Compute g^A from g^R using involution
        ga_eq = self._compute_advanced(gr_eq)

        # Compute g^K using Keldysh relation
        gk_eq = gr_eq @ f_eq - f_eq @ ga_eq

        return gr_eq, gk_eq

    def _get_thermal_occupation(self, temperature):
        """
        Generate equilibrium thermal occupation function.

        Args:
            temperature: Temperature in energy units

        Returns:
            f_eq: Thermal distribution f(ω) = tanh(ω/2T) as NambuTensor
        """

        # Compute thermal distribution
        f_array = jnp.tanh(0.5 * self.usadel_solver.w_arr / temperature)

        # Create NambuTensor (identity in Nambu space)
        f_eq = NambuTensor(f_array, pauli_channel=0)

        return f_eq

    def _compute_advanced(self, gr):
        """
        Compute advanced Green's function from retarded.

        Uses involution: g^A = -τ₃ (g^R)^† τ₃

        Args:
            gr: Retarded Green's function (NambuTensor)

        Returns:
            ga: Advanced Green's function (NambuTensor)
        """
        # Use the involution method from NambuTensor
        ga = -gr._involution()

        return ga

    def fourier_transform_to_two_time(self, gr_eq, gk_eq=None):
        """
        Transform equilibrium Green's functions from frequency to two-time representation.

        Performs inverse Fourier transform: g(τ) = ∫ dω/(2π) e^{-iωτ} g(ω)
        Then reshapes to g(t,t') using equilibrium property: g(t,t') = g(t-t') = g(τ)

        FT is done on the full time grid [-T_max, T_max] inferred from omega grid.
        Returns only the part where t < 0 and t' < 0.

        Args:
            gr_eq: Equilibrium retarded Green's function in frequency domain (NambuTensor)
            gk_eq: Optional equilibrium Keldysh Green's function in frequency domain (NambuTensor)

        Returns:
            gr_two_time: Retarded Green's function as NambuKeldyshTensor (2, 2, N_t, N_t)
                        where N_t corresponds to times t < 0
            gk_two_time: Keldysh Green's function as NambuKeldyshTensor (if gk_eq provided)
        """
        # Transform g^R
        gr_two_time = self._omega_to_two_time(gr_eq)

        if gk_eq is None:
            return gr_two_time

        # Transform g^K
        gk_two_time = self._omega_to_two_time(gk_eq)

        return gr_two_time, gk_two_time

    def _omega_to_two_time(self, g_omega):
        """
        Transform single Green's function from frequency to two-time.

        Steps:
        1. Infer full time grid [-T_max, T_max] from omega grid
        2. Extract Pauli components from NambuTensor
        3. Inverse FFT from ω to τ with proper normalization on full grid
        4. Reshape from g(τ) to g(t,t') where g(t,t') = g(t-t')
        5. Return only the part where t < 0 and t' < 0
        6. Convert to NambuKeldyshTensor

        Args:
            g_omega: Green's function in frequency domain (NambuTensor, shape (N_omega,))

        Returns:
            g_two_time: Green's function in two-time representation (NambuKeldyshTensor, shape (2, 2, N_t, N_t))
                       where N_t corresponds to times t < 0
        """
        # Get omega grid from usadel solver
        omega_grid = self.usadel_solver.w_arr
        n_omega = len(omega_grid)

        # Infer full time grid from omega grid
        # The omega grid corresponds to FFT of time grid from -T_max to T_max
        d_omega = omega_grid[1] - omega_grid[0]

        # From FFT relation: d_omega = 2π / (n_omega * dt)
        dt = 2.0 * np.pi / (n_omega * d_omega)

        # Full time grid spans from -T_max to T_max with n_omega points
        tmax = dt * (n_omega - 1) / 2.0
        time_grid_full = np.linspace(-tmax, tmax, n_omega)

        # Extract Pauli components from NambuTensor using trace
        g_pauli = []
        for pauli_idx in range(4):
            g_pauli.append(np.array(g_omega._trace(pauli_idx)))

        # Inverse Fourier transform for each Pauli component
        # Convention: g(τ) = ∫ dω/(2π) e^{-iωτ} g(ω)
        g_tau_pauli = []

        for pauli_component in g_pauli:
            # Undo the fftshift to prepare for ifft
            g_omega_unshifted = np.fft.ifftshift(pauli_component)

            # Perform inverse FFT
            g_tau_raw = np.fft.ifft(g_omega_unshifted)

            # Apply normalization: multiply by N * d_omega / (2π)
            g_tau = g_tau_raw * n_omega * d_omega / (2.0 * np.pi)

            # Shift to get correct tau ordering
            g_tau_shifted = np.fft.fftshift(g_tau)

            g_tau_pauli.append(g_tau_shifted)

        # Build g(t,t') on the full time grid, then truncate to t < 0, t' < 0
        # Create meshgrid of time values for all (t_i, t_j) pairs on FULL grid
        t_i, t_j = np.meshgrid(time_grid_full, time_grid_full, indexing='ij')

        # Compute tau = t_i - t_j for all pairs at once
        tau_matrix = t_i - t_j

        # Compute tau indices for all pairs
        # tau = 0 should be at the center of g_tau (index n_omega // 2)
        tau_idx_matrix = np.round(tau_matrix / dt).astype(int) + n_omega // 2

        # Create mask for valid indices
        valid_mask = (tau_idx_matrix >= 0) & (tau_idx_matrix < n_omega)

        g_two_time_pauli = []
        for g_tau in g_tau_pauli:
            # Initialize g(t,t') array on FULL grid
            g_tt_full = np.zeros((n_omega, n_omega), dtype=complex)

            # Fill g(t,t') using advanced indexing (vectorized)
            g_tt_full[valid_mask] = g_tau[tau_idx_matrix[valid_mask]]

            # Extract only the part where t < 0 and t' < 0
            # The full grid is [-T_max, T_max], we want only [-T_max, 0)
            # This corresponds to indices [0, n_omega//2] (excluding the midpoint which is t=0)
            # For n_omega points spanning [-T_max, T_max], the first n_omega//2 points are t < 0
            n_negative = n_omega // 2  # Number of points with t < 0
            g_tt_negative = g_tt_full[:n_negative, :n_negative]

            g_two_time_pauli.append(g_tt_negative)

        # Convert from Pauli components to NambuKeldyshTensor
        g_two_time = None
        for pauli_idx in range(4):
            g_component = NambuKeldyshTensor(g_two_time_pauli[pauli_idx], pauli_channel=pauli_idx)
            if g_two_time is None:
                g_two_time = g_component
            else:
                g_two_time = g_two_time + g_component

        return g_two_time


