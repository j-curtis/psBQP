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
from matplotlib import pyplot as plt

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

        # Call the old equilibrium solver
        gr_eq, gap, current = self.usadel_solver._run_temperature_computation(Q=Q, T=temperature, gr0=gr0)
        self.gap_0 = gap
        print('Equilibrium gap is:', gap)
        
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

        For g^R, applies causality constraint: g^R(t,t') = g^R(t,t') * θ(t-t')

        Args:
            gr_eq: Equilibrium retarded Green's function in frequency domain (NambuTensor)
            gk_eq: Optional equilibrium Keldysh Green's function in frequency domain (NambuTensor)

        Returns:
            gr_two_time: Retarded Green's function as NambuKeldyshTensor (2, 2, N_t, N_t)
                        where N_t corresponds to times t < 0, with causality enforced
            gk_two_time: Keldysh Green's function as NambuKeldyshTensor (if gk_eq provided)
        """
        # Transform g^R
        gr_two_time = self._omega_to_two_time(gr_eq, g_type= 'r')

        # Apply causality constraint: g^R(t,t') = 0 for t < t'
        # This is θ(t-t') which zeros the upper triangle
        self._apply_causality(gr_two_time)

        gr_one_time = self.omega_to_one_time(gr_eq, g_type= 'r')
        gk_one_time = self.omega_to_one_time(gk_eq, g_type= 'k')

        if gk_eq is None:
            return gr_two_time

        # Transform g^K
        gk_two_time = self._omega_to_two_time(gk_eq, g_type= 'k')

        return gr_two_time, gk_two_time, gr_one_time, gk_one_time

    def _apply_causality(self, gr_two_time):
        """
        Apply causality constraint to retarded Green's function.

        Enforces g^R(t,t') = g^R(t,t') * θ(t-t') by zeroing upper triangle.
        In matrix form with indices [i,j], this means g[i,j] = 0 for i < j.

        Additionally enforces that τ₁ (X) and τ₂ (Y) Pauli components are zero
        on the diagonal (equal-time) by zeroing g^R[0,1,i,i] and g^R[1,0,i,i].

        Modifies gr_two_time in-place.

        Args:
            gr_two_time: Retarded Green's function (NambuKeldyshTensor, shape (2,2,Nt,Nt))
        """
        # Get time grid size
        Nt = gr_two_time.data.shape[2]

        # Create theta function mask: θ(t-t') = 1 for t >= t', 0 for t < t'
        # In matrix indices: mask[i,j] = 1 for i >= j (lower triangle + diagonal)
        theta_mask = np.tril(np.ones((Nt, Nt)))

        # Apply mask to all Nambu components
        # Broadcast: (2,2,Nt,Nt) * (Nt,Nt) -> (2,2,Nt,Nt)
        gr_two_time.data *= theta_mask[np.newaxis, np.newaxis, :, :]

        # Zero out off-diagonal Nambu elements on the diagonal (equal-time only)
        diagonal_indices = np.arange(Nt)
        gr_two_time.data[1, 1, diagonal_indices, diagonal_indices] = 0.0
        gr_two_time.data[0, 0, diagonal_indices, diagonal_indices] = 0.0

    def omega_to_one_time(self, g_omega, g_type):
        """
        Transform Green's function from frequency to one-time (relative time τ).

        Performs inverse Fourier transform: g(τ) = ∫ dω/(2π) e^{-iωτ} g(ω)

        For g^R, subtracts asymptotic τ₃ component before FFT to regularize the transform
        and avoid sinc oscillations near τ=0.

        Args:
            g_omega: Green's function in frequency domain (NambuTensor, shape (N_omega,))

        Returns:
            g_tau: Green's function in relative time (NambuKeldyshTensor, shape (2, 2, N_tau))
                  where τ ranges from -T_max to +T_max
        """
        # Get omega grid from old usadel solver
        omega_grid = self.usadel_solver.w_arr
        n_omega = len(omega_grid)

        temperature = 0.3 #! This may need to be changed for general T? 
        # Extract Pauli components from NambuTensor using trace
        # Store asymptotic coefficients to add back after FFT
        g_pauli = []
        asymptotic_coeffs = []  # Will store (type, C) for each component

        for pauli_idx in range(4):
            # factor of 2 because of Nambu convention and traces
            pauli_component = np.array(g_omega._trace(pauli_idx))/2

            # Regularization: Subtract asymptotic behavior to avoid sinc oscillations
            # Physical behavior at large |ω|:
            #   - τ₃ component: → C (constant)
            #   - τ₁, τ₂ components: → C/ω (off-diagonal BCS terms decay as Δ/ω)

            if g_type == 'r':
                if pauli_idx == 3:  # tau_3: constant asymptotic
                    C_constant = pauli_component[-1]
                    pauli_component = pauli_component - C_constant
                    asymptotic_coeffs.append(('constant', C_constant, None))
                elif pauli_idx == 2: # tau_1, tau_2: 1/ω asymptotic #! has to be generalized properly
                    # Estimate coefficients: g(ω) ≈ C/ω + C'/ω² at large ω
                    # Use least-squares fit over tail region
                    tail_start = int(0.8 * n_omega)  # Last 20% of points
                    omega_tail = omega_grid[tail_start:]
                    g_tail = pauli_component[tail_start:]

                    # Fit ω²·g(ω) = C·ω + C' (linear regression)
                    # This gives both C/ω and C'/ω² terms
                    y = omega_tail**2 * g_tail
                    X = np.column_stack([omega_tail, np.ones_like(omega_tail)])

                    # Least squares: [C, C'] = (X^T X)^{-1} X^T y
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    C_decay = coeffs[0]
                    C_prime = coeffs[1]

                    # Choose regularization scale ω₀
                    # Use a characteristic energy (e.g., twice the gap or broadening)
                    omega_10percent = np.max(omega_grid) * 1e-2
                    omega_0 = np.abs(omega_10percent) / 2.0  # Regularization scale
                    C_decay = -1j*self.gap_0
                    C_prime = -1j*self.gap_0 * (-1j * self.system_parameters['eta'])
                    # Subtract regularization: L(ω) = C·ω/(ω² + ω₀²) + C'/(ω² + ω₀²)
                    # First term → C/ω, second term → C'/ω² at large ω
                    regularization = (C_decay * omega_grid / (omega_grid**2 + omega_0**2) +
                                    C_prime / (omega_grid**2 + omega_0**2))
                    pauli_component = pauli_component - regularization
                    asymptotic_coeffs.append(('lorentzian_extended', C_decay, C_prime, omega_0))
                else:  # pauli_idx 0 or 1: no regularization
                    asymptotic_coeffs.append(None)
            elif g_type == 'k':  # Keldysh Green's function
                if pauli_idx == 3:  # tau_3: tanh(ω/2T) asymptotic for g^K
                    # At equilibrium, g^K_3(ω) = 2·tanh(ω/2T) at equilibrium
                    # Subtract this before FFT for better convergence
                    C_tanh = 2.0
                    tanh_omega = np.tanh(omega_grid / (2.0 * temperature))
                    pauli_component = pauli_component - C_tanh * tanh_omega

                    asymptotic_coeffs.append(('tanh', C_tanh, temperature))
                else:  # pauli_idx 0, 1, 2: no regularization for these
                    asymptotic_coeffs.append(None)
            else:  # Unknown g_type
                asymptotic_coeffs.append(None)

            g_pauli.append(pauli_component)

        # Compute d_omega for normalization
        d_omega = omega_grid[1] - omega_grid[0]

        # Inverse Fourier transform for each Pauli component
        # Convention: g(τ) = ∫ dω/(2π) e^{-iωτ} g(ω)
        g_tau_pauli = []

        for pauli_idx, pauli_component in enumerate(g_pauli):
            # Undo the fftshift to prepare for fft
            g_omega_unshifted = np.fft.ifftshift(pauli_component)

            # Physics: g(τ) = ∫ dω/(2π) e^{-iωτ} g(ω)
            # Discretized: g(τ_k) ≈ Σ_n g(ω_n) e^{-iω_n τ_k} * Δω/(2π)
            # numpy.fft.fft gives: Σ_n g(ω_n) e^{-2πi n k/N} (no 1/N factor)
            g_tau_raw = np.fft.fft(g_omega_unshifted)

            # Apply normalization from the integral measure: Δω/(2π)
            # Note: fft already gives the full sum, so just multiply by Δω/(2π)
            g_tau = g_tau_raw * d_omega / (2.0 * np.pi)

            # Shift to get correct tau ordering
            g_tau_shifted = np.fft.fftshift(g_tau)

            # Add back the asymptotic contribution in time domain
            # Reconstruct tau grid for the FFT output
            tau_grid_fft = np.linspace(-np.pi/d_omega, np.pi/d_omega, n_omega)

            # Only add back for indices that had regularization (skip None entries)
            if asymptotic_coeffs[pauli_idx] is not None:
                asym_type = asymptotic_coeffs[pauli_idx][0]
                C = asymptotic_coeffs[pauli_idx][1]

                if asym_type == 'lorentzian':  # Lorentzian: C·ω/(ω² + ω₀²)
                    omega_0 = asymptotic_coeffs[pauli_idx][2]
                    # FT[C·ω/(ω² + ω₀²)] = -(iC/2)·sign(τ)·exp(-ω₀|τ|)  [CORRECTED SIGN]
                    asymptotic_contribution = -(1j * C / 2.0) * np.sign(tau_grid_fft + 1e-10) * np.exp(-omega_0 * np.abs(tau_grid_fft))
                    g_tau_shifted = g_tau_shifted + asymptotic_contribution

                elif asym_type == 'lorentzian_extended':  # Extended: C·ω/(ω²+ω₀²) + C'/(ω²+ω₀²)
                    C_prime = asymptotic_coeffs[pauli_idx][2]
                    omega_0 = asymptotic_coeffs[pauli_idx][3]
                    # FT[C·ω/(ω²+ω₀²)] = -(iC/2)·sign(τ)·exp(-ω₀|τ|)  [CORRECTED SIGN]
                    # FT[C'/(ω²+ω₀²)] = (C'/2ω₀)·exp(-ω₀|τ|)
                    asymptotic_contribution = (-(1j * C / 2.0) * np.sign(tau_grid_fft + 1e-10) +
                                            (C_prime / (2.0 * omega_0))) * \
                                            np.exp(-omega_0 * np.abs(tau_grid_fft))
                    g_tau_shifted = g_tau_shifted + asymptotic_contribution

                elif asym_type == 'tanh':  # tanh(ω/2T) asymptotic for Keldysh tau_3
                    # FT[tanh(ω/2T)] = -iT / sinh(πτT)
                    # This decays exponentially as |τ| → ∞, unlike sign(ω)
                    T = asymptotic_coeffs[pauli_idx][2]  # Temperature

                    # Create mask to avoid division by zero at τ=0
                    mask = (np.abs(tau_grid_fft) > 1e-10)

                    # Initialize with zeros
                    asymptotic_contribution = np.zeros_like(tau_grid_fft, dtype=complex)

                    # Fill in -iT / sinh(πτT) for τ ≠ 0
                    asymptotic_contribution[mask] = (-1j * C * T /
                                                    np.sinh(np.pi * tau_grid_fft[mask] * T))

                    # At τ=0, use L'Hôpital's limit: lim_{τ→0} -iT/sinh(πτT) = -i/π
                    tau_zero_mask = (np.abs(tau_grid_fft) <= 1e-10)
                    asymptotic_contribution[tau_zero_mask] = -1j * C / np.pi

                    g_tau_shifted = g_tau_shifted + asymptotic_contribution

            # For constant asymptotic (tau_3 in retarded case), we don't add anything back
            # The constant in frequency → delta function at τ=0, which we ignore

            g_tau_pauli.append(g_tau_shifted)

        # Convert from Pauli components to NambuKeldyshTensor
        g_one_time = None
        for pauli_idx in range(4):
            g_component = NambuKeldyshTensor(g_tau_pauli[pauli_idx], pauli_channel=pauli_idx)
            if g_one_time is None:
                g_one_time = g_component
            else:
                g_one_time = g_one_time + g_component

        # Apply causality constraint only for retarded Green's function
        # θ(τ) enforces g^R(τ) = 0 for τ < 0
        # g^K should NOT have causality constraint to preserve symmetry
        if g_type == 'r':
            d_omega = omega_grid[1] - omega_grid[0]
            tau_grid_fft = np.linspace(-np.pi/d_omega, np.pi/d_omega, n_omega)
            theta_mask = (tau_grid_fft >= 0).astype(float)
            g_one_time.data *= theta_mask[np.newaxis, np.newaxis, :]

        return g_one_time

    def _omega_to_two_time(self, g_omega, g_type):
        """
        Transform single Green's function from frequency to two-time.

        Steps:
        1. Call omega_to_one_time() to get g(τ) with all regularizations
        2. Reshape from g(τ) to g(t,t') where g(t,t') = g(t-t')
        3. Return only the part where t < 0 and t' < 0

        Args:
            g_omega: Green's function in frequency domain (NambuTensor, shape (N_omega,))

        Returns:
            g_two_time: Green's function in two-time representation (NambuKeldyshTensor, shape (2, 2, N_t, N_t))
                       where N_t corresponds to times t < 0
        """
        # Get time grid parameters from grid_parameters
        if 'time_sampling' in self.grid_parameters:
            ntpoints = self.grid_parameters['time_sampling']
        elif 'n_tpoints' in self.grid_parameters:
            ntpoints = self.grid_parameters['n_tpoints']
        else:
            raise ValueError("grid_parameters must contain 'time_sampling' or 'n_tpoints'")

        if 'time_duration' in self.grid_parameters:
            tmax = self.grid_parameters['time_duration']
        elif 't_max' in self.grid_parameters:
            tmax = self.grid_parameters['t_max']
        else:
            raise ValueError("grid_parameters must contain 'time_duration' or 't_max'")

        # Compute time step (from UsadelKeldyshEvolution convention)
        dt = tmax / (ntpoints - 1)

        # Extend time grid from [-T_max, 0] to [-T_max, +T_max]
        n_extended = 2 * ntpoints - 1
        time_grid_full = np.linspace(-tmax, tmax, n_extended)

        # Get omega grid from old usadel solver
        omega_grid = self.usadel_solver.w_arr
        n_omega = len(omega_grid)

        # Call omega_to_one_time to get g(τ) with all regularizations
        g_one_time = self.omega_to_one_time(g_omega, g_type = g_type)

        # Extract Pauli components from the NambuKeldyshTensor
        g_tau_pauli = []
        for pauli_idx in range(4):
            # Extract component using trace and normalization
            g_tau_pauli.append(np.array(g_one_time.trace(pauli_idx))/2)

        # Build g(t,t') on the full extended time grid, then truncate to t < 0, t' < 0
        # Create meshgrid of time values for all (t_i, t_j) pairs on FULL grid
        t_i, t_j = np.meshgrid(time_grid_full, time_grid_full, indexing='ij')

        # Compute tau = t_i - t_j for all pairs at once
        tau_matrix = t_i - t_j

        # Compute tau indices for all pairs
        # tau = 0 should be at the center of g_tau (index n_omega // 2)

        #* New code
        # Compute actual tau spacing from FFT grid
        d_omega = omega_grid[1] - omega_grid[0]
        tau_grid_actual = np.linspace(-np.pi/d_omega, np.pi/d_omega, n_omega)
        dtau_fft = tau_grid_actual[1] - tau_grid_actual[0]


        # Compute tau indices by finding nearest neighbor in tau_grid_actual
        # Use searchsorted to find insertion points, then check which neighbor is closer
        tau_flat = tau_matrix.flatten()
        tau_indices_right = np.searchsorted(tau_grid_actual, tau_flat)

        # Clip to valid range
        tau_indices_right = np.clip(tau_indices_right, 0, n_omega - 1)
        tau_indices_left = np.clip(tau_indices_right - 1, 0, n_omega - 1)

        # Find which neighbor is closer
        dist_left = np.abs(tau_flat - tau_grid_actual[tau_indices_left])
        dist_right = np.abs(tau_flat - tau_grid_actual[tau_indices_right])

        # Choose the closer index
        tau_indices = np.where(dist_left < dist_right, tau_indices_left, tau_indices_right)
        tau_idx_matrix = tau_indices.reshape(tau_matrix.shape)


        d_omega = omega_grid[1] - omega_grid[0]
        dtau_fft = 2*np.pi / (d_omega * n_omega)
        dt_evolution = tmax / (ntpoints - 1)


        # Create mask for valid indices (g_tau has n_omega points from FFT)
        valid_mask = (tau_idx_matrix >= 0) & (tau_idx_matrix < n_omega)

        g_two_time_pauli = []
        for g_tau in g_tau_pauli:
            # Initialize g(t,t') array on FULL extended grid
            g_tt_full = np.zeros((n_extended, n_extended), dtype=complex)

            # Fill g(t,t') using advanced indexing (vectorized)
            g_tt_full[valid_mask] = g_tau[tau_idx_matrix[valid_mask]]

            # Extract only the part where t < 0 and t' < 0
            # The full grid is [-T_max, T_max], we want only [-T_max, 0)
            g_tt_negative = g_tt_full[:ntpoints, :ntpoints]

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


