"""
State object class for Keldysh formalism.
Stores retarded Green's function, Keldysh Green's function, and all system properties.
"""

import numpy as np
from nambu_keldysh_class import NambuKeldyshTensor


class StateObject:
    """
    Container for complete system state in Keldysh formalism.
    Stores Green's functions and derived quantities.
    """

    def __init__(self, gr, gk, bcs_coupling_constant, grid_params=None):
        """
        Initialize state object with Green's functions and grid parameters.

        Args:
            gr: Retarded Green's function g^R (NambuKeldyshTensor)
            gk: Keldysh Green's function g^K (NambuKeldyshTensor)
            grid_params: Dictionary with grid parameters
            bcs_coupling_constant: BCS coupling constant λ for gap equation
        """
        self.gr = gr
        self.gk = gk
        self.bcs_coupling_constant = bcs_coupling_constant

        # Extract grid parameters
        if grid_params is not None:
            self.T_max = grid_params.get('time_duration', grid_params.get('T_max'))

            # Compute dt from time grid
            if 'dt' in grid_params:
                self.dt = grid_params['dt']
            elif 'time_sampling' in grid_params and 'time_duration' in grid_params:
                self.dt = grid_params['time_duration'] / (grid_params['time_sampling'] - 1)
            else:
                self.dt = None
        else:
            self.T_max = None
            self.dt = None

    # ========== Green's Function Relations ==========

    def _r2a(self):
        """
        Compute advanced Green's function from retarded.

        Uses involution: g^A = -(g^R)^†

        Returns:
            NambuKeldyshTensor: Advanced Green's function g^A
        """
        return -self.gr.involution()

    # ========== State Properties ==========

    def get_gap_history(self):
        """
        Extract superconducting gap from Green's functions.

        Uses the gap equation: Δ(t) = λ Tr[τ₋ g^K(t,t)]
        where τ₋ = (τ₁ - iτ₂)/2 is the lowering operator.

        Returns:
            np.ndarray: Gap values Δ(t) at each time point
        """
        # Trace g^K over Nambu indices with lowering operator τ₋
        # This reduces (2, 2, N_t, N_t) -> (N_t, N_t)
        gk_traced = self.gk.trace(pauli_index='-')

        # Extract equal-time values g^K(t,t) using diagonal
        gk_diag = np.diagonal(gk_traced)

        # Gap equation: Δ = -λ/4 * Tr[τ₋ g^K(t,t)]
        gap_history = -0.25 * self.bcs_coupling_constant * gk_diag
        #TODO: check factor of 4!

        return gap_history

    def get_current_history(self, Q=None):
        """Compute total current. -- this is stage 2 of the project"""
        pass

    # ========== Utilities ==========

    def _update_state_object(self, new_gr, new_gk, time_index):
        """
        Update state with newly computed timestep.

        Inserts new_gr and new_gk at specified time_index in the stored data.

        Args:
            new_gr: New retarded Green's function at timestep (NambuKeldyshTensor)
            new_gk: New Keldysh Green's function at timestep (NambuKeldyshTensor)
            time_index: Time index to update

        Called by:
            - UsadelKeldyshEvolution._evolve_state_by_one_timestep()
        """
        pass

    # ========== Consistency Checks ==========

    def check_normalization(self):
        """Verify normalization: g^R @ g^R = -1."""
        pass

    def check_keldysh_relation(self):
        """Verify g^K = g^R @ f - f @ g^A."""
        pass

    # ========== String Representation ==========

    def __str__(self):
        """String representation showing state properties."""
        try:
            gap_history = self.get_gap_history()
            gap_str = f"Gap(t_final) = {gap_history[-1]:.6f}"
        except (ValueError, AttributeError, IndexError):
            gap_str = "Gap: Not computed"

        try:
            current_history = self.get_current_history()
            current_str = f"Current(t_final) = {current_history[-1]:.6f}"
        except (ValueError, AttributeError, IndexError, TypeError):
            current_str = "Current: Not implemented (Stage 2)"

        return f"StateObject:\n  {gap_str}\n  {current_str}\n  Shape: {self.gr.data.shape if self.gr is not None else 'N/A'}"

    # ========== Cleanup ==========

    def __del__(self):
        """Clean up resources."""
        pass
