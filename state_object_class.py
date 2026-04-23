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

    def __init__(self, gr=None, gk=None, grid_params=None):
        """
        Initialize state object with Green's functions and grid parameters.

        Args:
            gr: Retarded Green's function g^R (NambuKeldyshTensor)
            gk: Keldysh Green's function g^K (NambuKeldyshTensor)
            grid_params: Dictionary with grid parameters
        """
        self.gr = gr
        self.gk = gk

        # Extract grid parameters
        if grid_params is not None:
            self.cutoff = grid_params['cutoff']
            self.T_max = grid_params.get('time_duration', grid_params.get('T_max'))

            # Compute dt from time grid
            if 'dt' in grid_params:
                self.dt = grid_params['dt']
            elif 'time_sampling' in grid_params and 'time_duration' in grid_params:
                self.dt = grid_params['time_duration'] / (grid_params['time_sampling'] - 1)
            else:
                self.dt = None
        else:
            self.cutoff = None
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

    def get_gap(self):
        """Extract superconducting gap from Green's functions."""
        pass

    def get_current(self, Q=None):
        """Compute total current."""
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
        gap = self.get_gap()
        current = self.get_current()

        return f"Current gap is {gap}\nCurrent current is {current}"

    # ========== Cleanup ==========

    def __del__(self):
        """Clean up resources."""
        pass
