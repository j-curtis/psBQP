"""
Self-energy classes for the time-domain Keldysh/Usadel evolution.

A self-energy is described twice:
  - get_sigma_*          : the self-energy itself, a two-time NambuKeldyshTensor (2,2,Nt,Nt)
  - r_/k_equation_coefficients : the coefficients used in the g^R / g^K equations themselves, a terms_dict in the format of
                           UsadelKeldyshEvolution.construct_discrete_operators, i.e. {'typeN_<name>': {'L':..,'M':..,'R':..,'S':..}},
                           ready to be merged into evolution_terms.
Both sets take the same input: (state_object, params).
"""

from abc import ABC, abstractmethod
import numpy as np
from nambu_keldysh_class import NambuKeldyshTensor, get_expansion_tensor


class SelfEnergy(ABC):
    """
    Abstract base class of a self-energy in the time-domain Keldysh formalism.

    Attributes:
        order: order of the self-energy in g (0: independent of g, 1: linear in g, 2: quadratic in g).
               Not to be confused with the 'typeN' labels (0-8) of the discrete operators.
        scattering_rate: scattering rate of the mechanism (1.0 for the EM self-energy).
        name: label used to build term keys, e.g. 'type1_<name>'.

    Interface (state_object: StateObject, params: dict of extra inputs, e.g. thermal object, gap, vector potential):
        get_sigma_r / get_sigma_a / get_sigma_k          -> NambuKeldyshTensor, shape (2,2,Nt,Nt)
        r_equation_coefficients / k_equation_coefficients -> dict of terms (construct_discrete_operators format)

    Coefficient convention: terms sit on the same side of the equation as the derivative terms (operator on the left,
    old-time part enters rhs_vector with a minus sign), so a g-independent source S enters as rhs -= CN(S).
    """

    def __init__(self, order, scattering_rate=1.0, name=None):
        if order not in (0, 1, 2):
            raise ValueError(f"Self-energy order in g must be 0, 1 or 2, got {order}")

        if scattering_rate <= 0:
            raise ValueError(f"Scattering rate must be positive, got {scattering_rate}")
        self.order = order
        self.scattering_rate = scattering_rate
        self.name = name if name is not None else type(self).__name__

    # ========== Self-energy as two-time objects ==========

    @abstractmethod
    def get_sigma_r(self, state_object, params=None) -> NambuKeldyshTensor:
        """Retarded self-energy Sigma^R(t,t') as a (2,2,Nt,Nt) NambuKeldyshTensor."""

    @abstractmethod
    def get_sigma_a(self, state_object, params=None) -> NambuKeldyshTensor:
        """Advanced self-energy Sigma^A(t,t') as a (2,2,Nt,Nt) NambuKeldyshTensor."""

    @abstractmethod
    def get_sigma_k(self, state_object, params=None) -> NambuKeldyshTensor:
        """Keldysh self-energy Sigma^K(t,t') as a (2,2,Nt,Nt) NambuKeldyshTensor."""

    # ========== Self-energy as terms of the discrete evolution equation ==========

    @abstractmethod
    def r_equation_coefficients(self, state_object, params=None) -> dict:
        """Coefficients used in the g^R equation itself, as {'typeN_<name>': {operator key: tensor}}."""

    @abstractmethod
    def k_equation_coefficients(self, state_object, params=None) -> dict:
        """Coefficients used in the g^K equation itself, as {'typeN_<name>': {operator key: tensor}}."""


def _two_row_source(row_function):
    """Stack row_function(dt_shift) for dt_shift = 1 (row t-dt) and 0 (row t) into a (2,2,2,Nt) tensor (type0 form)."""
    rows = [row_function(dt_shift) for dt_shift in (1, 0)]
    return NambuKeldyshTensor(np.concatenate([row.data for row in rows], axis=2))


#! the expressions here need to be checked using analytical expressions!
class EM_selfenergy(SelfEnergy):
    """
    Electromagnetic (vector potential) self-energy, Sigma^X(t,t') = -i A(t) tau3 g^X(t,t') A(t') tau3 (e^2 D = 1), order 1 in g.

    params: {'A_history': array (Nt,) with the current time last, 'thermal': ThermalConvolutions, 'gap_tensor': gap (g^K equation only)}.
    The equation coefficients are the electromagnetic terms of _compute_new_gr_row / _compute_new_gk_complete; the thermal source
    uses the full (f and F) convolutions, so params also needs 'gap_tensor'. The scattering rate is stored only (e^2 D = 1).
    """

    def __init__(self, scattering_rate=1.0):
        super().__init__(order=1, scattering_rate=scattering_rate, name='em')

    # ========== helpers ==========

    @staticmethod
    def _vector_potential(state_object, params):
        A_history = np.asarray(params['A_history'])
        return A_history, NambuKeldyshTensor(A_history, pauli_channel=0)

    def _sigma(self, g, A_tensor):
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        return -1j * (A_tensor * tau3 * g * A_tensor * tau3)

    # ========== Self-energy (regular part; the local -i A(t)^2 tau3 delta(t-t') piece is not included) ==========

    def get_sigma_r(self, state_object, params=None):
        _, A_tensor = self._vector_potential(state_object, params)
        return self._sigma(state_object.gr, A_tensor)

    def get_sigma_a(self, state_object, params=None):
        _, A_tensor = self._vector_potential(state_object, params)
        return self._sigma(state_object._r2a(), A_tensor)

    def get_sigma_k(self, state_object, params=None):
        _, A_tensor = self._vector_potential(state_object, params)
        return self._sigma(state_object.gk, A_tensor)

    # ========== Coefficients of the equations ==========

    def r_equation_coefficients(self, state_object, params=None):
        A_history, A_tensor = self._vector_potential(state_object, params)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        #~ expansion_tensor = NambuKeldyshTensor(np.ones(len(A_history)), pauli_channel=0)
        expansion_tensor = get_expansion_tensor(len(A_history))

        return {
            'type1_em_local': {'L': 1j * A_tensor * A_tensor * tau3 * expansion_tensor},
            'type2_em_local': {'R': -1j * (A_tensor * A_tensor) * tau3},
            'type8_em_cross': {'L': +1j * A_tensor * tau3, 'R': A_tensor},
            'type8_em_cross2': {'L': -1j * A_tensor, 'R': A_tensor * tau3},
            'type5_em_1': {'L': 1j * A_tensor * tau3, 'M': A_tensor * tau3, 'R': expansion_tensor},
            'type5_em_2': {'L': -1j * expansion_tensor, 'M': A_tensor * tau3, 'R': A_tensor * tau3},
        }

    def k_equation_coefficients(self, state_object, params=None):
        A_history, A_tensor = self._vector_potential(state_object, params)
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        expansion_tensor = get_expansion_tensor(len(A_history))

        terms = {
            'type1_em_local': {'L': 1j * A_tensor * A_tensor * tau3},
            'type2_em_local': {'R': 1j * (A_tensor * A_tensor) * tau3},
            'type8_em_cross1': {'L': -1j * A_tensor, 'R': A_tensor * tau3},
            'type8_em_cross2': {'L': -1j * A_tensor * tau3, 'R': A_tensor},
        }

        if not np.allclose(A_history, 0):
            gr = state_object.gr
            ga = state_object._r2a()
            thermal = params['thermal']
            gap_tensor = params['gap_tensor']

            terms['type3_em1'] = {'L': 1j * A_tensor * tau3 * gr * A_tensor * tau3}
            terms['type6_em2'] = {'L': -1j * gr * A_tensor * tau3, 'R': A_tensor * tau3}
            terms['type4_em1'] = {'R': -1j * A_tensor * tau3 * ga * A_tensor * tau3}
            terms['type7_em2'] = {'L': 1j * A_tensor * tau3, 'R': A_tensor * tau3 * ga}

            #* g-independent sources: type0 puts -CN(S) on the rhs, hence S = -source
            def em_f_row(dt_shift):
                dt_end = None if dt_shift == 0 else -dt_shift
                A_diff = A_tensor[-1-dt_shift] * expansion_tensor - A_tensor
                return -2j * thermal.thermal_dist[-1-dt_shift:dt_end, :] * (A_diff * A_diff)

            def em_thermal_row(dt_shift):
                dt_end = None if dt_shift == 0 else -dt_shift
                #~ term1_left = -2j * thermal.convolve_thermal_dist_left(A_tensor[-1-dt_shift] * tau3 * gr[-1-dt_shift:dt_end, :] * A_tensor, time_index=-1-dt_shift)
                #~ term2_left = +2j * thermal.convolve_thermal_dist_left(gr[-1-dt_shift:dt_end, :] * A_tensor * tau3, time_index=-1-dt_shift) * A_tensor
                #~ term1and2_right = -2j * thermal.convolve_thermal_dist_right(A_tensor[-1-dt_shift] * tau3 * A_tensor * ga - A_tensor * ga * A_tensor * tau3, time_index=-1-dt_shift)
                #* full(X*tau3) = 2*dist(X) + int(X*tau3): same dist part as before, plus the F (gap) part
                term1_left = -1j * thermal.convolve_thermal_full_left(A_tensor[-1-dt_shift] * tau3 * gr[-1-dt_shift:dt_end, :] * A_tensor * tau3, gap_tensor, time_index=-1-dt_shift)
                term2_left = +1j * thermal.convolve_thermal_full_left(gr[-1-dt_shift:dt_end, :] * A_tensor * tau3 * tau3, gap_tensor, time_index=-1-dt_shift) * A_tensor
                term1and2_right = -1j * thermal.convolve_thermal_full_right(tau3 * (A_tensor[-1-dt_shift] * tau3 * A_tensor * ga - A_tensor * ga * A_tensor * tau3), gap_tensor, time_index=-1-dt_shift)
                return term1_left + term2_left + term1and2_right

            terms['type0_em_f'] = {'S': -_two_row_source(em_f_row)}
            terms['type0_em_thermal'] = {'S': -_two_row_source(em_thermal_row)}

        return terms
