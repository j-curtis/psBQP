"""
Self-energy classes for various scattering mechanisms in Keldysh formalism.
Abstract base class defines interface for computing Sigma^R and Sigma^K.
"""

from abc import ABC, abstractmethod
import numpy as np
from nambu_keldysh_class import NambuKeldyshTensor


class SelfEnergy(ABC):
    """Abstract base class for self-energy calculations."""

    def __init__(self, scattering_rate: float, omega_arr):
        """Initialize self-energy with scattering rate and frequency grid."""
        pass

    @abstractmethod
    def _sigma_r(self, gr, f):
        """Compute retarded self-energy Sigma^R from Green's functions."""
        pass

    @abstractmethod
    def _sigma_k(self, gr, f):
        """Compute Keldysh self-energy Sigma^K from Green's functions."""
        pass

    @abstractmethod
    def _sigma_shape(self):
        """Return shape of self-energy tensor."""
        pass

    @abstractmethod
    def _get_sigma_indicies(self):
        """Return which Pauli indices are non-zero."""
        pass

    def _self_consistency(self, system_state, sigma):
        """Compute self-consistency residual: sigma - Sigma[g]."""
        pass

    def _unflatten_sigma(self, data):
        """Reconstruct self-energy Nambu tensor from flattened data."""
        pass

    def _unflatten_nambu_object(self, data, data_shape, included_indices=(0, 1, 2, 3)):
        """Helper for unflattening."""
        pass


class ElasticScattering(SelfEnergy):
    """Elastic impurity scattering (angle-averaged)."""

    def __init__(self, scattering_rate: float, theta_arr, omega_arr):
        """Initialize elastic scattering self-energy."""
        pass

    def _sigma_r(self, system_state):
        """Retarded self-energy for elastic scattering."""
        pass

    def _sigma_k(self, system_state):
        """Keldysh self-energy for elastic scattering."""
        pass

    def _sigma_shape(self):
        """Shape of elastic scattering self-energy."""
        pass

    def _get_sigma_indicies(self):
        """Pauli indices for elastic scattering."""
        pass

    def _sigma_jacobian(self, system_state, z):
        """Jacobian for Newton-Raphson solver."""
        pass


class DynesScattering(SelfEnergy):
    """Dynes pair-breaking parameter (phenomenological broadening)."""

    def __init__(self, scattering_rate: float, theta_arr, omega_arr):
        """Initialize Dynes scattering."""
        pass

    def _sigma_r(self, system_state):
        """Retarded self-energy: -i*Gamma*tau_3."""
        pass

    def _sigma_k(self, system_state):
        """Keldysh self-energy for Dynes."""
        pass

    def _sigma_shape(self):
        """Shape: (1, 1) - frequency/angle independent."""
        pass

    def _get_sigma_indicies(self):
        """Only tau_3 component."""
        pass

    def _sigma_jacobian(self, system_state, z):
        """Jacobian (constant, so returns zeros)."""
        pass


class PhononScattering(SelfEnergy):
    """Phonon-mediated scattering with energy-dependent kernel."""

    def __init__(self, scattering_rate: float, omega_arr, temperature):
        """Initialize phonon scattering with temperature."""
        pass

    def _sigma_r(self, gr, f):
        """Retarded self-energy with phonon kernel."""
        pass

    def _sigma_k(self, gr, f):
        """Keldysh self-energy with phonon kernel."""
        pass

    def _sigma_shape(self):
        """Shape: (n_omega,) - frequency dependent."""
        pass

    def _get_sigma_indicies(self):
        """Y and Z Pauli components."""
        pass
