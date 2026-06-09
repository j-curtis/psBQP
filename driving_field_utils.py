"""
Driving field utilities for psBQP-keldysh simulations.

Provides interface for specifying time-dependent driving fields
(vector potentials) with various functional forms.

Supported field types:
    - 'constant': Constant amplitude field
    - 'gaussian': Gaussian pulse (optionally modulated by cosine oscillation)
    - 'oscillatory': Cosine field
    - 'DC': Linear ramp
    - 'step': Step function at specified time
"""

import numpy as np


def compute_driving_field(field_type: str, params: dict, times: np.ndarray) -> np.ndarray:
    """
    Compute time-dependent driving field from explicit parameters.

    Generates vector potential A(t) array from field type and parameters.

    Args:
        field_type: Type of driving field ('constant', 'gaussian', 'oscillatory', 'DC', 'step')
        params: Dictionary of field-specific parameters
        times: Time array (1D numpy array, typically from np.arange(num_timesteps) * dt)

    Returns:
        Vector potential array A(t) with shape matching times, dtype=complex

    Supported field types:

        'constant': Constant amplitude field
            Parameters:
                - amplitude (complex): Constant field value
            Formula: A(t) = amplitude

        'gaussian': Gaussian pulse (optionally modulated by cosine oscillation) centered at T/2
            Parameters:
                - amplitude (complex): Peak amplitude
                - FWHM (float): Full width at half maximum of Gaussian envelope
                - frequency (float, optional): Carrier frequency. If 0 or not provided, returns pure Gaussian
                - phase (float, optional): Phase offset in radians (default: 0, gives A(0)=amplitude for cos)
            Formula:
                - If frequency provided: A(t) = amplitude * exp(-4*ln(2) * (t-T/2)²/FWHM²) * cos(2π * frequency * t + phase)
                - If no frequency: A(t) = amplitude * exp(-4*ln(2) * (t-T/2)²/FWHM²)

        'oscillatory': Cosine field
            Parameters:
                - amplitude (complex): Oscillation amplitude
                - frequency (float): Oscillation frequency
                - phase (float, optional): Phase offset in radians (default: 0, gives A(0)=amplitude)
            Formula: A(t) = amplitude * cos(2π * frequency * t + phase)

        'DC': Linear ramp (DC bias)
            Parameters:
                - amplitude (complex): Ramp rate (field per unit time)
            Formula: A(t) = amplitude * t

        'step': Step function (Heaviside) at specified time
            Parameters:
                - amplitude (complex): Field amplitude after step
                - t_step (float): Time at which step occurs
            Formula: A(t) = 0 for t < t_step, amplitude for t ≥ t_step

    Raises:
        ValueError: If field_type is unknown or required parameters are missing
        TypeError: If parameter types are incorrect
    """

    # Dispatch to field-specific function
    if field_type == 'constant':
        return _compute_constant_field(times, params)
    elif field_type == 'gaussian':
        return _compute_gaussian_field(times, params)
    elif field_type == 'oscillatory':
        return _compute_oscillatory_field(times, params)
    elif field_type == 'DC':
        return _compute_dc_field(times, params)
    elif field_type == 'step':
        return _compute_step_field(times, params)
    else:
        raise ValueError(
            f"Unknown field_type: '{field_type}'. "
            f"Valid types: 'constant', 'gaussian', 'oscillatory', 'DC', 'step'"
        )

def _compute_constant_field(times: np.ndarray, params: dict) -> np.ndarray:
    """Constant field: A(t) = amplitude"""
    amplitude = params['amplitude']
    return np.ones_like(times, dtype=complex) * amplitude


def _compute_oscillatory_field(times: np.ndarray, params: dict) -> np.ndarray:
    """
    Oscillatory field: A(t) = amplitude * cos(2π * frequency * t + phase)

    Default phase=0 gives A(0)=amplitude (cos starts at maximum).
    """
    amplitude = complex(params['amplitude'])
    frequency = float(params['frequency'])
    phase = float(params.get('phase', 0.0))  # Default phase = 0

    return amplitude * np.cos(2.0 * np.pi * frequency * times + phase)


def _compute_dc_field(times: np.ndarray, params: dict) -> np.ndarray:
    """DC ramp: A(t) = amplitude * t"""
    amplitude = complex(params['amplitude'])
    return amplitude * times

def _compute_gaussian_field(times: np.ndarray, params: dict) -> np.ndarray:
    """
    Gaussian pulse (optionally modulated by cosine oscillation), centered at T/2.

    If frequency is provided and non-zero:
        A(t) = amplitude * exp(-4*ln(2) * (t-T/2)²/FWHM²) * cos(2π * frequency * t + phase)

    If frequency is not provided or is zero:
        A(t) = amplitude * exp(-4*ln(2) * (t-T/2)²/FWHM²)

    Default phase=0 gives A(0)=amplitude*envelope(0) for cos.
    """
    amplitude = complex(params['amplitude'])
    FWHM = float(params['FWHM'])
    frequency = float(params.get('frequency', 0.0))  # Default frequency = 0 (no oscillation)
    phase = float(params.get('phase', 0.0))  # Default phase = 0

    # Center time: middle of time window
    T_max = times[-1]
    t_center = T_max / 2.0

    # Gaussian envelope
    coefficient = -4.0 * np.log(2.0) / (FWHM ** 2)
    envelope = np.exp(coefficient * (times - t_center) ** 2)

    # Apply oscillatory carrier only if frequency is non-zero
    if frequency != 0.0:
        carrier = np.cos(2.0 * np.pi * frequency * times + phase)
        return amplitude * envelope * carrier
    else:
        # Pure Gaussian (no oscillation)
        return amplitude * envelope


def _compute_step_field(times: np.ndarray, params: dict) -> np.ndarray:
    """
    Step function (Heaviside):
    A(t) = 0 for t < t_step
    A(t) = amplitude for t ≥ t_step
    """
    amplitude = complex(params['amplitude'])
    t_step = float(params['t_step'])

    # Use np.where for efficient array operation
    return np.where(times >= t_step, amplitude, 0.0 + 0.0j)
