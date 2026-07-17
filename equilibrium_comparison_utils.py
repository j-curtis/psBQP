"""
Equilibrium conductivity computation utilities.

Provides functions to compute equilibrium optical conductivity σ(ω) from
simulation parameters using the Keldysh-Usadel formalism.
"""

import numpy as np
import os
import sys
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_analysis import load_job_data
from equilibrium_class import EquilibriumSolver


def _extract_pauli_tau3(nambu_tensor):
    """
    Extract τ₃ Pauli component from NambuTensor.

    Args:
        nambu_tensor: Old NambuTensor from equilibrium solver

    Returns:
        numpy array of τ₃ component (shape: N_omega)
    """
    return np.array(nambu_tensor._trace(pauli_index=3)) / 2.0


def compute_equilibrium_conductivity(timestamp, job_index=0, running_machine='laptop'):
    """
    Compute equilibrium optical conductivity σ(ω) from simulation parameters.

    Reads gap, temperature, eta, and dt from saved simulation data, then computes
    equilibrium Green's functions and evaluates the conductivity integral:

    σ(ω) = -iπ/4 * ∫ dε/(2π) * Tr[τ₃ Gr(ε) τ₃ Gk(ε+ω) + τ₃ Ga(ε) τ₃ Gk(ε-ω)]

    Args:
        timestamp: Simulation timestamp folder name
        job_index: Job array index (default 0)
        running_machine: 'laptop' or 'cluster_euler' (default 'laptop')

    Returns:
        omega_grid: Probe frequency array (shape: N_omega)
        conductivity: Complex conductivity σ(ω) (shape: N_omega)
    """

    # Step 1: Load simulation data and extract parameters
    input_kwargs, save_data = load_job_data(timestamp, job_index, running_machine)

    system_params = input_kwargs['system_parameters']
    T_c = system_params['critical_temperature']
    temperature = system_params['temperature']
    eta = system_params['eta']

    # Validate eta
    if eta <= 0:
        raise ValueError(f"Broadening parameter eta must be positive, got {eta}")

    gap = np.abs(save_data['gaps'][0])

    # Extract time grid parameters
    if 'final_state' in save_data and save_data['final_state'] is not None:
        state = save_data['final_state']
        N_t = state.gr.data.shape[-1]
        T_max = state.T_max
        dt = state.dt if state.dt is not None else (T_max / (N_t - 1))
    else:
        # Fallback to input_kwargs
        grid_params = input_kwargs.get('grid_parameters', {})
        N_t = grid_params.get('time_sampling', grid_params.get('n_tpoints'))
        T_max = grid_params.get('time_duration', grid_params.get('t_max'))

        if N_t is None or T_max is None:
            raise ValueError("Could not extract time grid parameters from simulation data")

        dt = T_max / (N_t - 1)

    # Step 2: Construct epsilon grid using FFT conventions from dt and T_max
    # This matches the frequency grid from Fourier transform of the time grid
    N_t = N_t * 10
    freq = np.fft.fftfreq(N_t, d=dt)
    omega_fft = 2 * np.pi * freq
    epsilon_grid = np.fft.fftshift(omega_fft)

    epsilon_max = np.max(np.abs(epsilon_grid))

    grid_parameters = {
        'omega_grid': epsilon_grid,
        'energy_cutoff': epsilon_max,
        'eta': eta
    }

    system_parameters_eq = {
        'critical_temperature': T_c,
        'temperature': temperature,
        'current_maximum': 1.0
    }

    solver = EquilibriumSolver(
        grid_parameters=grid_parameters,
        system_parameters=system_parameters_eq,
        optimization_parameters=None,
        sigma_scatterings=None
    )

    # Step 4: Compute equilibrium Green's functions directly using the gap
    print(f"Computing equilibrium Green's functions...")
    print(f"  Gap: {gap:.4f}, Temperature: {temperature:.4f}, eta: {eta:.4f}")

    # Access the underlying UsadelEvolution object
    usadel = solver.usadel_solver

    # Get thermal occupation function
    import jax.numpy as jnp
    f_thermal = usadel._thermal_occupation_numbers(temperature)

    # Import NambuTensor from old code
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equilibrium_python'))
    from nambu_class import NambuTensor

    f_thermal_tensor = NambuTensor(f_thermal, 0)

    # Construct gr directly using the gap (no self-consistency)
    hr_eq = usadel._get_hr(Q=0.0, delta=gap, sigma_r=None)
    gr_eq = usadel._hr2gr(hr_eq)

    # Compute gk from gr and thermal occupation: gk = gr @ f + f @ gr_involution
    gk_eq = gr_eq @ f_thermal_tensor + f_thermal_tensor @ gr_eq._involution()

    # Compute ga from gr: ga = -gr_involution
    ga_eq = -gr_eq._involution()

    # Step 5: Get epsilon grid that was actually used
    epsilon_grid_actual = solver.usadel_solver.w_arr

    print(f"\nEpsilon grid info:")
    print(f"  Size: {len(epsilon_grid_actual)}")
    print(f"  Range: [{np.min(epsilon_grid_actual):.4f}, {np.max(epsilon_grid_actual):.4f}]")
    print(f"  Spacing: {epsilon_grid_actual[1] - epsilon_grid_actual[0]:.6f}")

    # Select only positive frequencies up to 10*gap
    omega_max_cutoff = 10 * gap
    omega_mask = (np.array(epsilon_grid_actual) >= 0) & (np.array(epsilon_grid_actual) <= omega_max_cutoff)
    omega_indices = np.where(omega_mask)[0]
    omega_grid = np.array(epsilon_grid_actual)[omega_indices]
    n_omega = len(omega_grid)

    print(f"\nComputing conductivity integral...")
    print(f"  Omega range: [0, {omega_max_cutoff:.2f}] (10*gap)")
    print(f"  Omega points: {n_omega}, Epsilon points: {len(epsilon_grid_actual)}")

    # Step 6: Create tau_3 Pauli matrix as NambuTensor
    tau_3 = NambuTensor(jnp.ones(len(epsilon_grid_actual)), pauli_channel=3)

    # Step 7: Compute conductivity integral with proper matrix operations
    conductivity = np.zeros(n_omega, dtype=complex)

    # Find center index (zero energy)
    center_idx = np.argmin(np.abs(epsilon_grid_actual))

    for idx, i in enumerate(tqdm(omega_indices, desc="Computing σ(ω)")):
        # Compute shift relative to center
        shift = i - center_idx

        # Create shifted Gk Green's functions by shifting the data array
        # Gk(ε + ω): shift indices by +shift
        if shift >= 0:
            gk_plus_data = jnp.zeros_like(gk_eq.data)
            gk_plus_data = gk_plus_data.at[:, :, :len(epsilon_grid_actual)-shift].set(gk_eq.data[:, :, shift:])
        else:
            gk_plus_data = jnp.zeros_like(gk_eq.data)
            gk_plus_data = gk_plus_data.at[:, :, -shift:].set(gk_eq.data[:, :, :len(epsilon_grid_actual)+shift])
        gk_plus = NambuTensor(gk_plus_data, None)

        # Gk(ε - ω): shift indices by -shift
        if shift >= 0:
            gk_minus_data = jnp.zeros_like(gk_eq.data)
            gk_minus_data = gk_minus_data.at[:, :, shift:].set(gk_eq.data[:, :, :len(epsilon_grid_actual)-shift])
        else:
            gk_minus_data = jnp.zeros_like(gk_eq.data)
            gk_minus_data = gk_minus_data.at[:, :, :len(epsilon_grid_actual)+shift].set(gk_eq.data[:, :, -shift:])
        gk_minus = NambuTensor(gk_minus_data, None)

        # Perform matrix multiplications: τ₃ @ Gr(ε) @ τ₃ @ Gk(ε+ω)
        term1 = tau_3 @ gr_eq @ tau_3 @ gk_plus

        # Perform matrix multiplications: τ₃ @ Ga(ε) @ τ₃ @ Gk(ε-ω)
        term2 = tau_3 @ ga_eq @ tau_3 @ gk_minus

        # Sum the terms
        integrand_matrix = term1 + term2

        # Take trace with pauli_index=0 (identity = regular trace)
        integrand_trace = integrand_matrix._trace(pauli_index=0)

        # Integrate over ε: dε/(2π)
        integral = np.trapz(np.array(integrand_trace), np.array(epsilon_grid_actual)) / (2.0 * np.pi)

        # Apply prefactor -iπ/4
        conductivity[idx] = -1j * np.pi / 4.0 * integral

    print(f"Conductivity computation complete.")

    return omega_grid, conductivity/(-1j * omega_grid)
