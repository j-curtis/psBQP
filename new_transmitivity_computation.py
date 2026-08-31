import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'equilibrium_python'))

from postprocessed_analysis import load_postprocessed_data
from nambu_class import NambuTensor
from system_state import SupercondctingState


def load_static_jq_data(timestamp, directory_job_index=0, running_machine='laptop'):
    """Load J(Q) data from postprocessed static analysis

    Args:
        timestamp: Timestamp of postprocessed data
        directory_job_index: Index of postprocessed result file
        running_machine: 'laptop' or 'cluster_euler'

    Returns:
        Q_values: Vector potential A (raw, not normalized)
        J_values: Current J (real part, raw)
        T_c: Critical temperature
        sigma_n: Normal conductivity
        A_0: Maximum vector potential (for reference)
    """
    data = load_postprocessed_data(timestamp, directory_job_index, running_machine)

    # Extract the static observables - ensure they are real arrays
    Q_values = np.real(data['averaged_vector_potentials'])  # Vector potential (real part)
    J_values = np.abs(data['averaged_currents'])  # Current (absolute value - must be positive)

    # Get system parameters
    system_params = data['system_parameters'][0]
    T_c = system_params.get('critical_temperature', 1.0)
    sigma_n = system_params.get('normal_conductivity', 1.0)

    # Just store A_0 for reference, but don't normalize Q
    A_0 = np.max(np.abs(Q_values))
    if A_0 < 1e-12:
        A_0 = 1.0

    # Sort by Q in case data is not sorted (use raw values)
    sort_idx = np.argsort(Q_values)
    Q_values_sorted = Q_values[sort_idx]
    J_values_sorted = J_values[sort_idx]

    return Q_values_sorted, J_values_sorted, T_c, sigma_n, A_0


def get_tc_rescaling(timestamp=None):
    """Return T_c rescaling factor (set to 1.0 as specified)"""
    return 1.0

def superfluid_current(timestamp, directory_job_index=0):
    """Load superfluid current data from postprocessed static analysis"""
    Q_vals, J_vals, T_c, sigma_n, A_0 = load_static_jq_data(timestamp, directory_job_index)
    return Q_vals, J_vals

def normal_conductivity(timestamp, directory_job_index=0):
    """Return constant normal conductivity for all Q

    Note: Static analysis doesn't compute dtQ, so we return constant sigma_n
    """
    Q_vals, J_vals, T_c, sigma_n, A_0 = load_static_jq_data(timestamp, directory_job_index)

    # Return constant sigma_n for all Q values
    sigma_n_array = np.full_like(Q_vals, sigma_n)

    return Q_vals, sigma_n_array


def show_density_and_conductivity(timestamp, directory_job_index=0, use_triangular=False):
    """Plot equilibrium properties and save to file

    Args:
        timestamp: Timestamp of postprocessed data
        directory_job_index: Index of postprocessed result file
        use_triangular: If True, overlay triangular characteristic (default: False)
    """
    Q_vals, J_vals, T_c, sigma_n, A_0 = load_static_jq_data(timestamp, directory_job_index)

    # Load gap data if available
    data = load_postprocessed_data(timestamp, directory_job_index)
    gap_values = np.abs(data['averaged_gaps'])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot current characteristic
    if use_triangular:
        # Construct triangular characteristic
        Q_c, slope = construct_triangular_characteristic(Q_vals, J_vals)
        Q_dense = np.linspace(Q_vals[0], Q_vals[-1], 500)
        I_triangular = triangular_current(Q_dense, Q_c, slope)

        ax1.plot(Q_vals, J_vals, 'o', alpha=0.3, label='Full data', markersize=3)
        ax1.plot(Q_dense, I_triangular, 'r-', linewidth=2, label=f'Triangular (Q_c={Q_c:.3f})')
        ax1.legend()
    else:
        ax1.plot(Q_vals, J_vals)

    ax1.set_xlabel(r'$A/A_0$')
    ax1.set_ylabel(r'$J$')
    ax1.set_title('Current vs Normalized Vector Potential')
    ax1.grid(True)

    ax2.plot(Q_vals, np.full_like(Q_vals, sigma_n))
    ax2.set_xlabel(r'$A/A_0$')
    ax2.set_ylabel(r'$\sigma_n$')
    ax2.set_title('Normal Conductivity (constant)')
    ax2.grid(True)

    ax3.plot(Q_vals, gap_values)
    ax3.set_xlabel(r'$A/A_0$')
    ax3.set_ylabel(r'$|\Delta|$')
    ax3.set_title('Gap vs Vector Potential')
    ax3.grid(True)

    # Fourth panel: show J vs A in absolute units
    ax4.plot(Q_vals * A_0, J_vals)
    ax4.set_xlabel(r'$A$')
    ax4.set_ylabel(r'$J$')
    ax4.set_title('Current vs Vector Potential (absolute units)')
    ax4.grid(True)

    plt.tight_layout()

    output_filename = f'timestamp_{timestamp}_equilibrium_properties.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_filename}")
    plt.close()

    return output_filename

def linear_interpolator(x_array, y_array, x_val):
    """Linear interpolation of y_array at position x_val

    Args:
        x_array: Array of x coordinates (must be sorted)
        y_array: Array of y values corresponding to x_array
        x_val: Value at which to interpolate (scalar or array)

    Returns:
        Interpolated value(s) at x_val
    """
    x_array = jnp.asarray(x_array)
    y_array = jnp.asarray(y_array)
    x_val = jnp.asarray(x_val)

    scalar_input = x_val.ndim == 0
    x_val = jnp.atleast_1d(x_val)

    def interpolate_single(x):
        idx = jnp.searchsorted(x_array, x)

        idx = jnp.clip(idx, 1, len(x_array) - 1)

        x0 = x_array[idx - 1]
        x1 = x_array[idx]
        y0 = y_array[idx - 1]
        y1 = y_array[idx]

        slope = (y1 - y0) / (x1 - x0)
        y_interp = y0 + slope * (x - x0)

        below = x < x_array[0]
        above = x > x_array[-1]
        y_interp = jnp.where(below, y_array[0], y_interp)
        y_interp = jnp.where(above, y_array[-1], y_interp)

        return y_interp

    if jax.__version__ >= "0.4.0":
        result = jax.vmap(interpolate_single)(x_val)
    else:
        result = jnp.array([interpolate_single(x) for x in x_val])

    if scalar_input:
        return result[0]
    return result


def ivp_solver(equation_rhs, y0, t_span, t_eval=None, args=None, method='RK45', **kwargs):
    """Solve initial value problem dy/dt = f(t, y)

    Args:
        equation_rhs: Right-hand side function f(t, y, *args)
        y0: Initial condition(s)
        t_span: Tuple (t_start, t_end)
        t_eval: Array of times at which to store solution (optional)
        args: Additional arguments to pass to equation_rhs
        method: Integration method ('RK45', 'DOP853', 'BDF', etc.)
        **kwargs: Additional arguments for solve_ivp

    Returns:
        Solution object with .t and .y attributes
    """
    if args is None:
        args = ()

    def rhs_wrapper(t, y):
        return equation_rhs(t, y, *args)

    solution = solve_ivp(
        rhs_wrapper,
        t_span,
        y0,
        t_eval=t_eval,
        method=method,
        **kwargs
    )

    return solution

def construct_triangular_characteristic(Q_vals, I_s_vals):
    """Construct triangular current characteristic from loaded data

    Args:
        Q_vals: Q values from equilibrium scan
        I_s_vals: Superfluid current values from equilibrium scan

    Returns:
        Q_c: Critical Q where current reaches maximum
        slope: Initial slope dI/dQ at Q=0
    """
    # Find Q_c where current reaches maximum (critical current)
    # Only consider Q > 0.05 to avoid issues near Q=0
    valid_mask = Q_vals > 0.05
    Q_vals_valid = Q_vals[valid_mask]
    I_s_vals_valid = I_s_vals[valid_mask]

    max_idx = np.argmax(np.abs(I_s_vals_valid))
    Q_c = Q_vals_valid[max_idx]

    # Compute initial slope from the first few points
    # Use finite difference on the first 5 points to get stable estimate
    n_points = min(3, len(Q_vals))
    slope = np.polyfit(Q_vals[:n_points], I_s_vals[:n_points], 1)[0]

    return Q_c, slope


def triangular_current(Q, Q_c, slope):
    """Triangular current with smooth tanh rolloff near Q_c

    Args:
        Q: Phase difference (scalar or array)
        Q_c: Critical Q value
        slope: Initial slope (for positive Q)

    Returns:
        Current with smooth cutoff: I = slope * Q * [1 - tanh((Q - Q_c)/width)]
    """
    Q = np.asarray(Q)
    Q_abs = np.abs(Q)
    Q_sign = np.sign(Q)

    # Smooth cutoff with narrow width (preserves slope at small Q)
    width = 0.1 * Q_c  # Narrow transition region
    cutoff_factor = 0.5 * (1.0 - np.tanh((Q_abs - Q_c) / width))

    current_abs = slope * Q_abs * cutoff_factor
    current = Q_sign * current_abs

    return current


def real_time_evolve(timestamp, amplitude, t_pulse, z_t, R_n, R_c, t_span=None, t_center=0.0, Q0=0.0, n_points=1000,
                     use_triangular=False, directory_job_index=0):
    """Real-time evolution of Q under Gaussian current pulse

    Args:
        timestamp: Timestamp of postprocessed data
        amplitude: Amplitude of the Gaussian pulse
        t_pulse: Pulse duration
        z_t: Transmission line impedance
        R_n: Normal resistance
        R_c: Contact resistance
        t_span: Tuple (t_start, t_end) for integration (default: centered around pulse)
        t_center: Center time of the Gaussian pulse (default: 0.0)
        Q0: Initial condition for Q (default: 0.0)
        n_points: Number of time points to evaluate (default: 1000)
        use_triangular: If True, use triangular current characteristic instead of full data (default: False)
        directory_job_index: Index of postprocessed result file

    Returns:
        t_array: Time array
        Q_array: Q(t) solution
        pulse_array: Incoming pulse values at each time
        current_array: Superfluid current values at each time
    """

    Q_vals_sf, I_s_values = superfluid_current(timestamp, directory_job_index)
    Q_vals_nc, sigma_n_values = normal_conductivity(timestamp, directory_job_index)

    # Compute C_0 first
    C_0 = float(np.max(np.abs(I_s_values)))

    # Normalize currents to be dimensionless (like old code)
    I_s_values_normalized = I_s_values / C_0

    # Construct triangular characteristic if requested
    if use_triangular:
        Q_c, slope = construct_triangular_characteristic(Q_vals_sf, I_s_values_normalized)

    if t_span is None:
        # Match old code: asymmetric window [-2*time_duration, time_duration]
        # For old code: time_duration = 10.0 → [-20, 10]
        time_duration = 3.0
        t_span = (t_center - 2 * time_duration, t_center + time_duration)

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    def incoming_pulse(t):
        return amplitude * np.exp(-(t - t_center)**2 * np.log(2)/(t_pulse)**2)

    if use_triangular:
        # Use triangular characteristic WITHOUT latching in dynamics
        # Latching will be applied in post-processing for robustness
        def dQ_dt(t, Q, Q_vals_nc, sigma_n_vals, Q_c_tri, slope_tri):
            Q_scalar = float(Q[0]) if hasattr(Q, '__len__') else float(Q)

            # Use triangular current (no latching during dynamics)
            I_s_Q = triangular_current(Q_scalar, Q_c_tri, slope_tri)

            sigma_n_Q = np.interp(np.abs(Q_scalar), Q_vals_nc, sigma_n_vals)
            pulse = incoming_pulse(t)

            contact_factor = 1 + R_c / (2 * z_t)
            denominator = 1 + (2 * z_t / R_n) * sigma_n_Q * contact_factor
            numerator = (pulse - I_s_Q * contact_factor) * (2 * z_t / R_n) * C_0

            return np.array([numerator / denominator])
    else:
        # Use full interpolated characteristic (enforce odd symmetry: I(-Q) = -I(Q))
        def dQ_dt(t, Q, Q_vals_sf, I_s_vals_norm, Q_vals_nc, sigma_n_vals):
            Q_scalar = float(Q[0]) if hasattr(Q, '__len__') else float(Q)

            # Enforce odd symmetry: evaluate at |Q| and multiply by sign(Q)
            Q_abs = np.abs(Q_scalar)
            Q_sign = np.sign(Q_scalar) if Q_scalar != 0 else 1.0
            # I_s_vals_norm is already dimensionless (normalized by C_0)
            I_s_Q = Q_sign * np.interp(Q_abs, Q_vals_sf, np.abs(I_s_vals_norm))

            sigma_n_Q = np.interp(Q_abs, Q_vals_nc, sigma_n_vals)
            pulse = incoming_pulse(t)

            contact_factor = 1 + R_c / (2 * z_t)
            denominator = 1 + (2 * z_t / R_n) * sigma_n_Q * contact_factor
            numerator = (pulse - I_s_Q * contact_factor) * (2 * z_t / R_n) * C_0

            return np.array([numerator / denominator])

    if use_triangular:
        args = (np.array(Q_vals_nc), np.array(sigma_n_values), Q_c, slope)
        initial_state = [Q0]  # Just Q, no flag
    else:
        args = (np.array(Q_vals_sf), np.array(I_s_values_normalized),
                np.array(Q_vals_nc), np.array(sigma_n_values))
        initial_state = [Q0]

    # Force small time steps to capture pulse dynamics
    max_step = t_pulse / 2.0  # Step size ~ half the pulse duration

    solution = ivp_solver(
        dQ_dt,
        initial_state,
        t_span,
        t_eval=t_eval,
        args=args,
        method='RK23',
        rtol=1e-6,
        atol=1e-9,
        max_step=max_step
    )

    Q_array = solution.y[0]
    pulse_array = np.array([incoming_pulse(t) for t in solution.t])

    if use_triangular:
        # Compute triangular current with smooth rolloff (no latching needed)
        superfluid_current_array = triangular_current(Q_array, Q_c, slope)
    else:
        # Enforce odd symmetry: I(Q) = sign(Q) * I(|Q|)
        # Use normalized values (dimensionless)
        superfluid_current_array = np.array([
            np.sign(Q) * np.interp(np.abs(Q), Q_vals_sf, np.abs(I_s_values_normalized))
            for Q in Q_array
        ])

    # Use |Q| for sigma_n interpolation to match dynamics
    dtQ_prefactor_array = np.array([np.interp(np.abs(Q), Q_vals_nc, sigma_n_values) for Q in Q_array])

    dQ_dt_array = np.gradient(Q_array, solution.t)
    normal_current_array = dtQ_prefactor_array * dQ_dt_array / C_0
    total_current_array = superfluid_current_array + normal_current_array

    return solution.t, Q_array, pulse_array, superfluid_current_array, normal_current_array, total_current_array


def plot_diagnostic_traces(timestamp, amplitude_list, t_pulse, z_t, R_n, R_c, use_triangular=False, directory_job_index=0):
    """Plot diagnostic time traces for selected amplitudes to debug transmitivity oscillations

    Args:
        timestamp: Timestamp of postprocessed data
        amplitude_list: List of amplitudes to plot
        t_pulse: Pulse duration
        z_t: Transmission line impedance
        R_n: Normal resistance
        R_c: Contact resistance
        use_triangular: Whether to use triangular characteristic
        directory_job_index: Index of postprocessed result file
    """
    fig, axes = plt.subplots(len(amplitude_list), 4, figsize=(18, 4*len(amplitude_list)))
    if len(amplitude_list) == 1:
        axes = axes.reshape(1, -1)

    # Load data to get Q_c if triangular
    if use_triangular:
        Q_vals, I_vals = superfluid_current(timestamp, directory_job_index)
        Q_c, slope = construct_triangular_characteristic(Q_vals, I_vals)
    else:
        Q_c = None

    for idx, amp in enumerate(amplitude_list):
        t, Q, pulse, sf_curr, norm_curr, total_curr = real_time_evolve(
            timestamp, amp, t_pulse, z_t, R_n, R_c, use_triangular=use_triangular, n_points=1000,
            directory_job_index=directory_job_index
        )

        # Q(t)
        axes[idx, 0].plot(t, Q, 'b-', linewidth=2)
        if Q_c is not None:
            axes[idx, 0].axhline(Q_c, color='r', linestyle='--', label=f'Q_c={Q_c:.3f}')
            axes[idx, 0].axhline(-Q_c, color='r', linestyle='--')
            axes[idx, 0].legend()
        axes[idx, 0].set_ylabel('Q(t)')
        axes[idx, 0].set_title(f'Amplitude = {amp:.3f}')
        axes[idx, 0].grid(True)

        # Currents
        axes[idx, 1].plot(t, pulse, 'k--', label='Incoming', alpha=0.5)
        axes[idx, 1].plot(t, sf_curr, 'b-', label='Superfluid', linewidth=2)
        axes[idx, 1].plot(t, norm_curr, 'r-', label='Normal', linewidth=2)
        axes[idx, 1].plot(t, total_curr, 'g-', label='Total', linewidth=2)
        axes[idx, 1].set_ylabel('Current')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)

        # Superfluid vs normal
        axes[idx, 2].plot(t, sf_curr, 'b-', label='Superfluid', linewidth=2)
        axes[idx, 2].plot(t, norm_curr, 'r-', label='Normal', linewidth=2)
        axes[idx, 2].set_ylabel('Current')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True)

        # Mark when Q exceeds Q_c
        if Q_c is not None:
            exceeded = np.abs(Q) > Q_c
            axes[idx, 3].plot(t, exceeded.astype(float), 'r-', linewidth=2)
            axes[idx, 3].set_ylabel('Q > Q_c')
            axes[idx, 3].set_ylim([-0.1, 1.1])
            if np.any(exceeded):
                t_exceed = t[exceeded][0]
                axes[idx, 3].axvline(t_exceed, color='k', linestyle='--',
                                    label=f't_exceed={t_exceed:.3f}')
                axes[idx, 3].legend()
        axes[idx, 3].grid(True)

        if idx == len(amplitude_list) - 1:
            for ax in axes[idx, :]:
                ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('transmitivity_diagnostic.png', dpi=150, bbox_inches='tight')
    print("Saved diagnostic plot to: transmitivity_diagnostic.png")
    plt.close()


if __name__ == "__main__":

    # Load data from postprocessed timestamp
    timestamp = '1784534890'
    directory_job_index = 0

    # Circuit parameters (updated as specified)
    z_t = 50.0
    R_n = 170.0
    R_c = 0.0  # Changed from 6.0 to 0.0

    # Load J(Q) data to get system parameters
    Q_vals, J_vals, T_c, sigma_n, A_0 = load_static_jq_data(timestamp, directory_job_index)

    # System parameters (updated as specified)
    # T = 0.2, eta = 0.1 (these should be in the postprocessed data)
    # tc_rescaling = 1.0
    tc_rescaling = 1.0
    T_c = 15.5
    # Pulse duration (divided by 4 as specified)
    t_pulse = (0.020 * T_c * 2 * np.pi) / 4.0

    # Choose current characteristic model
    use_triangular = False  # Set to True to use triangular I-V characteristic
    amplitude = 0.05

    print(f"Loaded data from timestamp: {timestamp}")
    print(f"T_c = {T_c:.6f}")
    print(f"sigma_n = {sigma_n:.6f}")
    print(f"A_0 = {A_0:.6f}")
    print(f"t_pulse = {t_pulse:.6f}")
    print(f"R_c = {R_c:.6f}") 

    print("Generating equilibrium property plots...")
    show_density_and_conductivity(timestamp, directory_job_index, use_triangular=use_triangular)

    print("\nRunning real-time evolution for weak pulse...")
    t_array, Q_array, pulse_array, sf_current, normal_current, total_current = real_time_evolve(
        timestamp,
        amplitude=amplitude,
        t_pulse=t_pulse,
        z_t=z_t,
        R_n=R_n,
        R_c=R_c,
        t_center=0.0,
        Q0=0.0,
        n_points=2000,
        use_triangular=use_triangular,
        directory_job_index=directory_job_index
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t_array, Q_array)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Q(t)')
    axes[0, 0].set_title('Phase Evolution')
    axes[0, 0].grid(True)

    # Compute linear response: FFT(pulse) * Z(omega) -> IFFT

    # Extract equilibrium properties at Q=0 for impedance calculation
    Q_vals_sf, I_s_vals = superfluid_current(timestamp, directory_job_index)
    Q_vals_nc, sigma_n_vals = normal_conductivity(timestamp, directory_job_index)

    # Superfluid current slope at Q=0 (use first few points for linear fit)
    n_points = min(3, len(Q_vals_sf))
    sf_slope_at_zero = np.polyfit(Q_vals_sf[:n_points], I_s_vals[:n_points], 1)[0]
    print(sf_slope_at_zero)
    # Normal conductivity (dtQ prefactor) at Q=0
    sigma_n_at_zero = np.interp(0.0, Q_vals_nc, sigma_n_vals)

    print(f"Equilibrium properties at Q=0:")
    print(f"  Superfluid slope dI_s/dQ = {sf_slope_at_zero:.6f}")
    print(f"  Normal conductivity sigma_n = {sigma_n_at_zero:.6f}")

    # no impedance
    Z_omega = lambda omega: 2*z_t/R_n * (sf_slope_at_zero + 1j * omega * 1)/ ( (sf_slope_at_zero + 1j * omega * 1)* (1 + R_c/2/z_t) * 2*z_t/R_n + 1j * omega)  # Placeholder impedance function (modify as needed)

    # FFT of incoming pulse
    pulse_fft = np.fft.fft(pulse_array)
    freqs = np.fft.fftfreq(len(t_array), d=(t_array[1] - t_array[0]))
    omega_array = 2 * np.pi * freqs

    # Apply impedance in frequency domain
    Z_values = np.array([Z_omega(omega) for omega in omega_array])
    transmitted_fft = pulse_fft * Z_values

    # Inverse FFT to get linear response in time domain
    linear_response = np.fft.ifft(transmitted_fft).real

    axes[0, 1].plot(t_array, pulse_array, label='Incoming pulse', linewidth=3)
    axes[0, 1].plot(t_array, total_current, label='Total current (full)', linewidth=3)
    axes[0, 1].plot(t_array, linear_response, label='linear response', linewidth=1.5)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Current')
    axes[0, 1].set_title('Total Current vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(t_array, sf_current, label='Superfluid', linewidth=2)
    axes[1, 0].plot(t_array, normal_current, label='Normal', linewidth=2)
    axes[1, 0].plot(t_array, sf_current + normal_current, label='Total', linewidth=2)
    #axes[1,0].set_ylim([0,0.12])

    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Current')
    axes[1, 0].set_title('Current Components -- new')
    axes[1, 0].legend()
    axes[1, 0].grid(False)

    axes[1, 1].plot(t_array, pulse_array - total_current)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('I_in - I_total')
    axes[1, 1].set_title('Current Difference')
    axes[1, 1].grid(True)

    plt.tight_layout()

    output_filename = f'timestamp_{timestamp}_realtime_evolution.png'
    #plt.show()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved real-time evolution plot to: {output_filename}")

    print("\n" + "="*60)
    print("Running transmitivity sweep...")
    print("="*60)

    # Load equilibrium data for inclusion in transmitivity output
    eq_data = load_postprocessed_data(timestamp, directory_job_index)

    q_scan_Qs = Q_vals  # Use the Q values from static analysis
    q_scan_currents = J_vals  # Use the J values from static analysis
    q_scan_gaps = np.abs(eq_data['averaged_gaps'])  # Gap values

    # Compute triangular parameters if using triangular model
    if use_triangular:
        Q_c_tri, slope_tri = construct_triangular_characteristic(q_scan_Qs, q_scan_currents)
    else:
        Q_c_tri, slope_tri = None, None

    amplitude_list = np.linspace(0.05, 10.0, 100)
    transmitivity_list = []
    max_incoming_list = []
    max_transmitted_list = []

    from tqdm import tqdm
    for amp in tqdm(amplitude_list, desc="Amplitude sweep"):
        t_arr, Q_arr, pulse_arr, sf_curr, norm_curr, total_curr = real_time_evolve(
            timestamp,
            amplitude=amp,
            t_pulse=t_pulse,
            z_t=z_t,
            R_n=R_n,
            R_c=R_c,
            t_center=0.0,
            Q0=0.0,
            n_points=5000,
            use_triangular=use_triangular,
            directory_job_index=directory_job_index
        )

        max_incoming = np.max(np.abs(pulse_arr))
        max_transmitted = np.max(np.abs(total_curr))
        transmitivity = max_transmitted / max_incoming

        max_incoming_list.append(max_incoming)
        max_transmitted_list.append(max_transmitted)
        transmitivity_list.append(transmitivity)

    max_incoming_list = np.array(max_incoming_list)
    max_transmitted_list = np.array(max_transmitted_list)
    transmitivity_list = np.array(transmitivity_list)

    transmitivity_data = {
        'timestamp': timestamp,
        'amplitude_list': amplitude_list,
        'max_incoming': max_incoming_list,
        'max_transmitted': max_transmitted_list,
        'transmitivity': transmitivity_list,
        't_pulse': t_pulse,
        'T_c': T_c,
        'sigma_n': sigma_n,
        'A_0': A_0,
        'R_c': R_c,
        'R_n': R_n,
        'z_t': z_t,
        'q_scan_Qs': q_scan_Qs,
        'q_scan_currents': q_scan_currents,
        'q_scan_gaps': q_scan_gaps,
        'use_triangular': use_triangular,
        'Q_c': Q_c_tri,
        'slope': slope_tri
    }

    data_filename = f'timestamp_{timestamp}_transmitivity_data.pkl'
    with open(data_filename, 'wb') as f:
        pickle.dump(transmitivity_data, f)
    print(f"\nSaved transmitivity data to: {data_filename}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(max_incoming_list, transmitivity_list, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Incoming Current Amplitude')
    ax1.set_ylabel('Transmitivity')
    ax1.set_title('Transmitivity vs Incoming Current')
    ax1.grid(True)

    ax2.plot(max_transmitted_list, transmitivity_list, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Transmitted Current Amplitude')
    ax2.set_ylabel('Transmitivity')
    ax2.set_title('Transmitivity vs Transmitted Current')
    ax2.grid(True)

    plt.tight_layout()

    transmitivity_filename = f'timestamp_{timestamp}_transmitivity.png'
    plt.savefig(transmitivity_filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved transmitivity plot to: {transmitivity_filename}")
    plt.close()

    # Optional: Plot diagnostic traces to understand oscillations
    # Uncomment to generate detailed time traces for selected amplitudes
    diagnostic_amps = [0.5, 1.0, 2.5, 2.6]  # Choose amplitudes where oscillations occur
    print("\nGenerating diagnostic traces...")
    plot_diagnostic_traces(timestamp, diagnostic_amps, t_pulse, z_t, R_n, R_c, use_triangular=use_triangular,
                          directory_job_index=directory_job_index)

    print("\nDone!")
