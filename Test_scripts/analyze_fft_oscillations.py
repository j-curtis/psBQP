"""
Analyze FFT of Green's functions to diagnose Gibbs oscillations.

Investigates:
1. Anti-diagonal extraction properties
2. Endpoint discontinuities
3. FFT spectrum analysis
4. Mitigation strategies (windowing, endpoint subtraction)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid LaTeX issues
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demler_tools.file_manager import path_management, io
path_management.initialize(project_name='psBQP-keldysh')

import data_analysis as da


def analyze_anti_diagonal_data(timestamp, job_index=0):
    """Analyze anti-diagonal extraction and FFT from real data."""

    print("="*80)
    print(f"ANALYZING FFT FOR TIMESTAMP: {timestamp}, JOB: {job_index}")
    print("="*80)

    # Load data
    input_kwargs, save_data = da.load_job_data(timestamp, job_index)
    state = save_data['final_state']

    # Get Green's function
    gr = state.gr
    N_t = gr.data.shape[2]
    dt = state.dt

    print(f"\nGrid parameters:")
    print(f"  N_t = {N_t}")
    print(f"  dt = {dt}")
    print(f"  T_max = {state.T_max}")
    print(f"  Anti-diagonal spacing: Δτ = {2*dt}")

    # Extract anti-diagonal (before reversal)
    g_offdiag = gr.off_diagonal()  # Shape (2, 2, N_t)

    # Analyze for each Pauli component
    pauli_labels = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']

    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.suptitle(f'Anti-Diagonal FFT Analysis: g^R\n(timestamp={timestamp}, job={job_index})',
                 fontsize=14)

    for pauli_idx in range(4):
        print(f"\n{'='*60}")
        print(f"PAULI COMPONENT: {pauli_labels[pauli_idx]}")
        print(f"{'='*60}")

        # Extract Pauli component from anti-diagonal
        # Use same trace operation as in energy_time_representation
        from nambu_keldysh_class import NambuKeldyshTensor
        g_offdiag_tensor = NambuKeldyshTensor(g_offdiag)
        g_pauli = g_offdiag_tensor.trace(pauli_index=pauli_idx) / 2

        # Create relative time axis (before reversal)
        tau_indices = np.arange(N_t)
        tau_before_reversal = ((N_t - 1) - 2*tau_indices) * dt

        # After reversal
        g_pauli_reversed = np.flip(g_pauli)
        tau_after_reversal = -tau_before_reversal[::-1]

        # Analyze endpoints
        print(f"\nBefore reversal:")
        print(f"  τ[0] = {tau_before_reversal[0]:.4f}, g[0] = {g_pauli[0]:.6e}")
        print(f"  τ[-1] = {tau_before_reversal[-1]:.4f}, g[-1] = {g_pauli[-1]:.6e}")
        print(f"  Endpoint discontinuity: |g[-1] - g[0]| = {abs(g_pauli[-1] - g_pauli[0]):.6e}")

        print(f"\nAfter reversal:")
        print(f"  τ[0] = {tau_after_reversal[0]:.4f}, g[0] = {g_pauli_reversed[0]:.6e}")
        print(f"  τ[-1] = {tau_after_reversal[-1]:.4f}, g[-1] = {g_pauli_reversed[-1]:.6e}")
        print(f"  Endpoint discontinuity: |g[-1] - g[0]| = {abs(g_pauli_reversed[-1] - g_pauli_reversed[0]):.6e}")

        # Plot time domain (left column)
        ax_time = axes[pauli_idx, 0]
        ax_time.plot(tau_after_reversal, g_pauli_reversed.real, 'b-', linewidth=1.5, label='Real')
        ax_time.plot(tau_after_reversal, g_pauli_reversed.imag, 'r-', linewidth=1.5, label='Imag')
        ax_time.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax_time.axvline(tau_after_reversal[0], color='green', linestyle=':', alpha=0.5, linewidth=2, label='τ_min')
        ax_time.axvline(tau_after_reversal[-1], color='green', linestyle=':', alpha=0.5, linewidth=2, label='τ_max')
        ax_time.set_xlabel('τ')
        ax_time.set_ylabel(f'{pauli_labels[pauli_idx]}')
        ax_time.set_title('Time Domain (after reversal)')
        ax_time.grid(True, alpha=0.3)
        if pauli_idx == 0:
            ax_time.legend(fontsize=8)

        # FFT (current implementation)
        g_fft = np.fft.ifft(g_pauli_reversed) * N_t * (2*dt)
        g_fft_shifted = np.fft.fftshift(g_fft)

        # Energy grid (current implementation)
        freq = np.fft.fftfreq(N_t, d=2*dt)
        energy = 2 * np.pi * freq
        energy_shifted = np.fft.fftshift(energy)

        # Plot FFT full range (middle column)
        ax_fft_full = axes[pauli_idx, 1]
        ax_fft_full.plot(energy_shifted, g_fft_shifted.real, 'b-', linewidth=1, label='Real')
        ax_fft_full.plot(energy_shifted, g_fft_shifted.imag, 'r-', linewidth=1, label='Imag')
        ax_fft_full.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax_fft_full.set_xlabel('ω')
        ax_fft_full.set_ylabel(f'{pauli_labels[pauli_idx]}')
        ax_fft_full.set_title('FFT: Full Range')
        ax_fft_full.grid(True, alpha=0.3)

        # Plot FFT zoomed (right column)
        ax_fft_zoom = axes[pauli_idx, 2]
        ax_fft_zoom.plot(energy_shifted, g_fft_shifted.real, 'b-', linewidth=1.5, label='Real')
        ax_fft_zoom.plot(energy_shifted, g_fft_shifted.imag, 'r-', linewidth=1.5, label='Imag')
        ax_fft_zoom.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax_fft_zoom.set_xlabel('ω')
        ax_fft_zoom.set_ylabel(f'{pauli_labels[pauli_idx]}')
        ax_fft_zoom.set_title('FFT: Zoomed (-15 to 15)')
        ax_fft_zoom.set_xlim(-15, 15)
        ax_fft_zoom.grid(True, alpha=0.3)
        if pauli_idx == 0:
            ax_fft_zoom.legend(fontsize=8)

        # Quantify oscillations
        # Measure power in different frequency bands
        low_mask = np.abs(energy_shifted) < 5
        mid_mask = (np.abs(energy_shifted) >= 5) & (np.abs(energy_shifted) < 10)
        high_mask = np.abs(energy_shifted) >= 10

        power_low = np.sum(np.abs(g_fft_shifted[low_mask])**2)
        power_mid = np.sum(np.abs(g_fft_shifted[mid_mask])**2)
        power_high = np.sum(np.abs(g_fft_shifted[high_mask])**2)
        power_total = power_low + power_mid + power_high

        print(f"\nSpectral power distribution:")
        print(f"  |ω| < 5:  {power_low:.6e} ({100*power_low/power_total:.1f}%)")
        print(f"  5 < |ω| < 10: {power_mid:.6e} ({100*power_mid/power_total:.1f}%)")
        print(f"  |ω| > 10: {power_high:.6e} ({100*power_high/power_total:.1f}%)")

        # Check for oscillations by looking at derivative
        if pauli_idx in [2, 3]:  # τ₂ and τ₃ most important
            zoom_mask = (energy_shifted >= -15) & (energy_shifted <= 15)
            g_zoom = g_fft_shifted[zoom_mask]
            omega_zoom = energy_shifted[zoom_mask]

            # Count zero crossings as proxy for oscillations
            real_crossings = np.sum(np.diff(np.sign(g_zoom.real)) != 0)
            imag_crossings = np.sum(np.diff(np.sign(g_zoom.imag)) != 0)

            print(f"  Zero crossings in [-15, 15]:")
            print(f"    Real part: {real_crossings}")
            print(f"    Imag part: {imag_crossings}")

    plt.tight_layout()
    save_path = os.path.join('Test_scripts', f'fft_analysis_{timestamp}_job{job_index}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved to: {save_path}")
    print(f"{'='*80}\n")

    return fig


def compare_mitigation_strategies(timestamp, job_index=0, pauli_idx=3):
    """Compare different strategies to mitigate Gibbs oscillations."""

    print("\n" + "="*80)
    print("COMPARING MITIGATION STRATEGIES")
    print("="*80)

    # Load data
    input_kwargs, save_data = da.load_job_data(timestamp, job_index)
    state = save_data['final_state']

    gr = state.gr
    N_t = gr.data.shape[2]
    dt = state.dt

    # Extract and process
    g_offdiag = gr.off_diagonal()
    from nambu_keldysh_class import NambuKeldyshTensor
    g_offdiag_tensor = NambuKeldyshTensor(g_offdiag)
    g_pauli = g_offdiag_tensor.trace(pauli_index=pauli_idx) / 2
    g_pauli_reversed = np.flip(g_pauli)

    tau = np.linspace(-(N_t-1)*dt, (N_t-1)*dt, N_t)

    # Strategy 1: No mitigation (current)
    g1 = g_pauli_reversed.copy()

    # Strategy 2: Subtract endpoint average
    endpoint_avg = (g_pauli_reversed[0] + g_pauli_reversed[-1]) / 2
    g2 = g_pauli_reversed - endpoint_avg

    # Strategy 3: Apply Hann window
    window_hann = np.hanning(N_t)
    g3 = g_pauli_reversed * window_hann

    # Strategy 4: Apply Tukey window (alpha=0.5, taper 25% at each end)
    from scipy.signal.windows import tukey
    window_tukey = tukey(N_t, alpha=0.5)
    g4 = g_pauli_reversed * window_tukey

    # Compute FFTs
    freq = np.fft.fftfreq(N_t, d=2*dt)
    energy = 2 * np.pi * freq
    energy_shifted = np.fft.fftshift(energy)

    strategies = [
        (g1, "No mitigation"),
        (g2, "Subtract endpoints"),
        (g3, "Hann window"),
        (g4, "Tukey window (α=0.5)")
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f'Gibbs Oscillation Mitigation Strategies: τ₃ component of g^R\n(timestamp={timestamp}, job={job_index})',
                 fontsize=14)

    for idx, (g_data, label) in enumerate(strategies):
        # Time domain
        ax_time = axes[0, idx]
        ax_time.plot(tau, g_data.real, 'b-', linewidth=1.5, label='Real')
        ax_time.plot(tau, g_data.imag, 'r-', linewidth=1.5, label='Imag')
        ax_time.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax_time.set_xlabel('τ')
        ax_time.set_title(f'{label}\nTime Domain')
        ax_time.grid(True, alpha=0.3)
        if idx == 0:
            ax_time.legend(fontsize=8)

        # Print endpoint info
        print(f"\n{label}:")
        print(f"  Endpoints: g[0]={g_data[0]:.6e}, g[-1]={g_data[-1]:.6e}")
        print(f"  Discontinuity: {abs(g_data[-1] - g_data[0]):.6e}")

        # FFT
        g_fft = np.fft.ifft(g_data) * N_t * (2*dt)
        g_fft_shifted = np.fft.fftshift(g_fft)

        ax_freq = axes[1, idx]
        ax_freq.plot(energy_shifted, g_fft_shifted.real, 'b-', linewidth=1.5, label='Real')
        ax_freq.plot(energy_shifted, g_fft_shifted.imag, 'r-', linewidth=1.5, label='Imag')
        ax_freq.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax_freq.set_xlabel('ω')
        ax_freq.set_title('FFT: |ω| < 15')
        ax_freq.set_xlim(-15, 15)
        ax_freq.grid(True, alpha=0.3)

        # Quantify
        zoom_mask = (energy_shifted >= -15) & (energy_shifted <= 15)
        max_val = np.max(np.abs(g_fft_shifted[zoom_mask]))
        print(f"  Max |g(ω)| for |ω|<15: {max_val:.6e}")

        # High frequency content
        high_mask = np.abs(energy_shifted) > 10
        high_power = np.sum(np.abs(g_fft_shifted[high_mask])**2)
        total_power = np.sum(np.abs(g_fft_shifted)**2)
        print(f"  High freq (|ω|>10) fraction: {100*high_power/total_power:.2f}%")

    plt.tight_layout()
    save_path = os.path.join('Test_scripts', f'mitigation_comparison_{timestamp}_job{job_index}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Comparison plot saved to: {save_path}")
    print(f"{'='*80}\n")

    return fig


if __name__ == '__main__':
    # Analyze specified timestamp
    timestamp = '1781705963'
    job_index = 0

    print("Starting FFT oscillation analysis...")
    print(f"Timestamp: {timestamp}, Job index: {job_index}\n")

    # Detailed analysis
    analyze_anti_diagonal_data(timestamp, job_index)

    # Compare mitigation strategies
    compare_mitigation_strategies(timestamp, job_index, pauli_idx=3)

    print("\nAnalysis complete!")
