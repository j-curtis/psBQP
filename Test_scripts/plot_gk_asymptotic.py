"""
Plot equilibrium g^K in frequency domain with and without asymptotic_added tail.

Shows:
1. g^K(ω) as computed by equilibrium solver (before line 311 addition)
2. g^K(ω) + asymptotic_added (what actually gets FFT'd)
3. The asymptotic_added itself: 2iΔ/ω · tanh(ω/2T)

This helps understand what's causing the 1/ω decay at large frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from equilibrium_class import EquilibriumSolver
from usadel_keldysh_evolution import UsadelKeldyshEvolution


def plot_gk_with_asymptotic_added(T_max=2*np.pi*5, N_t=101, temperature=0.5,
                             save_path=None, show_plot=True):
    """
    Plot equilibrium g^K(ω) with and without asymptotic_added regularization.

    Args:
        T_max: Time duration (same as time_duration in grid_parameters)
        N_t: Number of time points (same as time_sampling)
        temperature: System temperature
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """

    print("="*70)
    print("PLOTTING EQUILIBRIUM g^K WITH/WITHOUT ASYMPTOTIC")
    print("="*70)

    # Set up parameters matching run_and_compare style
    grid_parameters = {
        'time_sampling': N_t,
        'time_duration': T_max,
        'eta': 0.1
    }

    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': temperature,
        'eta': 0.1
    }

    optimization_parameters = {
        'tol': 1e-6,
        'maxiter': 1000,
        'method': 'anderson'
    }

    sigma_scatterings = {}

    print(f"\nParameters:")
    print(f"  T_max = {T_max:.4f}")
    print(f"  N_t = {N_t}")
    print(f"  Temperature = {temperature}")

    # Create evolution object to get omega grid
    evolution = UsadelKeldyshEvolution(
        grid_parameters=grid_parameters,
        system_parameters=system_parameters
    )

    # Add omega grid to grid_parameters for equilibrium solver
    grid_params_with_omega = grid_parameters.copy()
    grid_params_with_omega['omega_grid'] = evolution.omega_grid
    grid_params_with_omega['energy_cutoff'] = evolution.energy_cutoff

    # Create equilibrium solver
    equilibrium_solver = EquilibriumSolver(
        grid_params_with_omega,
        system_parameters,
        optimization_parameters,
        sigma_scatterings
    )

    print(f"\nOmega grid:")
    print(f"  n_omega = {len(evolution.omega_grid)}")
    print(f"  omega_max = {np.max(np.abs(evolution.omega_grid)):.3f}")
    print(f"  d_omega = {evolution.d_omega:.6e}")

    # Compute equilibrium Green's functions
    print(f"\nComputing equilibrium state...")
    gr_eq, gk_eq = equilibrium_solver.compute_equilibrium_gr(
        temperature=temperature,
        Q=0.0,
        gr0=None,
        compute_gk=True
    )

    gap = equilibrium_solver.gap_0
    print(f"  Equilibrium gap: Δ = {gap:.6f}")

    # Extract omega grid
    omega_grid = equilibrium_solver.usadel_solver.w_arr
    n_omega = len(omega_grid)

    # Extract g^K tau_2 component (Y matrix)
    # This is AFTER line 311 addition in equilibrium_class.py
    gk_tau2_after_line311 = np.array(gk_eq._trace(2)) / 2

    # Compute the asymptotic_added that line 311 ADDED
    # Line 311 adds: 2iΔ/ω · tanh(ω/2T) (actually with a minus sign)
    tanh_omega = np.tanh(omega_grid / (2.0 * temperature))
    asymptotic_added = -2.0j * gap / (omega_grid + 1e-6) * tanh_omega

    # Fix value at ω=0 using Taylor expansion
    zero_idx = np.argmin(np.abs(omega_grid))
    asymptotic_added = np.array(asymptotic_added, dtype=complex)
    asymptotic_added[zero_idx] = -1j * gap / temperature

    # Equilibrium solver output (before line 311) = what line 311 receives
    # This ALREADY has natural asymptotic_addeds from the physics
    gk_tau2_before_line311 = gk_tau2_after_line311 - asymptotic_added

    # To get regularized g^K (fast decay), we'd need to SUBTRACT the natural asymptotic_added
    # But line 311 does the OPPOSITE - it ADDS more
    gk_tau2_regularized = gk_tau2_before_line311 - asymptotic_added

    print(f"\nAt boundary (ω ≈ {omega_grid[0]:.2f}):")
    print(f"  After line 311 (what gets FFT'd):     {gk_tau2_after_line311[0]:.6e}")
    print(f"  Asymptotic added by line 311:         {asymptotic_added[0]:.6e}")
    print(f"  Before line 311 (solver output):      {gk_tau2_before_line311[0]:.6e}")
    print(f"  If we subtracted (regularized):       {gk_tau2_regularized[0]:.6e}")

    print(f"\nAt ω ≈ 0:")
    print(f"  After line 311:   {gk_tau2_after_line311[zero_idx]:.6e}")
    print(f"  Asymptotic added: {asymptotic_added[zero_idx]:.6e}")
    print(f"  Before line 311:  {gk_tau2_before_line311[zero_idx]:.6e}")
    print(f"  Regularized:      {gk_tau2_regularized[zero_idx]:.6e}")

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Real part - After line 311
    ax = axes[0, 0]
    ax.plot(omega_grid, np.real(gk_tau2_after_line311), 'b-', linewidth=2, label='After line 311')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'Re[$g^K_2(\omega)$]', fontsize=12)
    ax.set_title(r'Real Part: After Line 311 (actual FFT input)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 2: Imaginary part - After line 311
    ax = axes[1, 0]
    ax.plot(omega_grid, np.imag(gk_tau2_after_line311), 'b-', linewidth=2, label='After line 311')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'Im[$g^K_2(\omega)$]', fontsize=12)
    ax.set_title(r'Imaginary Part: After Line 311', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 3: Real part of asymptotic_added added
    ax = axes[0, 1]
    ax.plot(omega_grid, np.real(asymptotic_added), 'r-', linewidth=2, label='Added by line 311')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'Re[Asymptotic]', fontsize=12)
    ax.set_title(r'Real Part: Asymptotic Added by Line 311', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 4: Imaginary part of asymptotic_added added
    ax = axes[1, 1]
    ax.plot(omega_grid, np.imag(asymptotic_added), 'r-', linewidth=2, label='Added by line 311')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'Im[Asymptotic]', fontsize=12)
    ax.set_title(r'Imaginary Part: Asymptotic Added', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 5: Real part - Before line 311 (solver output)
    ax = axes[0, 2]
    ax.plot(omega_grid, np.real(gk_tau2_before_line311), 'g-', linewidth=2, label='Before line 311')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'Re[$g^K_2(\omega)$]', fontsize=12)
    ax.set_title(r'Real Part: Before Line 311 (solver output)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 6: Imaginary part - Before line 311
    ax = axes[1, 2]
    ax.plot(omega_grid, np.imag(gk_tau2_before_line311), 'g-', linewidth=2, label='Before line 311')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'Im[$g^K_2(\omega)$]', fontsize=12)
    ax.set_title(r'Imaginary Part: Before Line 311', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    fig.suptitle(rf'Equilibrium $g^K_2(\omega)$ with/without Asymptotic (T={temperature}, $\Delta$={gap:.4f})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Create second figure: decay analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: |g^K| comparison
    ax = axes2[0, 0]
    ax.semilogy(omega_grid, np.abs(gk_tau2_after_line311) + 1e-10, 'b-', linewidth=2, label='With asymptotic_added', alpha=0.7)
    ax.semilogy(omega_grid, np.abs(asymptotic_added) + 1e-10, 'r--', linewidth=2, label='Asymptotic only', alpha=0.7)
    ax.semilogy(omega_grid, np.abs(gk_tau2_before_line311) + 1e-10, 'g-', linewidth=2, label='Without asymptotic_added', alpha=0.7)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'$|g^K_2(\omega)|$', fontsize=12)
    ax.set_title(r'Magnitude Comparison (log scale)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)

    # Plot 2: ω · g^K (test for 1/ω decay)
    ax = axes2[0, 1]
    omega_gk_with = omega_grid * gk_tau2_after_line311
    omega_gk_no = omega_grid * gk_tau2_before_line311
    omega_asymp = omega_grid * asymptotic_added

    ax.plot(omega_grid, np.abs(omega_gk_with), 'b-', linewidth=2, label=r'$\omega \cdot g^K$ (with asymp)', alpha=0.7)
    ax.plot(omega_grid, np.abs(omega_asymp), 'r--', linewidth=2, label=r'$\omega \cdot$ asymptotic_added', alpha=0.7)
    ax.plot(omega_grid, np.abs(omega_gk_no), 'g-', linewidth=2, label=r'$\omega \cdot g^K$ (no asymp)', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'$|\omega \cdot g^K_2(\omega)|$', fontsize=12)
    ax.set_title(r'Decay Test: $\omega \cdot g^K$ (constant → 1/$\omega$ decay)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 3: ω² · g^K (test for 1/ω² decay)
    ax = axes2[1, 0]
    omega2_gk_with = omega_grid**2 * gk_tau2_after_line311
    omega2_gk_no = omega_grid**2 * gk_tau2_before_line311
    omega2_asymp = omega_grid**2 * asymptotic_added

    ax.plot(omega_grid, np.abs(omega2_gk_with), 'b-', linewidth=2, label=r'$\omega^2 \cdot g^K$ (with asymp)', alpha=0.7)
    ax.plot(omega_grid, np.abs(omega2_asymp), 'r--', linewidth=2, label=r'$\omega^2 \cdot$ asymptotic_added', alpha=0.7)
    ax.plot(omega_grid, np.abs(omega2_gk_no), 'g-', linewidth=2, label=r'$\omega^2 \cdot g^K$ (no asymp)', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$\omega$', fontsize=12)
    ax.set_ylabel(r'$|\omega^2 \cdot g^K_2(\omega)|$', fontsize=12)
    ax.set_title(r'Decay Test: $\omega^2 \cdot g^K$ (constant → 1/$\omega^2$ decay)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 4: Boundary values
    ax = axes2[1, 1]
    ax.axis('off')

    # Get boundary indices
    n_boundary = 10
    omega_left = omega_grid[:n_boundary]
    omega_right = omega_grid[-n_boundary:]

    stats_text = "Decay Analysis\n"
    stats_text += "="*50 + "\n\n"
    stats_text += f"Gap Δ = {gap:.6f}\n"
    stats_text += f"Temperature T = {temperature}\n\n"
    stats_text += f"At large |ω| (|ω| ≈ {np.abs(omega_grid[0]):.2f}):\n\n"

    stats_text += "With asymptotic_added:\n"
    stats_text += f"  g^K ≈ {np.mean(np.abs(gk_tau2_after_line311[:5])):.4e}\n"
    stats_text += f"  ω·g^K ≈ {np.mean(np.abs(omega_gk_with[:5])):.4e}\n"
    stats_text += f"  ω²·g^K ≈ {np.mean(np.abs(omega2_gk_with[:5])):.4e}\n\n"

    stats_text += "Without asymptotic_added:\n"
    stats_text += f"  g^K ≈ {np.mean(np.abs(gk_tau2_before_line311[:5])):.4e}\n"
    stats_text += f"  ω·g^K ≈ {np.mean(np.abs(omega_gk_no[:5])):.4e}\n"
    stats_text += f"  ω²·g^K ≈ {np.mean(np.abs(omega2_gk_no[:5])):.4e}\n\n"

    stats_text += "Asymptotic 2iΔ/ω·tanh:\n"
    stats_text += f"  Value ≈ {np.mean(np.abs(asymptotic_added[:5])):.4e}\n"
    stats_text += f"  ω·asymp ≈ {np.mean(np.abs(omega_asymp[:5])):.4e}\n"
    stats_text += f"  ω²·asymp ≈ {np.mean(np.abs(omega2_asymp[:5])):.4e}\n\n"

    # Interpretation
    omega_gk_boundary = np.mean(np.abs(omega_gk_with[:5]))
    omega2_gk_boundary = np.mean(np.abs(omega2_gk_with[:5]))

    stats_text += "\nInterpretation:\n"
    if omega_gk_boundary > 1.0:
        stats_text += "  ω·g^K → constant ≠ 0\n"
        stats_text += "  → Decay is 1/ω (SLOW!)\n"
        stats_text += "  → Missing tail significant\n"
    elif omega2_gk_boundary > 1.0:
        stats_text += "  ω²·g^K → constant ≠ 0\n"
        stats_text += "  → Decay is 1/ω² (moderate)\n"
        stats_text += "  → Missing tail ~3%\n"
    else:
        stats_text += "  ω²·g^K → 0\n"
        stats_text += "  → Decay faster than 1/ω²\n"
        stats_text += "  → Missing tail negligible\n"

    ax.text(0.05, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig2.suptitle(rf'Decay Analysis: $g^K_2(\omega)$ Asymptotic Behavior (T={temperature})',
                  fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        decay_path = save_path.replace('.png', '_decay.png')
        plt.savefig(decay_path, dpi=150, bbox_inches='tight')
        print(f"Decay analysis saved to: {decay_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nThe equilibrium solver output (before line 311) has value {gk_tau2_before_line311[0]:.3e} at boundary.")
    print(f"Line 311 ADDS asymptotic {asymptotic_added[0]:.3e}, creating {gk_tau2_after_line311[0]:.3e}.")
    print(f"\nBefore line 311: Small boundary value → potentially fast decay")
    print(f"After line 311:  Large boundary value → 1/ω decay → FFT errors")
    print(f"\nThe ~{np.abs(gk_tau2_before_line311[0]):.3e} solver output suggests it may already")
    print(f"contain natural asymptotics, and line 311 adds MORE on top.")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot equilibrium g^K with/without asymptotic_added tail',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--tmax', type=float, default=2*np.pi*5,
                        help='Time duration (default: 2π*5)')
    parser.add_argument('--nt', type=int, default=101,
                        help='Number of time points (default: 101)')
    parser.add_argument('--temperature', '-T', type=float, default=0.5,
                        help='System temperature (default: 0.5)')
    parser.add_argument('--save', type=str, default='Test_plots/gk_asymptotic_added.png',
                        help='Path to save figure (default: Test_plots/gk_asymptotic_added.png)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (only save)')

    args = parser.parse_args()

    # Create output directory if needed
    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot_gk_with_asymptotic_added(
        T_max=args.tmax,
        N_t=args.nt,
        temperature=args.temperature,
        save_path=args.save,
        show_plot=not args.no_show
    )
