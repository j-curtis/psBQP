"""
Run and Compare Test Script

Initialize equilibrium state, evolve for specified timesteps, and plot all g^R and g^K components.

Usage:
    python run_and_compare.py --timesteps 25
    python run_and_compare.py -n 10
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usadel_keldysh_evolution import UsadelKeldyshEvolution
from nambu_keldysh_class import NambuKeldyshTensor


def plot_nambu_components(nambu_tensor, time_grid, title="Nambu-Keldysh Tensor Components",
                          vmin=None, vmax=None, cmap='RdBu_r', save_path=None,
                          t_lim=None, tprime_lim=None):
    """
    Plot all 4 Pauli components of a NambuKeldyshTensor in a 4x2 grid.

    Args:
        nambu_tensor: NambuKeldyshTensor with shape (2, 2, Nt, Nt)
        time_grid: Array of time values
        title: Overall title for the figure
        vmin, vmax: Optional colorbar limits
        cmap: Colormap to use
        save_path: Optional path to save figure
        t_lim: Optional tuple (t_min, t_max) to zoom into specific t range
        tprime_lim: Optional tuple (tprime_min, tprime_max) to zoom into specific t' range
    """
    pauli_labels = ['I (Identity)', 'X (σ_x)', 'Y (σ_y)', 'Z (σ_z)']

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Time extent for imshow (origin='upper' puts smallest times in top-left)
    t_min, t_max = time_grid[0], time_grid[-1]
    extent = [t_min, t_max, t_max, t_min]

    for pauli_idx in range(4):
        # Extract Pauli component using trace
        component = nambu_tensor.trace(pauli_index=pauli_idx) / 2

        # Real part (top row)
        ax_real = axes[0, pauli_idx]
        im_real = ax_real.imshow(np.real(component), aspect='auto', origin='upper',
                                 extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_real.set_title(f"{pauli_labels[pauli_idx]} - Real", fontsize=11, fontweight='bold')
        ax_real.set_ylabel("t'", fontsize=10)
        ax_real.set_xlabel("t", fontsize=10)

        # Apply zoom limits if specified
        if t_lim is not None:
            ax_real.set_xlim(t_lim)
        if tprime_lim is not None:
            ax_real.set_ylim(tprime_lim[1], tprime_lim[0])  # Reversed for origin='upper'

        plt.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04)

        # Imaginary part (bottom row)
        ax_imag = axes[1, pauli_idx]
        im_imag = ax_imag.imshow(np.imag(component), aspect='auto', origin='upper',
                                 extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_imag.set_title(f"{pauli_labels[pauli_idx]} - Imag", fontsize=11, fontweight='bold')
        ax_imag.set_xlabel("t", fontsize=10)
        ax_imag.set_ylabel("t'", fontsize=10)

        # Apply zoom limits if specified
        if t_lim is not None:
            ax_imag.set_xlim(t_lim)
        if tprime_lim is not None:
            ax_imag.set_ylim(tprime_lim[1], tprime_lim[0])  # Reversed for origin='upper'

        plt.colorbar(im_imag, ax=ax_imag, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")

    plt.close()


def main(num_steps=25):
    print("="*70)
    print(f"Run and Compare Test: {num_steps} Timestep Evolution")
    print("="*70)
    print()

    # ========== Define Parameters ==========
    grid_parameters = {
        'time_sampling': 1001,
        'time_duration': 2 * np.pi * 5,
        'eta': 0.2
    }

    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': 0.5,
        'eta': 0.2
    }

    print("Grid parameters:")
    print(f"  Time points: {grid_parameters['time_sampling']}")
    print(f"  Time duration: {grid_parameters['time_duration']:.4f}")
    print(f"  Broadening η: {grid_parameters['eta']}")
    print()

    print("System parameters:")
    print(f"  Critical temperature: {system_parameters['critical_temperature']}")
    print(f"  Temperature: {system_parameters['temperature']}")
    print()

    # ========== Create Evolution Object ==========
    print("Creating evolution object...")
    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
    print(f"  Time grid: {evolution.ntpoints} points from {evolution.time_grid[0]:.2f} to {evolution.time_grid[-1]:.2f}")
    print(f"  dt: {evolution.delta_t:.6f}")
    print(f"  BCS coupling: {evolution._get_BCS_coupling():.4f}")
    print()

    # ========== Generate Initial Equilibrium State ==========
    print("Generating equilibrium initial state...")
    Q = 0.0  # No vector potential for equilibrium
    initial_state, gr_tau, gk_tau, J_eq = evolution.generate_initial_state(Q=Q)

    gap_history = initial_state.get_gap_history()
    gap_initial = gap_history[-1]

    print(f"  Initial state generated")
    print(f"  g^R shape: {initial_state.gr.data.shape}")
    print(f"  g^K shape: {initial_state.gk.data.shape}")
    print(f"  Initial gap: Δ = {np.real(gap_initial):.6f} + {np.imag(gap_initial):.6f}i")
    print(f"  |Δ| = {np.abs(gap_initial):.6f}")
    print(f"  Equilibrium current: J = {np.real(J_eq):.6e}")
    print()

    # ========== Save Initial State for Comparison ==========
    # Make deep copy IMMEDIATELY after generation (state will be modified in place during evolution!)
    import copy
    initial_state_copy = copy.deepcopy(initial_state)
    print("  Deep copy of initial state created for comparison")
    print()

    # ========== Plot Initial State ==========
    print("Plotting initial equilibrium state...")
    save_dir = 'Test_plots'
    os.makedirs(save_dir, exist_ok=True)

    # Zoom window for all plots
    zoom_window = (-1, 0)

    plot_nambu_components(
        initial_state_copy.gr,
        evolution.time_grid,
        title=f"Initial g^R (Equilibrium, T={system_parameters['temperature']}, Q={Q})",
        save_path=os.path.join(save_dir, 'gr_initial.png'),
        t_lim=None,
        tprime_lim=zoom_window
    )

    plot_nambu_components(
        initial_state_copy.gk,
        evolution.time_grid,
        title=f"Initial g^K (Equilibrium, T={system_parameters['temperature']}, Q={Q})",
        save_path=os.path.join(save_dir, 'gk_initial.png'),
        t_lim=zoom_window,
        tprime_lim=zoom_window
    )

    print()

    # ========== Perform Evolution ==========
    print("="*70)
    print(f"Evolving for {num_steps} timesteps...")
    print("="*70)

    driving_field = None  # No external driving

    result = evolution.real_time_evolution(
        initial_state,
        num_timesteps=num_steps,
        driving_field=driving_field,
        track_occupations=False
    )

    final_state = result['final_state']
    gaps = result['gaps']
    currents = result['currents']

    print(f"\n  Evolution complete!")
    print(f"  Final gap: Δ = {np.real(gaps[-1]):.6f} + {np.imag(gaps[-1]):.6f}i")
    print(f"  |Δ| = {np.abs(gaps[-1]):.6f}")
    print(f"  Gap change: ΔΔ = {np.abs(gaps[-1]) - np.abs(gap_initial):.6e}")
    print()

    # ========== Plot Gap Evolution ==========
    print("Plotting gap evolution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    timesteps = np.arange(num_steps)

    # Real and imaginary parts
    ax1.plot(timesteps, np.real(gaps), 'b-o', linewidth=2, markersize=6, label='Real')
    ax1.plot(timesteps, np.imag(gaps), 'r-o', linewidth=2, markersize=6, label='Imag')
    ax1.axhline(np.real(gap_initial), color='b', linestyle='--', alpha=0.5, label='Initial (Real)')
    ax1.axhline(np.imag(gap_initial), color='r', linestyle='--', alpha=0.5, label='Initial (Imag)')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Gap Δ(t)', fontsize=12)
    ax1.set_title('Gap Evolution: Real and Imaginary Parts', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Magnitude
    ax2.plot(timesteps, np.abs(gaps), 'g-o', linewidth=2, markersize=6, label='|Δ|')
    ax2.axhline(np.abs(gap_initial), color='g', linestyle='--', alpha=0.5, label='Initial |Δ|')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('|Δ(t)|', fontsize=12)
    ax2.set_title('Gap Evolution: Magnitude', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    gap_plot_path = os.path.join(save_dir, f'gap_evolution_{num_steps}steps.png')
    plt.savefig(gap_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {gap_plot_path}")
    plt.close()

    # ========== Plot Current Evolution ==========
    print("Plotting current evolution...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(timesteps, np.real(currents), 'b-o', linewidth=2, markersize=6, label='Real')
    ax.plot(timesteps, np.imag(currents), 'r-o', linewidth=2, markersize=6, label='Imag')
    ax.plot(timesteps, np.abs(currents), 'g-o', linewidth=2, markersize=6, label='|J|')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Current J(t)', fontsize=12)
    ax.set_title(f'Current Evolution ({num_steps} Timesteps)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-10)

    plt.tight_layout()
    current_plot_path = os.path.join(save_dir, f'current_evolution_{num_steps}steps.png')
    plt.savefig(current_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {current_plot_path}")
    plt.close()

    print()

    # ========== Plot Final State ==========
    print(f"Plotting final state after {num_steps} timesteps...")

    plot_nambu_components(
        final_state.gr,
        evolution.time_grid,
        title=f"Final g^R (After {num_steps} Steps, T={system_parameters['temperature']})",
        save_path=os.path.join(save_dir, 'gr_final.png'),
        t_lim=zoom_window,
        tprime_lim=zoom_window
    )

    plot_nambu_components(
        final_state.gk,
        evolution.time_grid,
        title=f"Final g^K (After {num_steps} Steps, T={system_parameters['temperature']})",
        save_path=os.path.join(save_dir, 'gk_final.png'),
        t_lim=zoom_window,
        tprime_lim=zoom_window
    )

    print()

    # ========== Plot Difference (Final - Initial) ==========
    print("Plotting difference (Final - Initial)...")

    # Compute differences (use initial_state_copy, not initial_state!)
    gr_diff_data = final_state.gr.data - initial_state_copy.gr.data
    gk_diff_data = final_state.gk.data - initial_state_copy.gk.data

    gr_diff = NambuKeldyshTensor(gr_diff_data)
    gk_diff = NambuKeldyshTensor(gk_diff_data)

    plot_nambu_components(
        gr_diff,
        evolution.time_grid,
        title=f"Δg^R = g^R(final) - g^R(initial)",
        save_path=os.path.join(save_dir, 'gr_difference.png'),
        t_lim=zoom_window,
        tprime_lim=zoom_window
    )

    plot_nambu_components(
        gk_diff,
        evolution.time_grid,
        title=f"Δg^K = g^K(final) - g^K(initial)",
        save_path=os.path.join(save_dir, 'gk_difference.png'),
        t_lim=zoom_window,
        tprime_lim=zoom_window
    )

    print()

    # ========== Summary Statistics ==========
    print("="*70)
    print("Summary Statistics")
    print("="*70)

    print("\nGap evolution:")
    print(f"  Initial:  Δ = {np.abs(gap_initial):.6f}")
    print(f"  Final:    Δ = {np.abs(gaps[-1]):.6f}")
    print(f"  Change:   ΔΔ = {np.abs(gaps[-1]) - np.abs(gap_initial):.6e}")
    print(f"  Relative: ΔΔ/Δ = {(np.abs(gaps[-1]) - np.abs(gap_initial))/np.abs(gap_initial)*100:.3e}%")

    print("\nGreen's function changes:")
    gr_max_diff = np.max(np.abs(gr_diff_data))
    gk_max_diff = np.max(np.abs(gk_diff_data))
    gr_initial_max = np.max(np.abs(initial_state_copy.gr.data))
    gk_initial_max = np.max(np.abs(initial_state_copy.gk.data))

    print(f"  Max |Δg^R|: {gr_max_diff:.6e}")
    print(f"  Max |g^R_initial|: {gr_initial_max:.6f}")
    print(f"  Relative: {gr_max_diff/gr_initial_max*100:.3e}%")

    print(f"  Max |Δg^K|: {gk_max_diff:.6e}")
    print(f"  Max |g^K_initial|: {gk_initial_max:.6f}")
    print(f"  Relative: {gk_max_diff/gk_initial_max*100:.3e}%")

    print("\nCurrent evolution:")
    print(f"  Initial:  J = {np.abs(J_eq):.6e}")
    print(f"  Final:    J = {np.abs(currents[-1]):.6e}")
    print(f"  Max |J|:  {np.max(np.abs(currents)):.6e}")
    print(f"  Mean |J|: {np.mean(np.abs(currents)):.6e}")

    print()
    print("="*70)
    print("Test complete! All plots saved to Test_plots/")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run Usadel-Keldysh evolution and compare initial/final states',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_and_compare.py --timesteps 25
  python run_and_compare.py -n 10
  python run_and_compare.py              # Uses default: 25 timesteps
        """
    )

    parser.add_argument(
        '-n', '--timesteps',
        type=int,
        default=25,
        help='Number of timesteps to evolve (default: 25)'
    )

    args = parser.parse_args()

    # Validate timesteps
    if args.timesteps < 1:
        parser.error("Number of timesteps must be at least 1")

    main(num_steps=args.timesteps)
