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
        'time_sampling': 751,
        'time_duration': 2 * np.pi * 5,
        'eta': 0.2
    }

    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': 0.1,
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

    # ========== Check Normalization and FDT ==========
    print("="*70)
    print("Checking Normalization and FDT")
    print("="*70)

    # Ensure thermal distributions are computed
    if not hasattr(evolution, 'thermal_dist'):
        evolution.get_thermal_occupation(system_parameters['temperature'])
        evolution.get_thermal_integral(system_parameters['temperature'])
        evolution.get_log_two_time(system_parameters['temperature'])
        evolution.get_thermal_sum(system_parameters['temperature'])

    # Check initial state
    print("\nChecking initial state...")
    gr_errors_init, gr_totals_init = initial_state_copy.check_gr_normalization(t1_idx=-1)
    gk_errors_init, gk_totals_init, gk_components_init = initial_state_copy.check_keldysh_normalization(
        t1_idx=-1,
        thermal_dist=evolution.thermal_dist,
        thermal_integral=evolution.thermal_integral,
        thermal_sum_left=evolution.thermal_sum_left,
        thermal_sum_right=evolution.thermal_sum_right,
        log_two_time=evolution.log_two_time,
        tmax=evolution.tmax
    )
    gk_fdt_init, gk_actual_init, fdt_error_init, fdt_max_error_init = initial_state_copy.check_fdt(
        f_thermal=evolution.thermal_dist,
        f_thermal_integral=evolution.thermal_integral,
        time_index=-1,
        thermal_sum_left=evolution.thermal_sum_left,
        thermal_sum_right=evolution.thermal_sum_right,
        log_two_time=evolution.log_two_time,
        tmax=evolution.tmax
    )

    print(f"  gr norm error (max):  {np.max(gr_errors_init):.2e}")
    print(f"  gk constraint error (max): {np.max(gk_errors_init):.2e}")
    print(f"  FDT error (max):      {fdt_max_error_init:.2e}")

    # Check final state
    print("\nChecking final state...")
    gr_errors_final, gr_totals_final = final_state.check_gr_normalization(t1_idx=-1)
    gk_errors_final, gk_totals_final, gk_components_final = final_state.check_keldysh_normalization(
        t1_idx=-1,
        thermal_dist=evolution.thermal_dist,
        thermal_integral=evolution.thermal_integral,
        thermal_sum_left=evolution.thermal_sum_left,
        thermal_sum_right=evolution.thermal_sum_right,
        log_two_time=evolution.log_two_time,
        tmax=evolution.tmax
    )
    gk_fdt_final, gk_actual_final, fdt_error_final, fdt_max_error_final = final_state.check_fdt(
        f_thermal=evolution.thermal_dist,
        f_thermal_integral=evolution.thermal_integral,
        time_index=-1,
        thermal_sum_left=evolution.thermal_sum_left,
        thermal_sum_right=evolution.thermal_sum_right,
        log_two_time=evolution.log_two_time,
        tmax=evolution.tmax
    )

    print(f"  gr norm error (max):  {np.max(gr_errors_final):.2e}")
    print(f"  gk constraint error (max): {np.max(gk_errors_final):.2e}")
    print(f"  FDT error (max):      {fdt_max_error_final:.2e}")

    # Plot normalization comparisons
    print("\nPlotting normalization checks...")
    pauli_names = [r'$\tau_0$ (I)', r'$\tau_1$ (X)', r'$\tau_2$ (Y)', r'$\tau_3$ (Z)']

    # gr normalization comparison
    fig_gr, axes_gr = plt.subplots(2, 2, figsize=(12, 10))
    fig_gr.suptitle('g^R Normalization: Initial vs Final', fontsize=14, fontweight='bold')

    for pauli_idx in range(4):
        ax = axes_gr.flat[pauli_idx]
        comp_init = gr_totals_init[pauli_idx, :]
        comp_final = gr_totals_final[pauli_idx, :]

        ax.plot(evolution.time_grid, np.abs(comp_init), 'b-', linewidth=2, label='Init', alpha=0.7, marker='o', markevery=50, markersize=4)
        ax.plot(evolution.time_grid, np.abs(comp_final), 'r-', linewidth=2, label='Final', alpha=0.7, marker='^', markevery=50, markersize=4)

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([evolution.time_grid[0], 0])
        max_err_init = np.abs(comp_init).max()
        max_err_final = np.abs(comp_final).max()
        ax.text(0.02, 0.98, f'Init: {max_err_init:.2e}\nFinal: {max_err_final:.2e}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    gr_norm_path = os.path.join(save_dir, f'gr_norm_comparison_{num_steps}steps.png')
    plt.savefig(gr_norm_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {gr_norm_path}")
    plt.close()

    # gk normalization comparison
    fig_gk, axes_gk = plt.subplots(2, 2, figsize=(12, 10))
    fig_gk.suptitle('g^K Keldysh Constraint: Initial vs Final', fontsize=14, fontweight='bold')

    for pauli_idx in range(4):
        ax = axes_gk.flat[pauli_idx]
        comp_init = gk_totals_init[pauli_idx, :]
        comp_final = gk_totals_final[pauli_idx, :]

        ax.plot(evolution.time_grid, np.abs(comp_init), 'b-', linewidth=2, label='Init', alpha=0.7, marker='o', markevery=50, markersize=4)
        ax.plot(evolution.time_grid, np.abs(comp_final), 'r-', linewidth=2, label='Final', alpha=0.7, marker='^', markevery=50, markersize=4)

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([evolution.time_grid[0], 0])

        max_err_init = np.abs(comp_init).max()
        max_err_final = np.abs(comp_final).max()
        ax.text(0.02, 0.98, f'Init: {max_err_init:.2e}\nFinal: {max_err_final:.2e}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    gk_norm_path = os.path.join(save_dir, f'gk_norm_comparison_{num_steps}steps.png')
    plt.savefig(gk_norm_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {gk_norm_path}")
    plt.close()

    # FDT comparison
    fig_fdt, axes_fdt = plt.subplots(2, 2, figsize=(12, 10))
    fig_fdt.suptitle('FDT Check: Initial vs Final (Zoomed to t\' ∈ [-10, 0.05])', fontsize=14, fontweight='bold')

    for pauli_idx in range(4):
        ax = axes_fdt.flat[pauli_idx]

        error_init_pauli = (fdt_error_init.trace(pauli_idx) / 2)[0, :]
        error_final_pauli = (fdt_error_final.trace(pauli_idx) / 2)[0, :]

        ax.scatter(evolution.time_grid, np.abs(error_init_pauli), c='b', s=10, label='Init Error', alpha=0.6)
        ax.scatter(evolution.time_grid, np.abs(error_final_pauli), c='r', s=10, label='Final Error', alpha=0.6, marker='^')

        ax.set_xlabel(r"$t'$", fontsize=10)
        ax.set_ylabel(f'{pauli_names[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_names[pauli_idx]}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-10, 0.05])  # Zoomed to see near-diagonal region

        max_err_init = np.abs(error_init_pauli).max()
        max_err_final = np.abs(error_final_pauli).max()
        max_idx_init = np.argmax(np.abs(error_init_pauli))
        max_idx_final = np.argmax(np.abs(error_final_pauli))

        # Calculate offset from diagonal (last element)
        offset_init = len(error_init_pauli) - 1 - max_idx_init
        offset_final = len(error_final_pauli) - 1 - max_idx_final

        ax.text(0.02, 0.98, f'Init max: {max_err_init:.2e}\n  at idx {max_idx_init} (offset {offset_init})\n'
                            f'Final max: {max_err_final:.2e}\n  at idx {max_idx_final} (offset {offset_final})',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fdt_comp_path = os.path.join(save_dir, f'fdt_comparison_{num_steps}steps.png')
    plt.savefig(fdt_comp_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fdt_comp_path}")
    plt.close()

    # gk2 component breakdown - Initial and Final
    print("\nPlotting gk2 component breakdowns...")
    pauli_idx = 2  # tau_2

    for state_name, gk_comps, time_grid in [
        ('Initial', gk_components_init, evolution.time_grid),
        ('Final', gk_components_final, evolution.time_grid)
    ]:
        fig_gk2, axes_gk2 = plt.subplots(2, 2, figsize=(14, 10))
        fig_gk2.suptitle(rf'g^K Constraint: $\tau_2$ Component Breakdown ({state_name} State)', fontsize=14, fontweight='bold')

        total = gk_comps['commutator'][pauli_idx, :] * 0  # Initialize as zero
        components_list = [
            ('commutator', r'$[\tau_3, g^K]$', 'b-'),
            ('gr_gk_conv_pure', r'$g^R \otimes g^K$', 'r-'),
            ('gk_ga_conv_pure', r'$g^K \otimes g^A$', 'g-'),
            ('thermal_gr', r'$g^R \otimes f$', 'm--'),
            ('thermal_ga', r'$f \otimes g^A$', 'c--'),
            ('thermal_gap_gr_conv', r'$g^R \otimes \Delta f$', 'orange'),
            ('thermal_gap_ga_conv', r'$\Delta f \otimes g^A$', 'brown'),
            ('thermal_gap_commutator', r'$[\tau_3, \Delta f]$', 'purple')
        ]

        # Calculate total
        for comp_key, _, _ in components_list:
            total += gk_comps[comp_key][pauli_idx, :]

        # Absolute values (linear scale)
        ax = axes_gk2[0, 0]
        ax.plot(time_grid, np.abs(total), 'k-', linewidth=2.5, label='Total', alpha=0.9)
        for comp_key, label, style in components_list:
            comp_data = gk_comps[comp_key][pauli_idx, :]
            linestyle = ':' if 'gap' in comp_key else ('-' if '--' not in style else '--')
            ax.plot(time_grid, np.abs(comp_data), style if style != 'orange' and style != 'brown' and style != 'purple' else linestyle,
                    color=style if style in ['orange', 'brown', 'purple'] else None,
                    linewidth=1.5, label=label, alpha=0.7)
        ax.set_xlabel(r'$t$', fontsize=11)
        ax.set_ylabel(r'|$\tau_2$|', fontsize=11)
        ax.set_title('Absolute Value (linear)', fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

        # Absolute values (log scale)
        ax = axes_gk2[0, 1]
        ax.semilogy(time_grid, np.abs(total) + 1e-20, 'k-', linewidth=2.5, label='Total', alpha=0.9)
        for comp_key, label, style in components_list:
            comp_data = gk_comps[comp_key][pauli_idx, :]
            linestyle = ':' if 'gap' in comp_key else ('-' if '--' not in style else '--')
            ax.semilogy(time_grid, np.abs(comp_data) + 1e-20, style if style != 'orange' and style != 'brown' and style != 'purple' else linestyle,
                        color=style if style in ['orange', 'brown', 'purple'] else None,
                        linewidth=1.5, label=label, alpha=0.7)
        ax.set_xlabel(r'$t$', fontsize=11)
        ax.set_ylabel(r'|$\tau_2$|', fontsize=11)
        ax.set_title('Absolute Value (log)', fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([time_grid[0], 0])

        # Key components comparison (absolute values)
        ax = axes_gk2[1, 0]
        key_components = [
            ('commutator', r'$[\tau_3, g^K]$', 'b-'),
            ('thermal_gap_gr_conv', r'$g^R \otimes \Delta f$', 'orange'),
            ('thermal_gap_ga_conv', r'$\Delta f \otimes g^A$', 'brown'),
        ]
        ax.plot(time_grid, np.abs(total), 'k-', linewidth=2.5, label='Total', alpha=0.9)
        for comp_key, label, style in key_components:
            comp_data = gk_comps[comp_key][pauli_idx, :]
            ax.plot(time_grid, np.abs(comp_data), style if style != 'orange' and style != 'brown' else '-',
                    color=style if style in ['orange', 'brown'] else None,
                    linewidth=1.5, label=label, alpha=0.7)
        ax.set_xlabel(r'$t$', fontsize=11)
        ax.set_ylabel(r'|$\tau_2$|', fontsize=11)
        ax.set_title('Key Components (linear)', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

        # Statistics
        ax = axes_gk2[1, 1]
        ax.axis('off')
        stats_text = rf"$\tau_2$ Component Stats ({state_name}):" + "\n\n"
        stats_text += f"Total max:          {np.abs(total).max():.2e}\n"
        for comp_key, label, _ in components_list:
            comp_data = gk_comps[comp_key][pauli_idx, :]
            # Strip LaTeX from label for stats
            clean_label = label.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')[:15]
            stats_text += f"{clean_label:20s} {np.abs(comp_data).max():.2e}\n"
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', family='monospace')

        plt.tight_layout()
        gk2_path = os.path.join(save_dir, f'gk2_breakdown_{state_name.lower()}_{num_steps}steps.png')
        plt.savefig(gk2_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {gk2_path}")
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
