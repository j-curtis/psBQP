"""
dt Scaling Study

Evolve the equilibrium state for the same physical time on several time grids and compare
the gap drift and the FDT / constraint / normalization errors as a function of dt.

Usage:
    python dt_scaling_study.py                 # evolve for t = 1.0 on N_t = 501, 751, 1001
    python dt_scaling_study.py --time 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
import copy
import argparse
import contextlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usadel_keldysh_evolution import UsadelKeldyshEvolution


PAULI_NAMES = [r'$\tau_0$', r'$\tau_1$', r'$\tau_2$', r'$\tau_3$']


def max_errors(state, evolution):
    """Maximum |error| per Pauli channel of the FDT, g^K constraint and g^R normalization checks (last row)."""
    _, gr_totals = state.check_gr_normalization(t1_idx=-1)
    _, gk_totals, _ = state.check_keldysh_normalization(
        -1, evolution.thermal_dist, evolution.thermal_integral,
        thermal_sum_left=evolution.thermal_sum_left, thermal_sum_right=evolution.thermal_sum_right,
        log_two_time=evolution.log_two_time)
    _, _, fdt_error_row, _ = state.check_fdt(
        evolution.thermal_dist, evolution.thermal_integral, time_index=-1,
        thermal_sum_left=evolution.thermal_sum_left, thermal_sum_right=evolution.thermal_sum_right,
        log_two_time=evolution.log_two_time, tmax=evolution.tmax)

    fdt_totals = np.array([(fdt_error_row.trace(p) / 2)[0, :] for p in range(4)])
    return {'fdt': np.abs(fdt_totals).max(axis=1),
            'gk_constraint': np.abs(gk_totals).max(axis=1),
            'gr_norm': np.abs(gr_totals).max(axis=1)}


def run_single(N_t, evolution_time, system_parameters):
    """Evolve the equilibrium state on a grid with N_t points for (approximately) evolution_time."""
    grid_parameters = {'time_sampling': N_t, 'time_duration': 2 * np.pi * 5, 'eta': system_parameters['eta']}

    with contextlib.redirect_stdout(io.StringIO()):
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
        initial_state, _, _, _ = evolution.generate_initial_state(Q=0.0)

    dt = evolution.delta_t
    num_steps = int(round(evolution_time / dt))
    initial_state_copy = copy.deepcopy(initial_state)
    gap_eq = initial_state.get_gap_history()[-1]

    print(f"N_t = {N_t}: dt = {dt:.5f}, {num_steps} steps, Delta_eq = {np.real(gap_eq):.6f}")
    result = evolution.real_time_evolution(initial_state, num_timesteps=num_steps, driving_field=None, track_occupations=False)

    return {'dt': dt,
            'times': dt * np.arange(1, num_steps + 1),
            'gap_eq': gap_eq,
            'gaps': result['gaps'],
            'errors_initial': max_errors(initial_state_copy, evolution),
            'errors_final': max_errors(result['final_state'], evolution)}


def main(N_t_values, evolution_time):
    system_parameters = {'critical_temperature': 1.0, 'temperature': 0.1, 'eta': 0.2}
    save_dir = 'Test_plots'
    os.makedirs(save_dir, exist_ok=True)

    results = {N_t: run_single(N_t, evolution_time, system_parameters) for N_t in N_t_values}
    dts = np.array([results[N_t]['dt'] for N_t in N_t_values])

    # ========== Summary table ==========
    print("\n" + "=" * 90)
    print(f"Results after t = {evolution_time}")
    print("=" * 90)
    for N_t in N_t_values:
        res = results[N_t]
        drift = res['gaps'][-1] - res['gap_eq']
        print(f"N_t = {N_t:5d}  dt = {res['dt']:.5f}  final Delta - Delta_eq = {np.real(drift):+.3e}")
        for key in ['fdt', 'gk_constraint', 'gr_norm']:
            init = "  ".join(f"{v:.2e}" for v in res['errors_initial'][key])
            final = "  ".join(f"{v:.2e}" for v in res['errors_final'][key])
            print(f"    {key:14s} tau0..3  init: {init}   final: {final}")

    # ========== Gap evolution vs time ==========
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for N_t in N_t_values:
        res = results[N_t]
        ax.plot(res['times'], np.real(res['gaps'] - res['gap_eq']), '-o', markersize=3, label=f"N_t = {N_t}, dt = {res['dt']:.4f}")
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel(r'Re $\Delta(t) - \Delta_{eq}$', fontsize=12)
    ax.set_title('Gap drift vs time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'dt_scaling_gap_vs_time_T{evolution_time}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

    # ========== Gap drift vs dt ==========
    drifts = np.array([np.abs(results[N_t]['gaps'][-1] - results[N_t]['gap_eq']) for N_t in N_t_values])
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(dts, drifts, 'ko-', linewidth=2, markersize=7, label=r'$|\Delta(t_{end}) - \Delta_{eq}|$')
    ax.loglog(dts, drifts[0] * dts / dts[0], 'b--', alpha=0.6, label=r'$\propto dt$')
    ax.loglog(dts, drifts[0] * (dts / dts[0])**2, 'r--', alpha=0.6, label=r'$\propto dt^2$')
    ax.set_xlabel('dt', fontsize=12)
    ax.set_ylabel('gap drift', fontsize=12)
    ax.set_title(f'Gap drift at t = {evolution_time} vs dt', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(save_dir, f'dt_scaling_gap_vs_dt_T{evolution_time}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ========== Errors vs dt ==========
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, key, title in zip(axes, ['fdt', 'gk_constraint', 'gr_norm'], ['FDT', r'$g^K$ constraint', r'$g^R$ normalization']):
        for pauli_idx in range(4):
            err_init = np.array([results[N_t]['errors_initial'][key][pauli_idx] for N_t in N_t_values])
            err_final = np.array([results[N_t]['errors_final'][key][pauli_idx] for N_t in N_t_values])
            line, = ax.loglog(dts, err_final, 'o-', linewidth=2, label=f'{PAULI_NAMES[pauli_idx]} final')
            ax.loglog(dts, err_init, 's--', color=line.get_color(), alpha=0.5, label=f'{PAULI_NAMES[pauli_idx]} init')
        ax.loglog(dts, 1e-2 * dts / dts[0], 'k:', alpha=0.5, label=r'$\propto dt$')
        ax.loglog(dts, 1e-2 * (dts / dts[0])**2, 'k-.', alpha=0.5, label=r'$\propto dt^2$')
        ax.set_xlabel('dt', fontsize=12)
        ax.set_ylabel('max |error|', fontsize=12)
        ax.set_title(f'{title} (t = {evolution_time})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(save_dir, f'dt_scaling_errors_vs_dt_T{evolution_time}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gap drift and error scaling with dt at fixed physical evolution time')
    parser.add_argument('-t', '--time', type=float, default=1.0, help='Physical evolution time (default: 1.0)')
    parser.add_argument('-N', '--ntpoints', type=int, nargs='+', default=[501, 751, 1001], help='Time grid sizes (default: 501 751 1001)')
    args = parser.parse_args()

    main(args.ntpoints, args.time)
