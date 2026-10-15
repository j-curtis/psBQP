"""
dt Scaling Study

Evolve the equilibrium state for the same physical time on several time grids and compare
the gap drift and the FDT / constraint / normalization errors as a function of dt.

With --Q a constant vector potential A = Q is applied and the initial state is the finite-Q equilibrium
(self-energy 'em' on); --baseline also runs Q = 0 and reports the excess drift caused by the field.

Usage:
    python dt_scaling_study.py                 # evolve for t = 1.0 on N_t = 501, 751, 1001
    python dt_scaling_study.py --time 2.0
    python dt_scaling_study.py --Q 0.2 --temperature 0.5 --baseline -N 201 301 401 -t 3.0
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
    #~ _, gk_totals, _ = state.check_keldysh_normalization(
    #~     -1, evolution.thermal_dist, evolution.thermal_integral,
    #~     thermal_sum_left=evolution.thermal_sum_left, thermal_sum_right=evolution.thermal_sum_right,
    #~     log_two_time=evolution.log_two_time)
    #~ _, _, fdt_error_row, _ = state.check_fdt(
    #~     evolution.thermal_dist, evolution.thermal_integral, time_index=-1,
    #~     thermal_sum_left=evolution.thermal_sum_left, thermal_sum_right=evolution.thermal_sum_right,
    #~     log_two_time=evolution.log_two_time, tmax=evolution.tmax)
    _, gk_totals, _ = state.check_keldysh_normalization(-1, evolution.thermal)
    _, _, fdt_error_row, _ = state.check_fdt(evolution.thermal, time_index=-1)

    fdt_totals = np.array([(fdt_error_row.trace(p) / 2)[0, :] for p in range(4)])
    return {'fdt': np.abs(fdt_totals).max(axis=1),
            'gk_constraint': np.abs(gk_totals).max(axis=1),
            'gr_norm': np.abs(gr_totals).max(axis=1)}


def error_profiles(state, evolution):
    """Norm over the Nambu-Keldysh components of the FDT, g^K constraint and g^R normalization violation of the last row t = t_last, for every t'."""
    gr_errors, _ = state.check_gr_normalization(t1_idx=-1)
    gk_errors, _, _ = state.check_keldysh_normalization(-1, evolution.thermal)
    _, _, fdt_error_row, _ = state.check_fdt(evolution.thermal, time_index=-1)
    fdt_errors = np.sqrt(np.sum(np.abs(fdt_error_row.data)**2, axis=(0, 1)))[0, :]
    return {'fdt': fdt_errors, 'gk_constraint': np.asarray(gk_errors), 'gr_norm': np.asarray(gr_errors)}


def run_single(N_t, evolution_time, system_parameters, Q=0.0):
    """Evolve the equilibrium state (at vector potential / phase gradient Q) on a grid with N_t points for (approximately) evolution_time."""
    grid_parameters = {'time_sampling': N_t, 'time_duration': 2 * np.pi * 5, 'eta': system_parameters['eta']}
    self_energy_dictionary = {'em': 1.0} if Q != 0.0 else {}

    with contextlib.redirect_stdout(io.StringIO()):
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters, self_energy_dictionary=self_energy_dictionary)
        #~ initial_state, _, _, _ = evolution.generate_initial_state(Q=0.0)
        initial_state, _, _, _ = evolution.generate_initial_state(Q=Q)

    dt = evolution.delta_t
    num_steps = int(round(evolution_time / dt))
    initial_state_copy = copy.deepcopy(initial_state)
    gap_eq = initial_state.get_gap_history()[-1]

    print(f"N_t = {N_t}: Q = {Q}: dt = {dt:.5f}, {num_steps} steps, Delta_eq = {np.real(gap_eq):.6f}", flush=True)
    #* constant vector potential A = Q over the whole run (the history is filled with driving_field[0])
    driving_field = np.full(num_steps, Q) if Q != 0.0 else None
    #~ result = evolution.real_time_evolution(initial_state, num_timesteps=num_steps, driving_field=None, track_occupations=False)
    result = evolution.real_time_evolution(initial_state, num_timesteps=num_steps, driving_field=driving_field, track_occupations=False)

    times = dt * np.arange(1, num_steps + 1)

    #* gap(t) plot of this run, written right after the run (before the diagnostics), so a later failure or an interrupted study loses nothing
    os.makedirs('Test_plots', exist_ok=True)
    run_plot_path = os.path.join('Test_plots', f'dt_scaling_gap_vs_time_run_N{N_t}_Q{Q}_T{evolution_time}_temp{system_parameters["temperature"]}_eta{system_parameters["eta"]}.png')
    fig_run, ax_run = plt.subplots(1, 1, figsize=(8, 5))
    ax_run.plot(np.concatenate([[0.0], times]), np.real(np.concatenate([[gap_eq], np.asarray(result['gaps'])])), '-o', markersize=3)
    ax_run.set_xlabel('t', fontsize=12)
    ax_run.set_ylabel(r'Re $\Delta(t)$', fontsize=12)
    ax_run.set_title(f'Gap(t): N_t = {N_t}, dt = {dt:.4f}, Q = {Q}, T = {system_parameters["temperature"]}, $\\eta$ = {system_parameters["eta"]}', fontsize=12, fontweight='bold')
    ax_run.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_run)
    print(f"    saved gap(t) plot: {run_plot_path}", flush=True)

    def safe_max_errors(state, label):
        try:
            return max_errors(state, evolution)
        except Exception as error:
            print(f"    WARNING: {label} diagnostics failed ({type(error).__name__}: {error}); errors set to NaN", flush=True)
            return {key: np.full(4, np.nan) for key in ('fdt', 'gk_constraint', 'gr_norm')}

    return {'dt': dt,
            'times': times,
            'gap_eq': gap_eq,
            'gaps': result['gaps'],
            #~ 'errors_initial': max_errors(initial_state_copy, evolution),
            #~ 'errors_final': max_errors(result['final_state'], evolution)}
            'errors_initial': safe_max_errors(initial_state_copy, 'initial'),
            'errors_final': safe_max_errors(result['final_state'], 'final'),
            'profiles_initial': error_profiles(initial_state_copy, evolution),
            'profiles_final': error_profiles(result['final_state'], evolution)}


def main(N_t_values, evolution_time, Q=0.0, temperature=0.1, eta=0.2, baseline=False):
    system_parameters = {'critical_temperature': 1.0, 'temperature': temperature, 'eta': eta}
    save_dir = 'Test_plots'
    os.makedirs(save_dir, exist_ok=True)
    tag = f'T{evolution_time}_Q{Q}'

    results = {N_t: run_single(N_t, evolution_time, system_parameters, Q) for N_t in N_t_values}
    dts = np.array([results[N_t]['dt'] for N_t in N_t_values])

    #* Q = 0 run on the same grids: the field-induced part of the drift is the difference of the relative drifts
    results_baseline = {N_t: run_single(N_t, evolution_time, system_parameters, 0.0) for N_t in N_t_values} if (baseline and Q != 0.0) else None

    # ========== Summary table ==========
    print("\n" + "=" * 90)
    print(f"Results after t = {evolution_time}")
    print("=" * 90)
    for N_t in N_t_values:
        res = results[N_t]
        drift = res['gaps'][-1] - res['gap_eq']
        relative = np.real(drift) / np.real(res['gap_eq'])
        line = f"N_t = {N_t:5d}  dt = {res['dt']:.5f}  final Delta - Delta_eq = {np.real(drift):+.3e}  relative = {relative:+.3e}"
        if results_baseline is not None:
            base = results_baseline[N_t]
            base_relative = np.real(base['gaps'][-1] - base['gap_eq']) / np.real(base['gap_eq'])
            line += f"  Q=0 relative = {base_relative:+.3e}  excess (field) = {relative - base_relative:+.3e}"
        print(line)
        for key in ['fdt', 'gk_constraint', 'gr_norm']:
            init = "  ".join(f"{v:.2e}" for v in res['errors_initial'][key])
            final = "  ".join(f"{v:.2e}" for v in res['errors_final'][key])
            print(f"    {key:14s} tau0..3  init: {init}   final: {final}")

    # ========== Error profiles of the last row vs t' (one figure per N_t) ==========
    profile_titles = {'fdt': 'FDT', 'gk_constraint': r'$g^K$ constraint', 'gr_norm': r'$g^R$ normalization'}
    for N_t in N_t_values:
        res = results[N_t]
        t_prime = (np.arange(N_t) - (N_t - 1)) * res['dt']
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        for ax, key in zip(axes, profile_titles):
            ax.semilogy(t_prime, res['profiles_initial'][key], '--', label='initial state')
            ax.semilogy(t_prime, res['profiles_final'][key], '-', label=f't = {evolution_time}')
            ax.set_xlabel(r"$t' - t_{last}$", fontsize=12)
            ax.set_ylabel(r'$\|$error$\|$ at $t = t_{last}$', fontsize=12)
            ax.set_title(f'{profile_titles[key]}: N_t = {N_t}, dt = {res["dt"]:.4f}, Q = {Q}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        path = os.path.join(save_dir, f'dt_scaling_error_profiles_N{N_t}_{tag}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    # ========== Gap evolution vs time ==========
    #* Re Delta(t) itself (t = 0 is Delta_eq) for every dt; with --baseline the Q = 0 evolution on the same grids is added (dashed)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    fig_drift, ax_drift = plt.subplots(1, 1, figsize=(9, 6))
    for index, N_t in enumerate(N_t_values):
        res = results[N_t]
        color = f'C{index}'
        times_full = np.concatenate([[0.0], res['times']])
        gaps_full = np.concatenate([[res['gap_eq']], np.asarray(res['gaps'])])
        ax.plot(times_full, np.real(gaps_full), '-o', color=color, markersize=3, label=f"N_t = {N_t}, dt = {res['dt']:.4f}")
        ax_drift.plot(times_full, np.real(gaps_full - res['gap_eq']), '-o', color=color, markersize=3, label=f"N_t = {N_t}, dt = {res['dt']:.4f}")
        if results_baseline is not None:
            base = results_baseline[N_t]
            base_times_full = np.concatenate([[0.0], base['times']])
            base_gaps_full = np.concatenate([[base['gap_eq']], np.asarray(base['gaps'])])
            ax.plot(base_times_full, np.real(base_gaps_full), '--s', color=color, alpha=0.7, markersize=3, label=f"N_t = {N_t}, Q = 0")
            ax_drift.plot(base_times_full, np.real(base_gaps_full - base['gap_eq']), '--s', color=color, alpha=0.7, markersize=3, label=f"N_t = {N_t}, Q = 0")

    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel(r'Re $\Delta(t)$', fontsize=12)
    ax.set_title(f'Gap(t) for all dt (Q = {Q}, T = {temperature}, $\\eta$ = {eta})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f'dt_scaling_gap_vs_time_{tag}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")

    ax_drift.set_xlabel('t', fontsize=12)
    ax_drift.set_ylabel(r'Re $\Delta(t) - \Delta_{eq}$', fontsize=12)
    ax_drift.set_title(f'Gap drift vs time (Q = {Q})', fontsize=13, fontweight='bold')
    ax_drift.legend(fontsize=9)
    ax_drift.grid(True, alpha=0.3)
    fig_drift.tight_layout()
    path = os.path.join(save_dir, f'dt_scaling_gap_drift_vs_time_{tag}.png')
    fig_drift.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig_drift)
    print(f"Saved: {path}")

    # ========== Gap drift vs dt ==========
    drifts = np.array([np.abs(results[N_t]['gaps'][-1] - results[N_t]['gap_eq']) for N_t in N_t_values])
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(dts, drifts, 'ko-', linewidth=2, markersize=7, label=r'$|\Delta(t_{end}) - \Delta_{eq}|$')
    ax.loglog(dts, drifts[0] * dts / dts[0], 'b--', alpha=0.6, label=r'$\propto dt$')
    ax.loglog(dts, drifts[0] * (dts / dts[0])**2, 'r--', alpha=0.6, label=r'$\propto dt^2$')
    ax.set_xlabel('dt', fontsize=12)
    ax.set_ylabel('gap drift', fontsize=12)
    ax.set_title(f'Gap drift at t = {evolution_time} vs dt (Q = {Q})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(save_dir, f'dt_scaling_gap_vs_dt_{tag}.png')
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
        ax.set_title(f'{title} (t = {evolution_time}, Q = {Q})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(save_dir, f'dt_scaling_errors_vs_dt_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gap drift and error scaling with dt at fixed physical evolution time')
    parser.add_argument('-t', '--time', type=float, default=1.0, help='Physical evolution time (default: 1.0)')
    parser.add_argument('-N', '--ntpoints', type=int, nargs='+', default=[501, 751, 1001], help='Time grid sizes (default: 501 751 1001)')
    parser.add_argument('-Q', '--Q', type=float, default=0.0, help='Constant vector potential A = Q and finite-Q equilibrium start (default: 0)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature in units of T_c (default: 0.1)')
    parser.add_argument('--eta', type=float, default=0.2, help='Dynes broadening (default: 0.2)')
    parser.add_argument('--baseline', action='store_true', help='Also run Q = 0 on the same grids and report the field-induced excess drift')
    args = parser.parse_args()

    main(args.ntpoints, args.time, Q=args.Q, temperature=args.temperature, eta=args.eta, baseline=args.baseline)
