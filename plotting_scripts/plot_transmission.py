"""
Plots of post-processed 'transmitivity' data of current pulse runs: incoming current, vector potential, gap and transmitted current versus time
(2x2 panels), the transmitivity J_max / I_in,max versus the incoming and the transmitted current amplitude and, optionally, the g^R / g^K
normalization errors versus time and the energy-time distribution functions.

The normalization and energy-time data are saved by code_run.evolve_keldysh_state(track_state=True / track_every_n=...) and passed on by the
'transmitivity' postprocessing.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import postprocessed_analysis as pa
import config
from plot_static_analysis import save_figure, add_shared_legend
from plot_state_evolution import GAP_LABELS, plot_normalization_vs_time, plot_energy_representation


def normal_state_transmitivity(circuit_params):
    """2 Z_T / (2 Z_T + R_n + R_c) of a circuit parameter dict (None if circuit_params is missing or incomplete)."""
    if not isinstance(circuit_params, dict) or 'Z_T' not in circuit_params or 'R_n' not in circuit_params:
        return None
    return 2 * circuit_params['Z_T'] / (2 * circuit_params['Z_T'] + circuit_params['R_n'] + circuit_params.get('R_c', 0.0))


def gap_series(gaps, mode):
    """|Delta(t)| in the normalization of `mode` ('absolute' keeps the units of the gap, T_c is divided out by the caller)."""
    gaps = np.abs(np.asarray(gaps))
    if mode == 'absolute':
        return gaps
    if mode == 'deviation':
        return gaps - gaps[0]
    return (gaps - gaps[0]) / gaps[0]


def plot_pulse_traces(timestamp, data, directory_job_index, job_indices, gap, xlim, input_time_shift, save_plot, save_dir):
    """I_in, A, |Delta| and J versus time, one curve per inner job, 2x2 panels."""
    labels, common_text, _ = pa._job_labels(data, job_indices)
    cmap = plt.get_cmap(config.COLORMAP)
    n_jobs = len(job_indices)

    config.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE_GRID_2X2, sharex=True)
    ax_input, ax_vpot, ax_gap, ax_current = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    result = {'times': [], 'input_times': [], 'incoming_current': [], 'gap': [], 'current': [], 'vector_potential': [], 'labels': labels,
              'job_indices': list(job_indices)}
    for position, job in enumerate(job_indices):
        t_c = data['system_parameters'][job].get('critical_temperature', 1.0)
        times = np.asarray(data['times'][job])
        dt = times[1] - times[0]
        gap_plot = gap_series(data['gap_evolution'][job], gap) / (t_c if gap != 'relative' else 1.0)
        current = np.real(np.asarray(data['current_evolution'][job]))
        vector_potential = np.real(np.asarray(data['vector_potential_evolution'][job]))
        incoming = data['incoming_current_pulses'][job]
        input_times = times + input_time_shift * dt

        color = config.COLOR_1 if n_jobs == 1 else cmap(np.interp(position / max(n_jobs - 1, 1), [0, 1], config.COLORMAP_RANGE))
        style = dict(color=color, linewidth=config.LINE_WIDTH, alpha=config.LINE_ALPHA)
        if incoming is not None:
            ax_input.plot(input_times, np.real(np.asarray(incoming)), label=labels[position], **style)
        ax_vpot.plot(times, vector_potential, label=labels[position], **style)
        ax_gap.plot(times, gap_plot, label=labels[position], **style)
        ax_current.plot(times, current, label=labels[position], **style)

        result['times'].append(times)
        result['input_times'].append(input_times)
        result['incoming_current'].append(None if incoming is None else np.real(np.asarray(incoming)))
        result['gap'].append(gap_plot)
        result['current'].append(current)
        result['vector_potential'].append(vector_potential)

    for ax, ylabel in ((ax_input, r'$I_{in}(t)$'), (ax_vpot, r'$A(t)$'), (ax_gap, GAP_LABELS[gap]), (ax_current, r'$J(t)$')):
        ax.set_ylabel(ylabel, fontsize=config.LABEL_SIZE)
        ax.grid(True, alpha=config.GRID_ALPHA)
        if xlim is not None:
            ax.set_xlim(xlim)
    for ax in (ax_gap, ax_current):
        ax.set_xlabel(r'$t$', fontsize=config.LABEL_SIZE)
    fig.suptitle(r'Incoming current, vector potential, gap and transmitted current' + ('' if not common_text else '\n' + r'\small{' + common_text + '}'),
                 fontsize=config.SUPTITLE_SIZE)
    add_shared_legend(fig, ax_vpot, n_jobs > 1)

    if save_plot:
        save_figure(fig, f'transmission_traces_{timestamp}_file{directory_job_index}', timestamp, save_dir)
    result['figure'] = fig
    return result


def file_labels(datas, file_indices):
    """Legend label per result file from the system / field parameters of its first job that differ between the files, and the common ones."""
    per_file = []
    for data in datas:
        parameters = pa._numeric_items(data['system_parameters'][0])
        if data.get('field_params') is not None:
            parameters.update(pa._numeric_items(data['field_params'][0]))
        per_file.append(parameters)

    names = sorted(set.intersection(*[set(p) for p in per_file]))
    varying = [n for n in names if len(per_file) > 1 and pa._varies([p[n] for p in per_file])]
    common = {n: per_file[0][n] for n in names if n not in varying and n != 'critical_temperature'}

    labels = [", ".join(f"{pa._parameter_label(n)} = {p[n]:.4g}" for n in varying) or f"file {index}" for p, index in zip(per_file, file_indices)]
    common_text = ", ".join(f"{pa._parameter_label(n)} = {v:.4g}" for n, v in common.items())
    return labels, common_text


def plot_transmitivity(timestamp, directory_job_indices, normal_reference, save_plot, save_dir):
    """Transmitivity J_max / I_in,max versus I_in,max and versus J_max, one curve per result file, 1x2 panels."""
    if directory_job_indices is None:
        file_indices = pa._list_result_file_indices(timestamp, config.RUNNING_MACHINE)
    elif np.isscalar(directory_job_indices):
        file_indices = [int(directory_job_indices)]
    else:
        file_indices = [int(i) for i in directory_job_indices]

    datas = [pa.load_postprocessed_data(timestamp, index, config.RUNNING_MACHINE) for index in file_indices]
    for index, data in zip(file_indices, datas):
        if 'transmitivity' not in data:
            raise ValueError(f"Timestamp {timestamp}, file {index} has no transmitivity; postprocess with compute_keywords=['transmitivity'] "
                             f"(keys present: {sorted(data.keys())})")
    labels, common_text = file_labels(datas, file_indices)
    cmap = plt.get_cmap(config.COLORMAP)
    n_files = len(datas)

    config.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=config.FIGSIZE_ROW_1X2)

    result = {'incoming_amplitude': [], 'transmitted_amplitude': [], 'transmitivity': [], 'normal_state': [], 'labels': labels, 'file_indices': file_indices}
    for position, data in enumerate(datas):
        incoming = np.asarray(data['incoming_current_amplitudes'], dtype=float)
        transmitted = np.asarray(data['transmitted_current_amplitudes'], dtype=float)
        transmitivity = np.asarray(data['transmitivity'], dtype=float)
        valid = np.isfinite(transmitivity)
        order = np.argsort(incoming[valid])

        color = config.COLOR_1 if n_files == 1 else cmap(np.interp(position / max(n_files - 1, 1), [0, 1], config.COLORMAP_RANGE))
        style = dict(marker=config.MARKER, color=color, linewidth=config.LINE_WIDTH, markersize=config.MARKER_SIZE, alpha=config.LINE_ALPHA,
                     label=labels[position])
        axes[0].plot(incoming[valid][order], transmitivity[valid][order], **style)
        axes[1].plot(transmitted[valid][order], transmitivity[valid][order], **style)

        reference = None
        circuit_params = data.get('circuit_params')
        if normal_reference and circuit_params is not None:
            reference = normal_state_transmitivity(circuit_params[0])
            if reference is not None:
                for ax in axes:
                    ax.axhline(reference, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

        result['incoming_amplitude'].append(incoming)
        result['transmitted_amplitude'].append(transmitted)
        result['transmitivity'].append(transmitivity)
        result['normal_state'].append(reference)

    for ax, xlabel in zip(axes, (r'$I_{in,\mathrm{max}}$', r'$I_{T,\mathrm{max}}$')):
        ax.set_xlabel(xlabel, fontsize=config.LABEL_SIZE)
        ax.set_ylabel(r'$T = I_{T,\mathrm{max}}/I_{in,\mathrm{max}}$', fontsize=config.LABEL_SIZE)
        ax.grid(True, alpha=config.GRID_ALPHA)
    axes[0].set_title(r'Transmitivity vs incoming current', fontsize=config.TITLE_SIZE)
    axes[1].set_title(r'Transmitivity vs transmitted current', fontsize=config.TITLE_SIZE)
    fig.suptitle(r'\small{' + common_text + '}' if common_text else '', fontsize=config.SUPTITLE_SIZE)
    add_shared_legend(fig, axes[0], n_files > 1)

    if save_plot:
        save_figure(fig, f'transmitivity_{timestamp}', timestamp, save_dir)
    result['figure'] = fig
    return result


def plot_transmission(timestamp=None, directory_job_index=None, directory_job_indices=None, job_indices=None, detailed_check=None, gap=None,
                      xlim=None, input_time_shift=None, normal_reference=None, energy_xlim=None, energy_n_slices=None, save_plot=None, save_dir=None):
    """
    Time traces and transmitivity of a 'transmitivity' postprocessed timestamp (current pulse runs).

    Always: I_in, A, |Delta|, J versus time (2x2, one curve per inner job of one result file) and the transmitivity versus I_in,max and J_max
    (1x2, one curve per result file).
    If detailed_check: also the g^R and g^K normalization errors versus time (needs a run with track_state=True) and, if the file has an
    'energy_representation' (run with track_every_n), g^R(eps, t), g^K(eps, t) and the distribution functions at a few times.

    Args:
        timestamp: postprocessed timestamp (default config.TRANSMISSION_TIMESTAMP)
        directory_job_index: result file of the time traces (default config.TRANSMISSION_DIRECTORY_JOB_INDEX)
        directory_job_indices: result files of the transmitivity plot (default config.TRANSMISSION_DIRECTORY_JOB_INDICES, None = all)
        job_indices: inner jobs of the time traces and detailed checks (default config.TRANSMISSION_JOB_INDICES, None = all)
        detailed_check, gap, xlim, input_time_shift, normal_reference, energy_xlim, energy_n_slices: default to the config.TRANSMISSION_* entries
        save_plot, save_dir: default to config.SAVE_PLOTS, config.SAVE_DIR

    Returns:
        dict with 'traces' and 'transmitivity' (the plotted series and figures) and, if detailed, 'normalization_figures' and 'energy_figures'
        (dicts job -> list of figures)
    """
    timestamp = str(config.TRANSMISSION_TIMESTAMP if timestamp is None else timestamp)
    directory_job_index = config.TRANSMISSION_DIRECTORY_JOB_INDEX if directory_job_index is None else directory_job_index
    directory_job_indices = config.TRANSMISSION_DIRECTORY_JOB_INDICES if directory_job_indices is None else directory_job_indices
    job_indices = config.TRANSMISSION_JOB_INDICES if job_indices is None else job_indices
    detailed_check = config.TRANSMISSION_DETAILED_CHECK if detailed_check is None else detailed_check
    gap = config.TRANSMISSION_GAP if gap is None else gap
    xlim = config.TRANSMISSION_XLIM if xlim is None else xlim
    input_time_shift = config.TRANSMISSION_INPUT_TIME_SHIFT if input_time_shift is None else input_time_shift
    normal_reference = config.TRANSMISSION_NORMAL_REFERENCE if normal_reference is None else normal_reference
    energy_xlim = config.TRANSMISSION_ENERGY_XLIM if energy_xlim is None else energy_xlim
    energy_n_slices = config.TRANSMISSION_ENERGY_N_SLICES if energy_n_slices is None else energy_n_slices
    save_plot = config.SAVE_PLOTS if save_plot is None else save_plot
    if gap not in GAP_LABELS:
        raise ValueError(f"gap must be one of {list(GAP_LABELS)}, got '{gap}'")

    data = pa.load_postprocessed_data(timestamp, directory_job_index, config.RUNNING_MACHINE)
    missing = [key for key in ('gap_evolution', 'current_evolution', 'vector_potential_evolution', 'incoming_current_pulses', 'times') if key not in data]
    if missing:
        raise ValueError(f"Timestamp {timestamp}, file {directory_job_index} lacks {missing}; postprocess with compute_keywords=['transmitivity'] "
                         f"(keys present: {sorted(data.keys())})")

    jobs = list(range(len(data['gap_evolution']))) if job_indices is None else [int(j) for j in job_indices]

    result = {'traces': plot_pulse_traces(timestamp, data, directory_job_index, jobs, gap, xlim, input_time_shift, save_plot, save_dir),
              'transmitivity': plot_transmitivity(timestamp, directory_job_indices, normal_reference, save_plot, save_dir)}

    if detailed_check:
        if 'normalization' not in data:
            print(f"Timestamp {timestamp} has no normalization data; run the evolution with track_state=True and postprocess it with 'transmitivity'")
        else:
            result['normalization_figures'] = {job: plot_normalization_vs_time(timestamp, data, directory_job_index, job, config.TRANSMISSION_ROWS_YSCALE,
                                                                                save_plot, save_dir) for job in jobs}

        if 'energy_representation' not in data:
            print(f"Timestamp {timestamp} has no energy representation; run the evolution with track_every_n and postprocess it with 'transmitivity'")
        else:
            result['energy_figures'] = {job: plot_energy_representation(timestamp, data, directory_job_index, job, energy_xlim, energy_n_slices,
                                                                        save_plot, save_dir) for job in jobs}

    return result


if __name__ == '__main__':
    plot_transmission()
