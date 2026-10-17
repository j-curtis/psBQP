"""
Plots of post-processed 'state_evolution' data: gap, current and vector potential versus time (three panels next to each other) and,
optionally, the g^R and g^K normalization errors versus time (2x2 panels over the Pauli components).

The normalization data is saved by code_run.evolve_keldysh_state(track_state=True) and passed on by the 'state_evolution' postprocessing.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import postprocessed_analysis as pa
import config
from plot_static_analysis import save_figure, plot_row_components_2x2, add_shared_legend, PAULI_NAMES

GAP_LABELS = {'absolute': r'$|\Delta(t)|/T_c$',
              'deviation': r'$(|\Delta(t)|-|\Delta(0)|)/T_c$',
              'relative': r'$(|\Delta(t)|-|\Delta(0)|)/|\Delta(0)|$'}


def job_parameters_text(data, job):
    """Parameters (system and field, without T_c) of one inner job as a LaTeX line."""
    parameters = pa._numeric_items(data['system_parameters'][job])
    if data.get('field_params') is not None:
        parameters.update(pa._numeric_items(data['field_params'][job]))
    return ', '.join(f"{pa._parameter_label(name)} = {value:.4g}" for name, value in parameters.items() if name != 'critical_temperature')


def plot_gap_current_potential_vs_time(timestamp, data, directory_job_index, job_indices, gap, xlim, save_plot, save_dir):
    """Gap, current and vector potential versus time, one curve per inner job, three panels in a row."""
    labels, common_text, _ = pa._job_labels(data, job_indices)
    cmap = plt.get_cmap(config.COLORMAP)
    n_jobs = len(job_indices)

    config.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=config.FIGSIZE_ROW_1X3)

    result = {'times': [], 'gap': [], 'current': [], 'vector_potential': [], 'labels': labels, 'job_indices': list(job_indices)}
    for position, job in enumerate(job_indices):
        gaps = np.abs(np.asarray(data['gap_evolution'][job]))
        t_c = data['system_parameters'][job].get('critical_temperature', 1.0)
        times = np.asarray(data['times'][job]) if data.get('times') is not None and data['times'][job] is not None else np.arange(len(gaps))

        if gap == 'absolute':
            gap_plot = gaps / t_c
        elif gap == 'deviation':
            gap_plot = (gaps - gaps[0]) / t_c
        else:
            gap_plot = (gaps - gaps[0]) / gaps[0]
        current = np.real(np.asarray(data['current_evolution'][job]))
        vector_potential = np.real(np.asarray(data['vector_potential_evolution'][job]))

        color = config.COLOR_1 if n_jobs == 1 else cmap(np.interp(position / max(n_jobs - 1, 1), [0, 1], config.COLORMAP_RANGE))
        for ax, values in zip(axes, (gap_plot, current, vector_potential)):
            ax.plot(times, values, color=color, linewidth=config.LINE_WIDTH, alpha=config.LINE_ALPHA, label=labels[position])

        result['times'].append(times)
        result['gap'].append(gap_plot)
        result['current'].append(current)
        result['vector_potential'].append(vector_potential)

    for ax, ylabel in zip(axes, (GAP_LABELS[gap], r'$J(t)$', r'$A(t)$')):
        ax.set_xlabel(r'$t$', fontsize=config.LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=config.LABEL_SIZE)
        ax.grid(True, alpha=config.GRID_ALPHA)
        if xlim is not None:
            ax.set_xlim(xlim)
    fig.suptitle(r'Gap, current and vector potential vs time' + ('' if not common_text else '\n' + r'\small{' + common_text + '}'), fontsize=config.SUPTITLE_SIZE)
    add_shared_legend(fig, axes[0], n_jobs > 1)

    if save_plot:
        save_figure(fig, f'state_evolution_{timestamp}_file{directory_job_index}', timestamp, save_dir)
    result['figure'] = fig
    return result


def plot_fft_vs_frequency(timestamp, data, directory_job_index, job_indices, xlim, yscale, save_plot, save_dir):
    """|FFT| of the gap deviation, the current and the vector potential versus frequency, one curve per inner job, three panels in a row."""
    labels, common_text, _ = pa._job_labels(data, job_indices)
    cmap = plt.get_cmap(config.COLORMAP)
    n_jobs = len(job_indices)

    config.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=config.FIGSIZE_ROW_1X3)

    result = {'omega': [], 'gap_fft': [], 'current_fft': [], 'vector_potential_fft': [], 'labels': labels, 'job_indices': list(job_indices)}
    for position, job in enumerate(job_indices):
        t_c = data['system_parameters'][job].get('critical_temperature', 1.0)
        omega = np.asarray(data['omega'][job]) / t_c
        spectra = [np.abs(np.asarray(data[key][job])) for key in ('gap_fft', 'current_fft', 'vpot_fft')]
        gap_equilibrium = np.abs(data['gap_evolution'][job][0]) / t_c

        color = config.COLOR_1 if n_jobs == 1 else cmap(np.interp(position / max(n_jobs - 1, 1), [0, 1], config.COLORMAP_RANGE))
        for ax, values in zip(axes, spectra):
            ax.plot(omega, values, color=color, linewidth=config.LINE_WIDTH, alpha=config.LINE_ALPHA, label=labels[position])
            #* equilibrium gap and twice the gap (both signs of the frequency), not drawn for a (nearly) normal state
            for multiple, name in (((1, r'$\Delta$'), (2, r'$2\Delta$')) if gap_equilibrium >= config.STATE_EVOLUTION_FFT_MIN_GAP else ()):
                for sign in (-1, 1):
                    ax.axvline(sign * multiple * gap_equilibrium, color=config.COLOR_2 if n_jobs == 1 else color, linestyle=':', linewidth=1.5, alpha=0.7)
                ax.text(multiple * gap_equilibrium, 1.01, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                        fontsize=config.LEGEND_SIZE, color=config.COLOR_2 if n_jobs == 1 else color)

        result['omega'].append(omega)
        result['gap_fft'].append(spectra[0])
        result['current_fft'].append(spectra[1])
        result['vector_potential_fft'].append(spectra[2])

    for ax, ylabel in zip(axes, (r'$|\Delta(\omega)|$', r'$|J(\omega)|$', r'$|A(\omega)|$')):
        ax.set_xlabel(r'$\omega/T_c$', fontsize=config.LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=config.LABEL_SIZE)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=config.GRID_ALPHA)
        if xlim is not None:
            ax.set_xlim(xlim)
    fig.suptitle(r'Fourier transform of gap, current and vector potential' + ('' if not common_text else '\n' + r'\small{' + common_text + '}'), fontsize=config.SUPTITLE_SIZE)
    add_shared_legend(fig, axes[0], n_jobs > 1)

    if save_plot:
        save_figure(fig, f'state_evolution_fft_{timestamp}_file{directory_job_index}', timestamp, save_dir)
    result['figure'] = fig
    return result


def plot_normalization_vs_time(timestamp, data, directory_job_index, job, yscale, save_plot, save_dir):
    """g^R and g^K normalization errors on the diagonal t = t' versus time, one 2x2 figure each (Re / Im per Pauli component)."""
    history = data['normalization'][job]
    if history is None:
        print(f"Job {job}: no normalization data stored (run without track_state=True)")
        return None

    panels = [('gr_normalization', 'gr_normalization', r'$g^R$ normalization error vs time'),
              ('gk_normalization', 'gk_normalization', r'$g^K$ normalization error vs time')]
    subtitle = job_parameters_text(data, job)

    figures = []
    for key, name, title in panels:
        fig = plot_row_components_2x2(history['times'], history[key], title, yscale=yscale, subtitle=subtitle, xlabel=r'$t$')
        figures.append(fig)
        if save_plot:
            save_figure(fig, f'{name}_vs_time_{timestamp}_file{directory_job_index}_job{job}', timestamp, save_dir)
    return figures


def _slice_color(position, n_slices):
    """Color of time slice number `position` (light to dark with time)."""
    if n_slices == 1:
        return config.COLOR_1
    return plt.get_cmap(config.COLORMAP)(np.interp(position / max(n_slices - 1, 1), [0, 1], config.COLORMAP_RANGE))


def plot_energy_representation(timestamp, data, directory_job_index, job, xlim, n_slices, save_plot, save_dir):
    """
    Energy-time representations of one inner job (from 'energy_representation' of the postprocessing, runs with track_every_n):
    g^R(eps, t) and g^K(eps, t) as 2x2 figures over the Pauli components and the distribution functions f_L = tanh + n_0, f_T = n_3.
    n_slices times (first ... last tracked step, labelled by the window-center time) are drawn, real part solid, imaginary part dashed.
    The saved g^R and g^K are the full functions (tau_3 of the delta function, 2 tau_3 tanh(eps/2T) and the thermal gap term added back by the postprocessing);
    the distributions are f_L = tanh(eps/2T) + n_0 and f_T = n_3 from the saved occupation n.
    """
    entry = data['energy_representation'][job]
    if entry is None:
        print(f"Job {job}: no energy representation stored (run without track_every_n)")
        return None

    t_c = data['system_parameters'][job].get('critical_temperature', 1.0)
    energy = np.asarray(entry['energy_grid']) / t_c
    n_steps = len(entry['step_indices'])
    slices = np.unique(np.linspace(0, n_steps - 1, min(n_slices, n_steps)).round().astype(int))
    parameters = job_parameters_text(data, job)
    subtitle = (parameters + ', ' if parameters else '') + r'solid: $\mathrm{Re}$, dashed: $\mathrm{Im}$'

    def draw(ax, values, position_slices=slices):
        for position, s in enumerate(position_slices):
            color = _slice_color(position, len(position_slices))
            ax.plot(energy, np.real(values[s]), color=color, linewidth=config.LINE_WIDTH, label=r'$t=%.3g$' % entry['center_times'][s])
            ax.plot(energy, np.imag(values[s]), color=color, linewidth=config.LINE_WIDTH, linestyle='--', label='_nolegend_')

    def finish(fig, axes, title, name):
        for ax in np.ravel(axes):
            ax.set_xlabel(r'$\epsilon/T_c$', fontsize=config.LABEL_SIZE)
            ax.grid(True, alpha=config.GRID_ALPHA)
            if xlim is not None:
                ax.set_xlim(xlim)
        fig.suptitle(title + '\n' + r'\small{' + subtitle + '}', fontsize=config.SUPTITLE_SIZE)
        add_shared_legend(fig, np.ravel(axes)[0], len(slices) > 1)
        if save_plot:
            save_figure(fig, f'energy_{name}_{timestamp}_file{directory_job_index}_job{job}', timestamp, save_dir)

    config.apply_style()
    figures = []
    #~ for key_full, key_raw, title, name in (('gr', 'gr_raw', r'$g^R(\epsilon, t)$', 'gr'), ('gk', 'gk_raw', r'$g^K(\epsilon, t)$', 'gk')):
    #~     values = entry[key_full if full else key_raw]
    for key, title, name in (('gr', r'$g^R(\epsilon, t)$', 'gr'), ('gk', r'$g^K(\epsilon, t)$', 'gk')):
        values = entry[key]
        fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE_GRID_2X2, sharex=True)
        for p, ax in enumerate(axes.flat):
            draw(ax, values[:, p])
            ax.set_title(r'$' + PAULI_NAMES[p] + r'$ component', fontsize=config.TITLE_SIZE)
        finish(fig, axes, title, name)
        figures.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=config.FIGSIZE_ROW_1X2)
    #~ draw(axes[0], entry['distribution_L'])
    draw(axes[0], entry['thermal'][None, :] + entry['occupation'][:, 0])
    axes[0].plot(energy, entry['thermal'], color='black', linestyle=':', linewidth=config.LINE_WIDTH, label=r'$\tanh(\epsilon/2T)$')
    axes[0].set_title(r'energy mode $f_L = \tanh(\epsilon/2T) + n_0$', fontsize=config.TITLE_SIZE)
    #~ draw(axes[1], entry['distribution_T'])
    draw(axes[1], entry['occupation'][:, 3])
    axes[1].set_title(r'charge mode $f_T = n_3$', fontsize=config.TITLE_SIZE)
    finish(fig, axes, r'Distribution functions $f(\epsilon, t)$', 'distribution')
    figures.append(fig)
    return figures


def plot_state_evolution(timestamp=None, detailed_check=None, directory_job_index=None, job_indices=None, gap=None, xlim=None,
                         include_fft=None, fft_xlim=None, fft_yscale=None, energy_xlim=None, energy_n_slices=None,
                         save_plot=None, save_dir=None):
    """
    Gap, current and vector potential versus time for the inner jobs of a 'state_evolution' postprocessed file (three panels in a row).
    If include_fft: a second figure with |FFT| of the gap deviation, current and vector potential versus frequency (same layout).
    If detailed_check: also the g^R and g^K normalization errors versus time (2x2 panels, one set per job) and, if the postprocessed file has an
    'energy_representation' (run with track_every_n), g^R(eps, t), g^K(eps, t) and the distribution functions at a few times (see plot_energy_representation).

    Args:
        timestamp: postprocessed timestamp (default config.STATE_EVOLUTION_TIMESTAMP)
        detailed_check: bool (default config.STATE_EVOLUTION_DETAILED_CHECK)
        directory_job_index: result file index (default config.STATE_EVOLUTION_DIRECTORY_JOB_INDEX)
        job_indices: inner jobs (default config.STATE_EVOLUTION_JOB_INDICES, None = all)
        gap: 'absolute', 'deviation' or 'relative' (default config.STATE_EVOLUTION_GAP)
        xlim: time axis limits (default config.STATE_EVOLUTION_XLIM)
        include_fft, fft_xlim, fft_yscale: default to config.STATE_EVOLUTION_INCLUDE_FFT, _FFT_XLIM, _FFT_YSCALE
        energy_xlim, energy_n_slices: default to config.STATE_EVOLUTION_ENERGY_XLIM, _ENERGY_N_SLICES
        save_plot, save_dir: default to config.SAVE_PLOTS, config.SAVE_DIR

    Returns:
        dict with the plotted series ('times', 'gap', 'current', 'vector_potential', 'labels', 'job_indices', 'figure') and, if detailed,
        'normalization_figures' and 'energy_figures' (dicts job -> list of figures)
    """
    timestamp = str(config.STATE_EVOLUTION_TIMESTAMP if timestamp is None else timestamp)
    detailed_check = config.STATE_EVOLUTION_DETAILED_CHECK if detailed_check is None else detailed_check
    directory_job_index = config.STATE_EVOLUTION_DIRECTORY_JOB_INDEX if directory_job_index is None else directory_job_index
    job_indices = config.STATE_EVOLUTION_JOB_INDICES if job_indices is None else job_indices
    gap = config.STATE_EVOLUTION_GAP if gap is None else gap
    xlim = config.STATE_EVOLUTION_XLIM if xlim is None else xlim
    include_fft = config.STATE_EVOLUTION_INCLUDE_FFT if include_fft is None else include_fft
    fft_xlim = config.STATE_EVOLUTION_FFT_XLIM if fft_xlim is None else fft_xlim
    fft_yscale = config.STATE_EVOLUTION_FFT_YSCALE if fft_yscale is None else fft_yscale
    energy_xlim = config.STATE_EVOLUTION_ENERGY_XLIM if energy_xlim is None else energy_xlim
    energy_n_slices = config.STATE_EVOLUTION_ENERGY_N_SLICES if energy_n_slices is None else energy_n_slices
    save_plot = config.SAVE_PLOTS if save_plot is None else save_plot
    if gap not in GAP_LABELS:
        raise ValueError(f"gap must be one of {list(GAP_LABELS)}, got '{gap}'")

    data = pa.load_postprocessed_data(timestamp, directory_job_index, config.RUNNING_MACHINE)
    missing = [key for key in ('gap_evolution', 'current_evolution', 'vector_potential_evolution') if key not in data]
    if missing:
        raise ValueError(f"Timestamp {timestamp}, file {directory_job_index} lacks {missing}; postprocess with compute_keywords=['state_evolution'] "
                         f"(keys present: {sorted(data.keys())})")

    jobs = list(range(len(data['gap_evolution']))) if job_indices is None else [int(j) for j in job_indices]

    result = plot_gap_current_potential_vs_time(timestamp, data, directory_job_index, jobs, gap, xlim, save_plot, save_dir)

    if include_fft:
        if 'gap_fft' not in data:
            print(f"Timestamp {timestamp}, file {directory_job_index} has no Fourier transforms; postprocess with 'state_evolution' (or 'spectral_analysis') again")
        else:
            result['fft'] = plot_fft_vs_frequency(timestamp, data, directory_job_index, jobs, fft_xlim, fft_yscale, save_plot, save_dir)

    if detailed_check:
        if 'normalization' not in data:
            print(f"Timestamp {timestamp} has no normalization data; run the evolution with track_state=True and postprocess it with 'state_evolution'")
        else:
            result['normalization_figures'] = {job: plot_normalization_vs_time(timestamp, data, directory_job_index, job,
                                                                                config.STATE_EVOLUTION_ROWS_YSCALE, save_plot, save_dir)
                                               for job in jobs}

        if 'energy_representation' not in data:
            print(f"Timestamp {timestamp} has no energy representation; run the evolution with track_every_n and postprocess it with 'state_evolution'")
        else:
            result['energy_figures'] = {job: plot_energy_representation(timestamp, data, directory_job_index, job, energy_xlim, energy_n_slices,
                                                                        save_plot, save_dir) for job in jobs}

    return result


if __name__ == '__main__':
    plot_state_evolution()
