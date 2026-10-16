"""
Plots of post-processed 'static' data: |gap|/T_c versus the varied parameter and, optionally, the Green's function rows
g^R(t=-1,t'), g^K(t=-1,t') and the consistency (normalization / FDT) error rows, each as 2x2 panels over the Pauli components.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import postprocessed_analysis as pa
import config

PAULI_NAMES = [r'\tau_0', r'\tau_1', r'\tau_2', r'\tau_3']


def save_figure(fig, filename, timestamp, save_dir=None):
    """Save into the config save folder; png/svg through demler_tools' extended_savefig (data timestamp in the metadata), pdf directly."""
    path = os.path.join(config.resolve_save_dir(save_dir), f'{filename}.{config.SAVE_FORMAT}')
    if config.SAVE_FORMAT == 'pdf':
        fig.savefig(path, metadata={'Subject': f'data timestamp {timestamp}'})
    else:
        pa.extended_savefig(fig, path, timestamp)


def add_shared_legend(fig, ax, show=True):
    """One legend below all panels, built from the handles of `ax`; also applies tight_layout leaving room for it."""
    handles, labels = ax.get_legend_handles_labels()
    if not show or not handles:
        fig.tight_layout()
        return
    ncol = min(len(handles), config.SHARED_LEGEND_NCOL)
    rows = -(-len(handles) // ncol)
    bottom = (0.3 + rows * 1.6 * config.LEGEND_SIZE / 72) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=config.LEGEND_SIZE, framealpha=config.LEGEND_FRAME_ALPHA,
               bbox_to_anchor=(0.5, 0.0))


#* parameters of the post-processed data that correspond to the sweep variables of equilibrium_sweep.py
EQUILIBRIUM_SWEEPS = {
    'temperature': ('temperature_sweep', 'temperatures'),
    'amplitude': ('vector_potential_sweep', 'vector_potentials'),
    'vector_potential': ('vector_potential_sweep', 'vector_potentials'),
}


def load_equilibrium_data(data_file=None):
    """Combined pickle written by equilibrium_sweep.run_equilibrium_sweeps (default config.STATIC_ANALYSIS_DATA_FILE)."""
    path = config.resolve_data_file(config.STATIC_ANALYSIS_DATA_FILE if data_file is None else data_file)
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_gap_vs_parameter(timestamp, directory_job_indices=None, inner_parameter=None, outer_parameters=None,
                          error_type=None, include_current=None, data_file=None, ylim=None, xlim=None,
                          save_plot=None, save_dir=None, verbose=True):
    """
    |gap|/T_c (top panel) and the current (bottom panel) versus the parameter varying inside the result files, one curve per file.
    If a data_file (pickle of equilibrium_sweep.py) is given, the equilibrium sweep of the same variable (temperature at A = 0,
    or vector potential at the sweep temperature) is overlaid as a dashed line in COLOR_2.

    Args:
        timestamp: post-processed timestamp
        directory_job_indices: int or list of result file indices (None = all)
        inner_parameter: x axis parameter (default: temperature if it varies)
        include_current: add the current panel (default config.STATIC_ANALYSIS_INCLUDE_CURRENT)
        data_file: equilibrium pickle (default config.STATIC_ANALYSIS_DATA_FILE, None = no overlay)

    Returns:
        dict with 'detection', 'x', 'gaps', 'gap_errors', 'currents', 'labels', 'file_indices', 'inner_parameter', 'equilibrium'
    """
    error_type = config.STATIC_ANALYSIS_ERROR_TYPE if error_type is None else error_type
    include_current = config.STATIC_ANALYSIS_INCLUDE_CURRENT if include_current is None else include_current
    data_file = config.STATIC_ANALYSIS_DATA_FILE if data_file is None else data_file
    save_plot = config.SAVE_PLOTS if save_plot is None else save_plot
    if error_type not in ('range', 'std'):
        raise ValueError(f"error_type must be 'range' or 'std', got '{error_type}'")

    detection = pa.detect_job_parameters(timestamp, config.RUNNING_MACHINE, directory_job_indices, verbose=verbose)

    if inner_parameter is None:
        if 'temperature' in detection['inner']:
            inner_parameter = 'temperature'
        elif detection['inner']:
            inner_parameter = detection['inner'][0]
        else:
            raise ValueError("No parameter varies inside the files; give inner_parameter explicitly")
    if inner_parameter not in detection['inner_values']:
        raise ValueError(f"'{inner_parameter}' does not vary inside the files. Inner parameters: {detection['inner']}")

    outer_names = outer_parameters if outer_parameters is not None else list(detection['outer'].keys())
    if outer_parameters is None and 'dt' in outer_names and 'time_sampling' in outer_names:
        outer_names = [n for n in outer_names if n != 'dt'] + ['dt']

    config.apply_style()
    if include_current:
        fig, (ax, ax_current) = plt.subplots(2, 1, figsize=config.FIGSIZE_STACKED, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
        ax_current = None
    cmap = plt.get_cmap(config.COLORMAP)
    n_files = len(detection['file_indices'])

    x_list, gap_list, error_list, current_list, labels = [], [], [], [], []
    for position, file_index in enumerate(detection['file_indices']):
        data = pa.load_postprocessed_data(timestamp, file_index, config.RUNNING_MACHINE)
        x = np.asarray(detection['inner_values'][inner_parameter][position], dtype=float)
        t_c = data['system_parameters'][0].get('critical_temperature', 1.0)
        gaps = np.abs(np.asarray(data['averaged_gaps'])) / t_c
        currents = np.real(np.asarray(data['averaged_currents']))

        #* errors as (2, n): lower and upper distance from the averaged gap
        gap_errors = None
        if error_type == 'range' and 'gap_evolution' in data:
            last_gaps = [np.abs(np.asarray(g))[-50:] / t_c for g in data['gap_evolution']]
            lower = np.maximum(gaps - np.array([g.min() for g in last_gaps]), 0.0)
            upper = np.maximum(np.array([g.max() for g in last_gaps]) - gaps, 0.0)
            gap_errors = np.array([lower, upper])
        elif 'gap_std' in data:
            gap_std = np.asarray(data['gap_std']) / t_c
            gap_errors = np.array([gap_std, gap_std])

        invalid = ~np.isfinite(gaps)
        if np.any(invalid):
            print(f"  file {file_index}: {int(invalid.sum())} NaN gap value(s) at {inner_parameter} = "
                  + ", ".join(f"{v:.4g}" for v in x[invalid]) + " (not plotted)")

        label = pa._outer_label(detection, position, outer_names)
        if verbose:
            print(f"  file {file_index}, {label}:")
            print(f"    {inner_parameter:>16s}  {'|gap|/T_c':>14s}  {'Re(current)':>14s}")
            for x_value, gap_value, current_value in zip(x[~invalid], gaps[~invalid], currents[~invalid]):
                print(f"    {x_value:16.6g}  {gap_value:14.8e}  {current_value:14.8e}")

        color = config.COLOR_1 if n_files == 1 else cmap(np.interp(position / max(n_files - 1, 1), [0, 1], config.COLORMAP_RANGE))
        style = dict(marker=config.MARKER, color=color, linewidth=config.LINE_WIDTH, markersize=config.MARKER_SIZE,
                     alpha=config.LINE_ALPHA, label=label)
        if gap_errors is None:
            ax.plot(x[~invalid], gaps[~invalid], **style)
        else:
            ax.errorbar(x[~invalid], gaps[~invalid], yerr=gap_errors[:, ~invalid], capsize=config.CAPSIZE,
                        elinewidth=config.ERRORBAR_LINE_WIDTH, **style)
        if ax_current is not None:
            ax_current.plot(x[~invalid], currents[~invalid], **style)
        x_list.append(x)
        gap_list.append(gaps)
        error_list.append(gap_errors)
        current_list.append(currents)
        labels.append(label)

    #* equilibrium reference (dashed, COLOR_2)
    equilibrium = None
    if data_file is not None:
        if inner_parameter not in EQUILIBRIUM_SWEEPS:
            print(f"No equilibrium sweep corresponds to '{inner_parameter}'; data_file not plotted")
        else:
            sweep_key, x_key = EQUILIBRIUM_SWEEPS[inner_parameter]
            sweep = load_equilibrium_data(data_file)[sweep_key]
            t_c = sweep['parameters']['critical_temperature']
            eq_x = np.asarray(sweep[x_key], dtype=float)
            eq_gaps = np.abs(np.asarray(sweep['gaps'])) / t_c
            eq_currents = config.STATIC_ANALYSIS_EQUILIBRIUM_CURRENT_SIGN * np.real(np.asarray(sweep['currents']))
            ok = np.isfinite(eq_gaps)
            eq_label = r'equilibrium' + (r', $T/T_c = %.3g$' % (sweep['parameters']['temperature'] / t_c) if 'temperature' in sweep['parameters'] else r', $A=0$')
            eq_style = dict(color=config.COLOR_2, linestyle='--', marker='s', linewidth=config.LINE_WIDTH, markersize=config.MARKER_SIZE,
                            alpha=config.LINE_ALPHA, label=eq_label)
            ax.plot(eq_x[ok], eq_gaps[ok], **eq_style)
            if ax_current is not None:
                ax_current.plot(eq_x[ok], eq_currents[ok], **eq_style)
            equilibrium = {'x': eq_x, 'gaps': eq_gaps, 'currents': eq_currents, 'sweep': sweep_key}

    x_label = pa._parameter_label(inner_parameter)
    ax.set_ylabel(r'$|\Delta|/T_c$', fontsize=config.LABEL_SIZE)
    constants = pa._constants_text(detection, exclude=(inner_parameter,) + tuple(outer_names))
    ax.set_title(r'Gap vs ' + x_label + ('' if not constants else '\n' + r'\small{' + constants + '}'), fontsize=config.TITLE_SIZE)
    if ax_current is not None:
        ax_current.set_ylabel(r'$J$', fontsize=config.LABEL_SIZE)
        ax_current.set_xlabel(x_label, fontsize=config.LABEL_SIZE)
    else:
        ax.set_xlabel(x_label, fontsize=config.LABEL_SIZE)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    for axis in (ax, ax_current):
        if axis is not None:
            axis.grid(True, alpha=config.GRID_ALPHA)
    add_shared_legend(fig, ax, n_files > 1 or bool(detection['outer']) or equilibrium is not None)

    if save_plot:
        save_figure(fig, f'gap_vs_{inner_parameter}_{timestamp}', timestamp, save_dir)

    return {'detection': detection, 'x': x_list, 'gaps': gap_list, 'gap_errors': error_list, 'currents': current_list, 'labels': labels,
            'file_indices': detection['file_indices'], 'inner_parameter': inner_parameter, 'equilibrium': equilibrium}


def plot_row_components_2x2(t_prime, rows, title, yscale='linear', subtitle=None, xlabel=r"$t'$"):
    """
    2x2 panels (tau_0, tau_1 / tau_2, tau_3) of a (4, N_t) complex row; every panel shows the real part (COLOR_1)
    and the imaginary part (COLOR_2). The figure title is `title` (with the parameters `subtitle` below it), the panel titles
    name the component.
    """
    config.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE_GRID_2X2, sharex=True)
    for p, ax in enumerate(axes.flat):
        ax.plot(t_prime, np.real(rows[p]), color=config.COLOR_1, linewidth=config.LINE_WIDTH, label=r'$\mathrm{Re}$')
        ax.plot(t_prime, np.imag(rows[p]), color=config.COLOR_2, linewidth=config.LINE_WIDTH, label=r'$\mathrm{Im}$')
        ax.set_title(r'$' + PAULI_NAMES[p] + r'$ component', fontsize=config.TITLE_SIZE)
        ax.set_ylabel(r'$' + PAULI_NAMES[p] + r'$', fontsize=config.LABEL_SIZE)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=config.GRID_ALPHA)
        ax.legend(fontsize=config.LEGEND_SIZE, framealpha=config.LEGEND_FRAME_ALPHA, loc='best')
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=config.LABEL_SIZE)
    fig.suptitle(title + (('\n' + r'\small{' + subtitle + '}') if subtitle else ''), fontsize=config.SUPTITLE_SIZE)
    fig.tight_layout()
    return fig


def plot_green_function_components(timestamp, directory_job_index=0, job_index=0, source=None, save_plot=None, save_dir=None):
    """g^R(t=-1,t') and g^K(t=-1,t') as two figures with 2x2 Pauli-component panels. Returns the rows dict of get_state_rows."""
    source = config.STATIC_ANALYSIS_ROWS_SOURCE if source is None else source
    save_plot = config.SAVE_PLOTS if save_plot is None else save_plot

    rows_data = pa.get_state_rows(timestamp, directory_job_index, job_index, source, config.RUNNING_MACHINE)
    subtitle = pa._rows_title(rows_data)

    figures = []
    for key, name, label in (('gr_row', 'gr', r'$g^R(t=-1,t^\prime)$'), ('gk_row', 'gk', r'$g^K(t=-1,t^\prime)$')):
        fig = plot_row_components_2x2(rows_data["t_prime"], rows_data[key], label, subtitle=subtitle)
        figures.append(fig)
        if save_plot:
            save_figure(fig, f'{name}_components_{timestamp}_file{directory_job_index}_job{job_index}', timestamp, save_dir)

    rows_data['figures'] = figures
    return rows_data


def plot_consistency_components(timestamp, directory_job_index=0, job_index=0, source=None, yscale=None,
                                save_plot=None, save_dir=None, print_maxima=True):
    """
    g^R normalization, g^K normalization and FDT error rows at t=-1 versus t', one 2x2 figure each.
    Returns the rows dict with 'maxima' and 'figures'.
    """
    source = config.STATIC_ANALYSIS_ROWS_SOURCE if source is None else source
    yscale = config.STATIC_ANALYSIS_ROWS_YSCALE if yscale is None else yscale
    save_plot = config.SAVE_PLOTS if save_plot is None else save_plot

    rows_data = pa.get_state_rows(timestamp, directory_job_index, job_index, source, config.RUNNING_MACHINE)
    t_prime = rows_data['t_prime']
    subtitle = pa._rows_title(rows_data)

    #* vector potential of the job (average of the last 50 timesteps, as stored by the 'static' postprocessing)
    data = pa.load_postprocessed_data(timestamp, directory_job_index, config.RUNNING_MACHINE)
    if 'averaged_vector_potentials' in data:
        vector_potential = float(np.real(data['averaged_vector_potentials'][job_index]))
        subtitle = (subtitle + ', ' if subtitle else '') + r'$A = %.3g$' % vector_potential

    panels = [('gr_normalization_row', 'gr_normalization', r'$g^R$ normalization error', 'g^R normalization'),
              ('gk_normalization_row', 'gk_normalization', r'$g^K$ normalization error', 'g^K normalization'),
              ('fdt_row', 'fdt', r'FDT error', 'FDT')]

    maxima, figures = {}, []
    for key, name, title, maxima_name in panels:
        rows = rows_data[key]
        fig = plot_row_components_2x2(t_prime, rows, title, yscale=yscale, subtitle=subtitle)
        figures.append(fig)
        if save_plot:
            save_figure(fig, f'{name}_error_components_{timestamp}_file{directory_job_index}_job{job_index}', timestamp, save_dir)

        components = {}
        for p in range(4):
            for part, values in (('Re', np.real(rows[p])), ('Im', np.imag(rows[p]))):
                index = int(np.nanargmax(np.abs(values))) if np.any(np.isfinite(values)) else 0
                components[f'{part} tau_{p}'] = (float(np.abs(values[index])), float(t_prime[index]))
        overall_key = max(components, key=lambda k: components[k][0])
        maxima[maxima_name] = {'overall': components[overall_key][0], 'overall_t_prime': components[overall_key][1],
                               'overall_component': overall_key, 'components': components}

    if print_maxima:
        print(f"Maximal errors at t = -1 (timestamp {timestamp}, file {directory_job_index}, job {job_index}, source: {rows_data['source']}"
              + (f", raw run {rows_data['raw_timestamp']}" if rows_data['raw_timestamp'] else "") + ")")
        for name, info in maxima.items():
            print(f"  {name}: max |error| = {info['overall']:.3e} ({info['overall_component']}, at t' = {info['overall_t_prime']:.3f})")
            for component, (value, at) in info['components'].items():
                print(f"      {component}: {value:.3e} (t' = {at:.3f})")


    rows_data['maxima'] = maxima
    rows_data['figures'] = figures
    return rows_data


def plot_static_analysis(timestamp=None, detailed_analysis=None, directory_job_indices=None, job_indices=None,
                         inner_parameter=None, include_current=None, data_file=None, save_plot=None, save_dir=None):
    """
    Static analysis of a post-processed timestamp.

    Always: |gap|/T_c and current versus the varied parameter (2x1), with the equilibrium sweep from data_file if given.
    If detailed_analysis: for every selected result file and inner job also the 2x2 g^R and g^K component plots and the
    2x2 consistency-error plots.

    Args:
        timestamp: post-processed timestamp (default config.STATIC_ANALYSIS_TIMESTAMP)
        detailed_analysis: bool (default config.STATIC_ANALYSIS_DETAILED)
        directory_job_indices: int or list of result file indices (default config.STATIC_ANALYSIS_DIRECTORY_JOB_INDICES, None = all)
        job_indices: inner jobs for the detailed plots (default config.STATIC_ANALYSIS_JOB_INDICES, None = all inner jobs of each file)
        inner_parameter: parameter on the x axis (default: temperature if it varies)
        include_current: add the current panel below the gap (default config.STATIC_ANALYSIS_INCLUDE_CURRENT)
        data_file: pickle of equilibrium_sweep.py overlaid on the gap and current panels (default config.STATIC_ANALYSIS_DATA_FILE)
        save_plot, save_dir: default to config.SAVE_PLOTS, config.SAVE_DIR

    Returns:
        dict with 'gap' (return value of plot_gap_vs_parameter) and, if detailed, 'rows' and 'consistency'
        (dicts keyed by (file_index, job_index))
    """
    timestamp = str(config.STATIC_ANALYSIS_TIMESTAMP if timestamp is None else timestamp)
    detailed_analysis = config.STATIC_ANALYSIS_DETAILED if detailed_analysis is None else detailed_analysis
    directory_job_indices = config.STATIC_ANALYSIS_DIRECTORY_JOB_INDICES if directory_job_indices is None else directory_job_indices
    job_indices = config.STATIC_ANALYSIS_JOB_INDICES if job_indices is None else job_indices

    result = {'gap': plot_gap_vs_parameter(timestamp, directory_job_indices, inner_parameter, include_current=include_current, data_file=data_file, save_plot=save_plot, save_dir=save_dir)}

    if detailed_analysis:
        result['rows'], result['consistency'] = {}, {}
        for file_index in result['gap']['file_indices']:
            n_jobs = len(pa.load_postprocessed_data(timestamp, file_index, config.RUNNING_MACHINE)['system_parameters'])
            for job_index in (range(n_jobs) if job_indices is None else job_indices):
                result['rows'][(file_index, job_index)] = plot_green_function_components(timestamp, file_index, job_index, save_plot=save_plot, save_dir=save_dir)
                result['consistency'][(file_index, job_index)] = plot_consistency_components(timestamp, file_index, job_index, save_plot=save_plot, save_dir=save_dir)

    return result


if __name__ == '__main__':
    plot_static_analysis()
