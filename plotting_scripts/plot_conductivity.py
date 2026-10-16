"""
Optical conductivity sigma(omega) = sign * J(omega) / (i omega A(omega)) from the Fourier transforms of a post-processed 'state_evolution'
(or 'spectral_analysis') timestamp. Mimics data_analysis_sc.extract_conductivity: Re sigma, Im sigma, J(omega) and E(omega) = i omega A(omega)
as 2x2 panels for omega > 0, one figure per inner job.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import postprocessed_analysis as pa
import config
from plot_static_analysis import save_figure
from plot_state_evolution import job_parameters_text

SIGMA_THRESHOLD = 1e-12     # |sigma| below this is not plotted


def compute_conductivity(data, job, threshold, sign):
    """
    sigma(omega) of one inner job from the saved Fourier transforms.

    The ratio does not depend on the phase shift or the dt prefactor of the saved transforms; they are divided by dt only so that
    the threshold acts on the raw FFT as in extract_conductivity.

    Returns:
        dict with 'omega', 'sigma', 'j_omega', 'e_omega', 'a_omega', 'mask' (valid points), 'gap_equilibrium'
    """
    omega = np.asarray(data['omega'][job])
    dt = data['times'][job][1] - data['times'][job][0]
    j_omega = np.asarray(data['current_fft'][job]) / dt
    a_omega = np.asarray(data['vpot_fft'][job]) / dt

    mask = (np.abs(a_omega) > threshold) & (np.abs(omega) > threshold)
    sigma = np.zeros_like(j_omega, dtype=complex)
    sigma[mask] = sign * j_omega[mask] / (1j * omega[mask] * a_omega[mask])

    return {'omega': omega, 'sigma': sigma, 'j_omega': j_omega, 'e_omega': 1j * omega * a_omega, 'a_omega': a_omega, 'mask': mask,
            'gap_equilibrium': float(np.abs(data['gap_evolution'][job][0]))}


def plot_conductivity_job(timestamp, data, directory_job_index, job, threshold, sign, xlim, ylim_re, ylim_im, external, save_plot, save_dir):
    """2x2 figure: Re sigma, Im sigma (top), J(omega), E(omega) (bottom), omega > 0."""
    result = compute_conductivity(data, job, threshold, sign)
    omega, sigma, mask = result['omega'], result['sigma'], result['mask']

    valid_omega = omega[mask]
    if len(valid_omega):
        x_limits = (0, max(valid_omega.max(), 0) * 1.1)
        print(f"Job {job}: valid frequencies [{valid_omega.min():.3f}, {valid_omega.max():.3f}], {int(mask.sum())}/{len(omega)} points, "
              f"max |A(omega)| = {np.abs(result['a_omega']).max():.3e}")
    else:
        x_limits = (0, 10 * result['gap_equilibrium'])
        print(f"Job {job}: no valid frequency (max |A(omega)| = {np.abs(result['a_omega']).max():.3e}, threshold {threshold:.1e})")
    if xlim is not None:
        x_limits = xlim

    config.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE_GRID_2X2)
    style = dict(linewidth=config.LINE_WIDTH, alpha=config.LINE_ALPHA)

    plot_mask = (omega > threshold) & (np.abs(sigma) > SIGMA_THRESHOLD)
    axes[0, 0].plot(omega[plot_mask], np.real(sigma)[plot_mask], color=config.COLOR_1, label='simulation', **style)
    axes[0, 1].plot(omega[plot_mask], np.imag(sigma)[plot_mask], color=config.COLOR_2, label='simulation', **style)

    for position, entry in enumerate(external or []):
        color = f'C{position + 2}'
        if entry.get('sigma_re') is not None:
            axes[0, 0].plot(entry['omega'], entry['sigma_re'], '--', color=color, label=entry.get('label', f'external {position + 1}'), **style)
        if entry.get('sigma_im') is not None:
            axes[0, 1].plot(entry['omega'], entry['sigma_im'], '--', color=color, label=entry.get('label', f'external {position + 1}'), **style)

    for ax, values, name in ((axes[1, 0], result['j_omega'], r'$J(\omega)$'), (axes[1, 1], result['e_omega'], r'$E(\omega)=i\omega A(\omega)$')):
        ax.plot(omega, np.real(values), color=config.COLOR_1, label=r'$\mathrm{Re}$', **style)
        ax.plot(omega, np.imag(values), color=config.COLOR_2, label=r'$\mathrm{Im}$', **style)
        ax.set_ylabel(name, fontsize=config.LABEL_SIZE)

    axes[0, 0].set_ylabel(r'$\mathrm{Re}\,\sigma(\omega)$', fontsize=config.LABEL_SIZE)
    axes[0, 1].set_ylabel(r'$\mathrm{Im}\,\sigma(\omega)$', fontsize=config.LABEL_SIZE)
    titles = (r'Real part of $\sigma$', r'Imaginary part of $\sigma$', r'Current', r'Electric field')
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=config.TITLE_SIZE)
        ax.set_xlabel(r'$\omega$', fontsize=config.LABEL_SIZE)
        ax.set_xlim(x_limits)
        ax.grid(True, alpha=config.GRID_ALPHA)
        ax.legend(fontsize=config.LEGEND_SIZE, framealpha=config.LEGEND_FRAME_ALPHA, loc='best')
        if len(valid_omega):
            ax.axvline(valid_omega.min(), color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
            ax.axvline(valid_omega.max(), color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
        ax.axvline(2 * result['gap_equilibrium'], color=config.COLOR_2, linestyle=':', linewidth=1, alpha=0.4)
    if ylim_re is not None:
        axes[0, 0].set_ylim(ylim_re)
    if ylim_im is not None:
        axes[0, 1].set_ylim(ylim_im)

    subtitle = job_parameters_text(data, job)
    fig.suptitle(r'Optical conductivity $\sigma(\omega) = ' + ('-' if sign < 0 else '') + r'J(\omega)/(i\omega A(\omega))$'
                 + ('\n' + r'\small{' + subtitle + '}' if subtitle else ''), fontsize=config.SUPTITLE_SIZE)
    fig.tight_layout()

    if save_plot:
        save_figure(fig, f'conductivity_{timestamp}_file{directory_job_index}_job{job}', timestamp, save_dir)
    result['figure'] = fig
    return result


def plot_conductivity(timestamp=None, directory_job_index=None, job_indices=None, threshold=None, sign=None, xlim=None, ylim_re=None,
                      ylim_im=None, external=None, save_plot=None, save_dir=None):
    """
    Optical conductivity sigma(omega) = sign * J(omega) / (i omega A(omega)) for the inner jobs of a post-processed file.

    Args:
        timestamp: postprocessed timestamp with Fourier transforms (default config.CONDUCTIVITY_TIMESTAMP)
        directory_job_index: result file index (default config.CONDUCTIVITY_DIRECTORY_JOB_INDEX)
        job_indices: inner jobs, one 2x2 figure each (default config.CONDUCTIVITY_JOB_INDICES, None = all)
        threshold: minimum |A(omega)| and |omega| for a valid point (default config.CONDUCTIVITY_THRESHOLD)
        sign: +1 or -1 (default config.CONDUCTIVITY_SIGN)
        xlim, ylim_re, ylim_im: axis limits (default config.CONDUCTIVITY_XLIM / _YLIM_RE / _YLIM_IM)
        external: optional list of dicts {'omega', 'sigma_re', 'sigma_im', 'label'} drawn dashed on the sigma panels
        save_plot, save_dir: default to config.SAVE_PLOTS, config.SAVE_DIR

    Returns:
        dict job -> compute_conductivity result plus 'figure' (None for jobs with zero vector potential)
    """
    timestamp = str(config.CONDUCTIVITY_TIMESTAMP if timestamp is None else timestamp)
    directory_job_index = config.CONDUCTIVITY_DIRECTORY_JOB_INDEX if directory_job_index is None else directory_job_index
    job_indices = config.CONDUCTIVITY_JOB_INDICES if job_indices is None else job_indices
    threshold = config.CONDUCTIVITY_THRESHOLD if threshold is None else threshold
    sign = config.CONDUCTIVITY_SIGN if sign is None else sign
    xlim = config.CONDUCTIVITY_XLIM if xlim is None else xlim
    ylim_re = config.CONDUCTIVITY_YLIM_RE if ylim_re is None else ylim_re
    ylim_im = config.CONDUCTIVITY_YLIM_IM if ylim_im is None else ylim_im
    save_plot = config.SAVE_PLOTS if save_plot is None else save_plot

    data = pa.load_postprocessed_data(timestamp, directory_job_index, config.RUNNING_MACHINE)
    missing = [key for key in ('omega', 'current_fft', 'vpot_fft', 'times', 'gap_evolution') if key not in data]
    if missing:
        raise ValueError(f"Timestamp {timestamp}, file {directory_job_index} lacks {missing}; postprocess with compute_keywords=['state_evolution'] "
                         f"(or 'spectral_analysis'), keys present: {sorted(data.keys())}")

    jobs = list(range(len(data['gap_evolution']))) if job_indices is None else [int(j) for j in job_indices]

    results = {}
    for job in jobs:
        max_a = np.max(np.abs(data['vector_potential_evolution'][job]))
        if max_a < 1e-10:
            print(f"Job {job}: vector potential is zero (max |A| = {max_a:.3e}), the conductivity needs a driven run")
            results[job] = None
            continue
        results[job] = plot_conductivity_job(timestamp, data, directory_job_index, job, threshold, sign, xlim, ylim_re, ylim_im, external,
                                             save_plot, save_dir)
    return results


if __name__ == '__main__':
    plot_conductivity()
