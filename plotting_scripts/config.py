"""
Central plotting configuration: timestamps, input parameters, save folder, fonts, line styles and colors.
"""

import os
import matplotlib.pyplot as plt

#* ---------------------------------------------------------------------------------------------
#* general input parameters
#* ---------------------------------------------------------------------------------------------
RUNNING_MACHINE = 'laptop'
SAVE_PLOTS = True

#* ---------------------------------------------------------------------------------------------
#* plot_static_analysis.py (each plot file has its own timestamp; None as a function argument falls back to these)
#* ---------------------------------------------------------------------------------------------
STATIC_ANALYSIS_TIMESTAMP = '1792145428'
STATIC_ANALYSIS_DETAILED = True
STATIC_ANALYSIS_EQUILIBRIUM_CURRENT_SIGN = -1   # equilibrium current is plotted as -J
STATIC_ANALYSIS_INCLUDE_CURRENT = True         # 2x1 plot: gap on top, current below
STATIC_ANALYSIS_DATA_FILE = 'equilibrium_results/equilibrium_sweep_20261016_104342.pkl'               # equilibrium_sweep.py pickle overlaid as reference (relative to psBQP-keldysh/), None = no overlay
STATIC_ANALYSIS_DIRECTORY_JOB_INDICES = None    # result files of the timestamp (None = all)
STATIC_ANALYSIS_JOB_INDICES = [0]               # inner jobs for the detailed plots (None = all)
STATIC_ANALYSIS_ERROR_TYPE = 'range'            # 'range' (min/max of |gap| over last 50 steps) or 'std'
STATIC_ANALYSIS_ROWS_SOURCE = 'auto'            # 'auto', 'rows' or 'raw'
STATIC_ANALYSIS_ROWS_YSCALE = 'linear'          # 'linear' or 'symlog'

#* ---------------------------------------------------------------------------------------------
#* plot_state_evolution.py
#* ---------------------------------------------------------------------------------------------
STATE_EVOLUTION_TIMESTAMP = '1792241185'
STATE_EVOLUTION_DIRECTORY_JOB_INDEX = 0        # result file (postprocessing job) of the timestamp
STATE_EVOLUTION_JOB_INDICES = [4]            # inner jobs to plot (None = all)
STATE_EVOLUTION_DETAILED_CHECK = True         # g^R / g^K normalization versus time (needs a run with track_state=True)
STATE_EVOLUTION_GAP = 'absolute'               # 'absolute' |Delta|/T_c, 'deviation' (|Delta(t)|-|Delta(0)|)/T_c, 'relative' (|Delta(t)|-|Delta(0)|)/|Delta(0)|
STATE_EVOLUTION_XLIM = None                    # time axis limits, e.g. (0, 30)
STATE_EVOLUTION_ROWS_YSCALE = 'linear'         # 'linear' or 'symlog' for the normalization plots
STATE_EVOLUTION_INCLUDE_FFT = True             # second figure: |FFT| of gap, current and vector potential versus frequency
STATE_EVOLUTION_FFT_XLIM = (-10, 10)           # frequency axis limits in omega/T_c
STATE_EVOLUTION_FFT_YSCALE = 'linear'          # 'linear' or 'log'
STATE_EVOLUTION_FFT_MIN_GAP = 0.01             # |Delta(0)|/T_c below this: no Delta / 2 Delta lines in the FFT plot
STATE_EVOLUTION_ENERGY_XLIM = (-10, 10)        # energy axis limits (eps/T_c) of the energy-time representation plots (detailed check, needs track_every_n)
STATE_EVOLUTION_ENERGY_N_SLICES = 3            # number of times (first ... last tracked step) drawn in each energy plot

#* ---------------------------------------------------------------------------------------------
#* plot_transmission.py (reads a 'transmitivity' postprocessed timestamp of current pulse runs)
#* ---------------------------------------------------------------------------------------------
TRANSMISSION_TIMESTAMP = '1792241185'
TRANSMISSION_DIRECTORY_JOB_INDEX = 0           # result file (postprocessing job) used for the time traces and the detailed checks
TRANSMISSION_DIRECTORY_JOB_INDICES = None      # result files overlaid in the transmitivity plot (None = all)
TRANSMISSION_JOB_INDICES = None                # inner jobs for the time traces and detailed checks (None = all)
TRANSMISSION_DETAILED_CHECK = True             # g^R / g^K normalization versus time and energy-time distributions (need track_state / track_every_n)
TRANSMISSION_GAP = 'absolute'                  # 'absolute', 'deviation' or 'relative' (see STATE_EVOLUTION_GAP)
TRANSMISSION_XLIM = None                       # time axis limits of the traces
TRANSMISSION_INPUT_TIME_SHIFT = -1             # I_in is plotted at times + SHIFT * dt (gap, J, A are saved at t_(n+1), input_pulse[n] belongs to t_n)
TRANSMISSION_NORMAL_REFERENCE = True           # dashed line 2 Z_T / (2 Z_T + R_n + R_c) in the transmitivity plot (needs circuit_params)
TRANSMISSION_ROWS_YSCALE = 'linear'            # 'linear' or 'symlog' for the normalization plots
TRANSMISSION_ENERGY_XLIM = (-10, 10)           # energy axis limits (eps/T_c) of the energy-time plots
TRANSMISSION_ENERGY_N_SLICES = 3               # number of times drawn in each energy plot

#* ---------------------------------------------------------------------------------------------
#* plot_conductivity.py (reads the Fourier transforms of a 'state_evolution' / 'spectral_analysis' postprocessed timestamp)
#* ---------------------------------------------------------------------------------------------
CONDUCTIVITY_TIMESTAMP = '1792228571'
CONDUCTIVITY_DIRECTORY_JOB_INDEX = 0           # result file (postprocessing job) of the timestamp
CONDUCTIVITY_JOB_INDICES = [0]                 # inner jobs, one 2x2 figure each (None = all)
CONDUCTIVITY_SIGN = -1                         # sigma = SIGN * J(omega) / (i omega A(omega)); -1 as in data_analysis_sc.extract_conductivity
#~ CONDUCTIVITY_THRESHOLD = 1e-6                  # minimum |A(omega)| (raw FFT, without the dt factor) and |omega| for a valid point
CONDUCTIVITY_THRESHOLD = 1e-3                  # sigma only where |A(omega)| > threshold * max|A(omega)| (relative to the peak of the pulse spectrum)
CONDUCTIVITY_XLIM = None                       # frequency axis limits (None: valid range plus 10% margin)
CONDUCTIVITY_YLIM_RE = None
CONDUCTIVITY_YLIM_IM = None

#* ---------------------------------------------------------------------------------------------
#* save location (relative paths are taken relative to this file)
#* ---------------------------------------------------------------------------------------------
SAVE_DIR = 'plotting_results'
SAVE_FORMAT = 'pdf'                 # 'pdf', 'svg' or 'png'

#* ---------------------------------------------------------------------------------------------
#* fonts
#* ---------------------------------------------------------------------------------------------
USE_LATEX = True
FONT_FAMILY = 'serif'
FONT_SERIF = ['Computer Modern Roman']
LATEX_PREAMBLE = r'\usepackage{amsmath} \usepackage{amssymb}'

FONT_SIZE = 20
TITLE_SIZE = 21
SUPTITLE_SIZE = 20
LABEL_SIZE = 21
TICK_SIZE = 18
LEGEND_SIZE = 15

#* ---------------------------------------------------------------------------------------------
#* lines, markers, grid
#* ---------------------------------------------------------------------------------------------
LINE_WIDTH = 2.5
MARKER = 'o'
MARKER_SIZE = 7
LINE_ALPHA = 0.9
CAPSIZE = 4
ERRORBAR_LINE_WIDTH = 1.8
GRID_ALPHA = 0.3
LEGEND_FRAME_ALPHA = 0.9
SHARED_LEGEND_NCOL = 4               # columns of the shared legend below multi-panel figures

#* ---------------------------------------------------------------------------------------------
#* colors: COLOR_1 / COLOR_2 are used for two-line plots (real / imaginary part)
#* ---------------------------------------------------------------------------------------------
COLOR_1 = 'blue'
COLOR_2 = 'red'
COLORMAP = 'Blues'                  # curves of several result files
COLORMAP_RANGE = (0.25, 1.0)        # used part of the colormap (lightest 25% cut)

#* ---------------------------------------------------------------------------------------------
#* figure sizes
#* ---------------------------------------------------------------------------------------------
FIGSIZE_SINGLE = (9, 6)
FIGSIZE_STACKED = (9, 9)
FIGSIZE_GRID_2X2 = (14, 10)
FIGSIZE_ROW_1X3 = (21, 6)
FIGSIZE_ROW_1X2 = (14, 6)


def apply_style():
    """Set the matplotlib rcParams from this file (call after importing modules that set their own rcParams)."""
    plt.rcParams['text.usetex'] = USE_LATEX
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.serif'] = FONT_SERIF
    plt.rcParams['text.latex.preamble'] = LATEX_PREAMBLE
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE
    plt.rcParams['legend.fontsize'] = LEGEND_SIZE
    plt.rcParams['lines.linewidth'] = LINE_WIDTH
    plt.rcParams['lines.markersize'] = MARKER_SIZE


def resolve_save_dir(save_dir=None):
    """Absolute save folder (config.SAVE_DIR relative to this file by default); created if missing."""
    folder = SAVE_DIR if save_dir is None else save_dir
    if not os.path.isabs(folder):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
    os.makedirs(folder, exist_ok=True)
    return folder


def resolve_data_file(data_file):
    """Absolute path of an external data pickle (relative paths are taken relative to the psBQP-keldysh folder)."""
    if os.path.isabs(data_file):
        return data_file
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_file)
