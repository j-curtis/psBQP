"""
Real-Time Evolution Test Script

This script loads an equilibrium initial state and performs real-time evolution
for 25 time steps, then plots the initial and final diagonal matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Import our classes
from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution


def plot_nambu_components(nambu_tensor, time_grid, title="Nambu-Keldysh Tensor Components",
                          vmin=None, vmax=None, cmap='RdBu_r',
                          t_lim=None, tprime_lim=None):
    """
    Plot all 4 Pauli components of a NambuKeldyshTensor in a 4x2 grid.

    Args:
        nambu_tensor: NambuKeldyshTensor with shape (2, 2, Nt, Nt)
                     component[i, j] represents g(t_i, t'_j)
        time_grid: Array of time values corresponding to the tensor indices
        title: Overall title for the figure
        vmin, vmax: Optional colorbar limits (if None, auto-scale per component)
        cmap: Colormap to use (default 'RdBu_r' for diverging red-blue)
        t_lim: Optional tuple (t_min, t_max) to zoom into specific t range
        tprime_lim: Optional tuple (tprime_min, tprime_max) to zoom into specific t' range

    Layout:
        Top row: Real parts of [I, X, Y, Z] components
        Bottom row: Imaginary parts of [I, X, Y, Z] components

    Convention:
        After transpose, x-axis = t (first index), y-axis = t' (second index)
        Smallest times in top-left corner
    """
    pauli_labels = ['I (Identity)', 'X (σ_x)', 'Y (σ_y)', 'Z (σ_z)']

    # Create figure with 4 columns (Pauli components) x 2 rows (real/imag)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Time extent for imshow (smallest times in top-left corner)
    # After transpose: x-axis is t, y-axis is t'
    t_min, t_max = time_grid[0], time_grid[-1]
    extent = [t_min, t_max, t_max, t_min]  # [left, right, bottom, top] with origin='upper'

    # Extract each Pauli component and plot
    for pauli_idx in range(4):
        # Extract Pauli component using trace (divide by 2 for normalization)
        component = nambu_tensor.trace(pauli_index=pauli_idx) / 2

        # Transpose: component[i,j] -> component.T[j,i]
        # So x-axis (columns) = i = t, y-axis (rows) = j = t'

        # Real part (top row)
        ax_real = axes[0, pauli_idx]
        im_real = ax_real.imshow(np.real(component), aspect='auto', origin='upper',
                                 extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_real.set_title(f"{pauli_labels[pauli_idx]} - Real", fontsize=10)
        ax_real.set_ylabel("t", fontsize=9)
        if pauli_idx == 0:
            ax_real.set_ylabel("t", fontsize=11, fontweight='bold')

        # Apply axis limits if specified
        if t_lim is not None:
            ax_real.set_xlim(t_lim)
        if tprime_lim is not None:
            ax_real.set_ylim(tprime_lim[1], tprime_lim[0])  # Reversed for origin='upper'

        plt.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04)

        # Imaginary part (bottom row)
        ax_imag = axes[1, pauli_idx]
        im_imag = ax_imag.imshow(np.imag(component), aspect='auto', origin='upper',
                                 extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_imag.set_title(f"{pauli_labels[pauli_idx]} - Imag", fontsize=10)
        ax_imag.set_xlabel("t'", fontsize=11, fontweight='bold')
        ax_imag.set_ylabel("t", fontsize=9)
        if pauli_idx == 0:
            ax_imag.set_ylabel("t", fontsize=11, fontweight='bold')

        # Apply axis limits if specified
        if t_lim is not None:
            ax_imag.set_xlim(t_lim)
        if tprime_lim is not None:
            ax_imag.set_ylim(tprime_lim[1], tprime_lim[0])  # Reversed for origin='upper'

        plt.colorbar(im_imag, ax=ax_imag, fraction=0.046, pad=0.04)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.show()


def load_state(filename):
    """
    Load a StateObject from a file.

    Args:
        filename: Path to saved state file

    Returns:
        StateObject: Loaded state object
    """
    # Add .pkl extension if not present
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'

    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Load using pickle
    with open(filename, 'rb') as f:
        state = pickle.load(f)

    print(f"State loaded from: {os.path.abspath(filename)}")
    print(f"  g^R shape: {state.gr.data.shape}")
    print(f"  g^K shape: {state.gk.data.shape}")

    # Extract gap if available
    try:
        gap_history = state.get_gap_history()
        print(f"  Gap at final time: {gap_history[-1]:.4f}")
    except:
        print("  Could not extract gap history")

    return state


def main():
    print("="*70)
    print("Real-Time Evolution Test")
    print("="*70)
    print()

    # ========== Define Parameters ==========
    grid_parameters = {
        'time_sampling': 1501,
        'time_duration': 2 * np.pi * 5,
        'eta': 0.2
    }

    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': 0.3,
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
    print(f"  BCS coupling: {evolution._get_BCS_coupling():.4f}")
    print()

    # ========== Load Initial State ==========
    print("Loading initial equilibrium state...")
    try:
        initial_state = load_state('initial_state_test.pkl')
    except FileNotFoundError:
        print("Initial state file not found. Generating new equilibrium state...")
        initial_state, _, _ = evolution.generate_initial_state(Q=0.0)
        print(f"Initial state generated")
        print(f"  g^R shape: {initial_state.gr.data.shape}")
        print(f"  g^K shape: {initial_state.gk.data.shape}")
    print()

    # ========== Perform Real-Time Evolution ==========
    print()
    print("="*70)
    print("Performing 200 timesteps and saving...")
    print("="*70)

    # Evolve from initial state for 200 steps
    simulated_state, gaps, currents = evolution.real_time_evolution(
        initial_state, num_timesteps=1600)
    print("Evolution complete!")
    print(f"  Final gap after 200 steps: {gaps[-1]:.6f}")

    # Save the simulated state
    with open('simulated_state.pkl', 'wb') as f:
        pickle.dump(simulated_state, f)

    abs_path = os.path.abspath('simulated_state.pkl')
    print(f"\nSimulated state saved to: {abs_path}")
    print(f"  g^R shape: {simulated_state.gr.data.shape}")
    print(f"  g^K shape: {simulated_state.gk.data.shape}")

    print()
    print("="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
