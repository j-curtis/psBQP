"""
Plot Real-Time Evolution Results

This script loads a saved state from real-time evolution and visualizes:
- Gap evolution over time (real, imaginary, absolute value)
- Current evolution over time (placeholder)
- Full Green's function matrices (debug mode)
- FDT verification for last row (debug mode)

Usage:
    python plot_real_time.py                    # Display gap plot
    python plot_real_time.py --debug            # Display all plots
    python plot_real_time.py --save             # Save gap plot to file
    python plot_real_time.py --debug --save     # Save all plots to files
    python plot_real_time.py --state <file>     # Specify state file
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our classes
from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution


def plot_nambu_components(nambu_tensor, time_grid, title="Nambu-Keldysh Tensor Components",
                          vmin=None, vmax=None, cmap='RdBu_r',
                          t_lim=None, tprime_lim=None, save_filename=None):
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
        save_filename: Optional filename to save the plot

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

    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_filename}")
    else:
        plt.show()

    plt.close()


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


def plot_gap_evolution(state, time_grid, save_filename=None):
    """
    Plot gap evolution over time.

    Shows real part (blue), imaginary part (red), and absolute value (black).

    Args:
        state: StateObject with gap history
        time_grid: Array of time values
        save_filename: Optional filename to save the plot
    """
    # Extract gap history
    gap_history = state.get_gap_history()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot real, imaginary, and absolute value
    ax.plot(time_grid, np.real(gap_history), 'b-', linewidth=2, label='Re[Δ(t)]', alpha=0.8)
    ax.plot(time_grid, np.imag(gap_history), 'r-', linewidth=2, label='Im[Δ(t)]', alpha=0.8)
    ax.plot(time_grid, np.abs(gap_history), 'k-', linewidth=2, label='|Δ(t)|', alpha=0.8)

    ax.set_xlabel('Time t', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap Δ(t)', fontsize=12, fontweight='bold')
    ax.set_title('Gap Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_grid[0], time_grid[-1]])

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_filename}")
    else:
        plt.show()

    plt.close()


def plot_current_evolution(state, time_grid, save_filename=None):
    """
    Plot current evolution over time (placeholder).

    Args:
        state: StateObject with current history
        time_grid: Array of time values
        save_filename: Optional filename to save the plot
    """
    # Placeholder for current evolution
    # Current implementation returns zeros from state.get_current_history()

    print("Current evolution plotting not yet implemented (placeholder).")
    print("  state.get_current_history() needs implementation.")

    # Create placeholder figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot zeros as placeholder
    current_placeholder = np.zeros(len(time_grid))
    ax.plot(time_grid, current_placeholder, 'g-', linewidth=2, label='Current (placeholder)', alpha=0.8)

    ax.set_xlabel('Time t', fontsize=12, fontweight='bold')
    ax.set_ylabel('Current J(t)', fontsize=12, fontweight='bold')
    ax.set_title('Current Evolution (Placeholder)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_grid[0], time_grid[-1]])

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_filename}")
    else:
        plt.show()

    plt.close()


def plot_fdt_comparison(state, thermal_dist, thermal_integral, time_grid, save_filename=None):
    """
    Plot FDT comparison for the last row of evolution.

    Verifies g^K = g^R @ f - f @ g^A using precise convolution.

    Args:
        state: StateObject with g^R and g^K
        thermal_dist: Thermal distribution f(t,t')
        thermal_integral: Thermal integral F(t,t')
        time_grid: Array of time values
        save_filename: Optional filename to save the plot
    """
    print("\nChecking FDT relation for last row...")

    # Use StateObject's check_fdt method
    gk_fdt_row, gk_actual_row, error_row, max_error = state.check_fdt(
        thermal_dist,
        thermal_integral,
        time_index=-1
    )

    print(f"  Maximum absolute error: {max_error:.6e}")

    # Create figure with 2x2 grid for each Pauli component
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pauli_names = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']

    for pauli_idx, (ax, pauli_name) in enumerate(zip(axes.flat, pauli_names)):
        # Extract Pauli components using trace
        gk_actual_pauli = gk_actual_row.trace(pauli_idx) / 2
        gk_fdt_pauli = gk_fdt_row.trace(pauli_idx) / 2

        # Plot real and imaginary parts
        ax.plot(time_grid, np.real(gk_actual_pauli[0, :]), 'b-', linewidth=2,
                label='g^K actual (Re)', alpha=0.7)
        ax.plot(time_grid, np.real(gk_fdt_pauli[0, :]), 'r--', linewidth=1.5,
                label='FDT prediction (Re)', alpha=0.7)
        ax.plot(time_grid, np.imag(gk_actual_pauli[0, :]), 'c-', linewidth=2,
                label='g^K actual (Im)', alpha=0.7)
        ax.plot(time_grid, np.imag(gk_fdt_pauli[0, :]), 'm--', linewidth=1.5,
                label='FDT prediction (Im)', alpha=0.7)

        ax.set_xlabel("t' (time)", fontsize=10)
        ax.set_ylabel(f'g^K(t_max, t\') - {pauli_name}', fontsize=10)
        ax.set_title(f'FDT Check: {pauli_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_grid[0], 0])

    plt.suptitle('FDT Verification: Last Row', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_filename}")
    else:
        plt.show()

    plt.close()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot real-time evolution results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (plot full matrices and FDT check)')
    parser.add_argument('--state', type=str, default='simulated_state.pkl',
                        help='Path to saved state file (default: simulated_state.pkl)')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files instead of displaying them')

    args = parser.parse_args()

    print("="*70)
    print("Plot Real-Time Evolution Results")
    print("="*70)
    print()

    # Load state
    print(f"Loading state from: {args.state}")
    try:
        state = load_state(args.state)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please provide a valid state file using --state <filename>")
        return
    print()

    # Infer grid parameters from state
    N_t = state.gr.data.shape[2]
    T_max = state.T_max if hasattr(state, 'T_max') else 10.0
    time_grid = np.linspace(-T_max, 0, N_t)

    print(f"Grid information:")
    print(f"  Time points: {N_t}")
    print(f"  Time range: [{time_grid[0]:.4f}, {time_grid[-1]:.4f}]")
    print()

    # ========== Plot Gap Evolution (Always) ==========
    print("="*70)
    print("Plotting Gap Evolution")
    print("="*70)
    plot_gap_evolution(state, time_grid,
                       save_filename='gap_evolution.png' if args.save else None)
    print()

    # ========== Plot Current Evolution (Placeholder) ==========
    # Uncomment if current evolution is implemented
    # print("="*70)
    # print("Plotting Current Evolution")
    # print("="*70)
    # plot_current_evolution(state, time_grid,
    #                        save_filename='current_evolution.png' if args.save else None)
    # print()

    # ========== Debug Mode ==========
    if args.debug:
        print("="*70)
        print("Debug Mode: Plotting Full Matrices")
        print("="*70)
        print()

        # Plot g^R matrix
        print("Plotting g^R (retarded Green's function)...")
        plot_nambu_components(state.gr, time_grid,
                              title="Retarded Green's Function g^R(t,t')",
                              save_filename='gr_full_matrix.png' if args.save else None)
        print()

        # Plot g^K matrix
        print("Plotting g^K (Keldysh Green's function)...")
        plot_nambu_components(state.gk, time_grid,
                              title="Keldysh Green's Function g^K(t,t')",
                              save_filename='gk_full_matrix.png' if args.save else None)
        print()

        # Plot FDT comparison
        print("="*70)
        print("Debug Mode: FDT Verification")
        print("="*70)

        # Generate thermal distribution (need to recreate evolution object)
        # Infer system parameters from state
        grid_parameters = {
            'time_sampling': N_t,
            'time_duration': T_max,
            'eta': 0.2  # Default value
        }

        system_parameters = {
            'critical_temperature': 1.0,  # Default value
            'temperature': 0.3,  # Default value
            'eta': 0.2  # Default value
        }

        print("\nCreating evolution object for thermal distribution...")
        print("  (Using default parameters: T_c=1.0, T=0.3, η=0.2)")
        evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)

        # Generate thermal distributions
        evolution.get_thermal_occupation(system_parameters['temperature'])
        evolution.get_thermal_integral(system_parameters['temperature'])

        thermal_dist = evolution.thermal_dist
        thermal_integral = evolution.thermal_integral

        # Plot FDT comparison
        plot_fdt_comparison(state, thermal_dist, thermal_integral, time_grid,
                            save_filename='fdt_verification_last_row.png' if args.save else None)
        print()

    print("="*70)
    print("Plotting Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
