"""
General comparison utilities for NambuKeldyshTensor objects.

This module provides functions to:
- Load StateObject from pickle files with auto-generation fallback
- Compare two NambuKeldyshTensor objects (row-wise)
- Plot comparisons with deviation markers

All plots are saved to Test_plots/ directory, never displayed interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nambu_keldysh_class import NambuKeldyshTensor
from state_object_class import StateObject
from usadel_keldysh_evolution import UsadelKeldyshEvolution



def compare_tensor_rows(tensor1, tensor2, x_values=None, row_index=-1):
    """
    Compare last rows of two NambuKeldyshTensor objects across all Pauli components.

    Extracts specified rows and computes differences for all 4 Pauli components,
    separately for real and imaginary parts.

    Args:
        tensor1: First NambuKeldyshTensor to compare
        tensor2: Second NambuKeldyshTensor to compare
        x_values: Optional array of x-values (for indexing max difference location)
        row_index: Row index to extract (default -1 for last row)

    Returns:
        dict: Comparison statistics with structure:
              {
                  'pauli_0': {'real': {...}, 'imag': {...}},
                  'pauli_1': {'real': {...}, 'imag': {...}},
                  'pauli_2': {'real': {...}, 'imag': {...}},
                  'pauli_3': {'real': {...}, 'imag': {...}}
              }
              where each {...} contains:
                  - max_diff: Maximum absolute difference
                  - max_diff_index: Index of maximum difference
                  - mean_diff: Mean absolute difference
                  - rms_diff: RMS difference
    """
    # Extract rows
    row1 = tensor1[row_index, :]
    row2 = tensor2[row_index, :]

    # Initialize results dictionary
    results = {}

    # Pauli component names
    pauli_names = ['pauli_0', 'pauli_1', 'pauli_2', 'pauli_3']

    # Compare each Pauli component
    for pauli_idx in range(4):
        # Extract Pauli component using trace (divide by 2 for normalization)
        comp1 = row1.trace(pauli_index=pauli_idx) / 2
        comp2 = row2.trace(pauli_index=pauli_idx) / 2

        # Handle shape: could be (1, Nt) or (Nt,)
        if comp1.ndim == 2:
            comp1 = comp1[0, :]
            comp2 = comp2[0, :]

        # Compute statistics for real part
        real_diff = np.abs(np.real(comp1) - np.real(comp2))
        max_diff_idx_real = np.argmax(real_diff)

        real_stats = {
            'max_diff': real_diff[max_diff_idx_real],
            'max_diff_index': max_diff_idx_real,
            'mean_diff': np.mean(real_diff),
            'rms_diff': np.sqrt(np.mean(real_diff**2))
        }

        # Compute statistics for imaginary part
        imag_diff = np.abs(np.imag(comp1) - np.imag(comp2))
        max_diff_idx_imag = np.argmax(imag_diff)

        imag_stats = {
            'max_diff': imag_diff[max_diff_idx_imag],
            'max_diff_index': max_diff_idx_imag,
            'mean_diff': np.mean(imag_diff),
            'rms_diff': np.sqrt(np.mean(imag_diff**2))
        }

        # Store results for this Pauli component
        results[pauli_names[pauli_idx]] = {
            'real': real_stats,
            'imag': imag_stats
        }

    return results


def plot_tensor_comparison(tensor1, tensor2, x_values, filename, title="Tensor Comparison", row_index=-1, save_dir='Test_plots'):
    """
    Plot comparison of two NambuKeldyshTensor objects with deviation markers.

    Creates a 2x2 grid showing all 4 Pauli components. Each subplot shows:
    - tensor1: blue solid (real), blue dashed (imag)
    - tensor2: red solid (real), red dashed (imag)
    - Vertical gray dotted line at maximum deviation point

    Saves plot to Test_plots/{filename}.png (no interactive display).

    Args:
        tensor1: First NambuKeldyshTensor to plot
        tensor2: Second NambuKeldyshTensor to plot
        x_values: Array of x-values for plotting
        filename: Output filename (without extension)
        title: Plot title
        row_index: Row index to extract (default -1 for last row)
        save_dir: Directory to save plots (default 'Test_plots')
    """
    # Extract rows
    row1 = tensor1[row_index, :]
    row2 = tensor2[row_index, :]

    # Pauli labels
    pauli_labels = ['τ₀ (I)', 'τ₁ (X)', 'τ₂ (Y)', 'τ₃ (Z)']

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pauli_idx in range(4):
        # Extract Pauli component
        comp1 = row1.trace(pauli_index=pauli_idx) / 2
        comp2 = row2.trace(pauli_index=pauli_idx) / 2

        # Handle shape
        if comp1.ndim == 2:
            comp1 = comp1[0, :]
            comp2 = comp2[0, :]

        # Get subplot
        ax = axes.flat[pauli_idx]

        # Plot tensor1 (blue)
        ax.plot(x_values, np.real(comp1), 'b-', linewidth=2, label='Tensor 1 (Real)', alpha=0.8)
        ax.plot(x_values, np.imag(comp1), 'b--', linewidth=2, label='Tensor 1 (Imag)', alpha=0.8)

        # Plot tensor2 (red)
        ax.plot(x_values, np.real(comp2), 'r-', linewidth=2, label='Tensor 2 (Real)', alpha=0.8)
        ax.plot(x_values, np.imag(comp2), 'r--', linewidth=2, label='Tensor 2 (Imag)', alpha=0.8)

        # Find maximum deviation point
        real_diff = np.abs(np.real(comp1) - np.real(comp2))
        imag_diff = np.abs(np.imag(comp1) - np.imag(comp2))
        max_real_idx = np.argmax(real_diff)
        max_imag_idx = np.argmax(imag_diff)

        # Use overall max deviation
        max_idx = max_real_idx if real_diff[max_real_idx] > imag_diff[max_imag_idx] else max_imag_idx
        max_x = x_values[max_idx]

        # Plot vertical line at max deviation
        ax.axvline(max_x, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Max deviation')

        # Formatting
        ax.set_xlabel('Time t\'', fontsize=10)
        ax.set_ylabel(f'{pauli_labels[pauli_idx]}', fontsize=10)
        ax.set_title(f'{pauli_labels[pauli_idx]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([x_values[0], x_values[-1]])

    plt.tight_layout()
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.97)

    # Save plot (no interactive display)
    output_path = os.path.join(save_dir, f'{filename}.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
