"""
Test script to verify thermal sum midpoint rule at boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.append('/home/filip/Documents/Research/BiPolaron/demler_tools')

from usadel_keldysh_evolution import UsadelKeldyshEvolution

# Create minimal grid parameters
grid_parameters = {
    'time_sampling': 751,
    'time_duration': 2 * np.pi * 5,
    'omega_sampling': 201,
    'cutoff': 20.0,
    'eta': 0.1
}

system_parameters = {
    'critical_temperature': 1.0,
    'temperature': 0.5,
    'eta': 0.1
}

# Create evolution object
evolution = UsadelKeldyshEvolution(
    grid_parameters=grid_parameters,
    system_parameters=system_parameters
)

print("="*70)
print("TESTING THERMAL SUM: ANALYTIC VS NUMERICAL COMPARISON")
print("="*70)

# Get thermal integral and thermal sum
T = 0.5
evolution.get_thermal_integral(T)
evolution.get_thermal_sum(T)

# Extract thermal_sum and thermal_integral from evolution object
# thermal_sum_left and thermal_sum_right have shape (2, 2, 2, N_t)
# thermal_integral has shape (2, 2, N_t, N_t)

# Extract the two time rows from thermal_sum
thermal_sum_left_minus = evolution.thermal_sum_left.data[0, 0, -2, :]  # [-2,:] row
thermal_sum_left = evolution.thermal_sum_left.data[0, 0, -1, :]  # [-1,:] row

thermal_sum_right_minus = evolution.thermal_sum_right.data[0, 0, -2, :]  # [-2,:] row
thermal_sum_right = evolution.thermal_sum_right.data[0, 0, -1, :]  # [-1,:] row

# Extract thermal_integral elements for comparison
# thermal_integral is the two-time function F(t, t')
thermal_integral_last_row = evolution.thermal_integral.data[0, 0, -1, :]  # [-1,:] for right sum
thermal_integral_second_last_row = evolution.thermal_integral.data[0, 0, -2, :]  # [-2,:] for right sum
thermal_integral_last_col = evolution.thermal_integral.data[0, 0, :, -1]  # [:,-1] for left sum
thermal_integral_second_last_col = evolution.thermal_integral.data[0, 0, :, -2]  # [:,-2] for left sum

# Debug: check shapes
print(f"\nDebug - Shapes:")
print(f"  thermal_sum_left.shape: {thermal_sum_left.shape}")
print(f"  thermal_integral_last_col.shape: {thermal_integral_last_col.shape}")
print(f"  thermal_sum_right.shape: {thermal_sum_right.shape}")
print(f"  thermal_integral_last_row.shape: {thermal_integral_last_row.shape}")

# Debug: check a few values
print(f"\nDebug - Sample values at t'=-1:")
print(f"  thermal_sum_left[-1] = {thermal_sum_left[-1]}")
print(f"  thermal_integral_last_col[-1] = {thermal_integral_last_col[-1]}")
print(f"  |difference| = {np.abs(thermal_sum_left[-1] - thermal_integral_last_col[-1])}")

# Compute errors as difference of absolute values
error_left_vs_integral = np.abs(thermal_sum_left) - np.abs(thermal_integral_last_col)
error_left_minus_vs_integral = np.abs(thermal_sum_left_minus) - np.abs(thermal_integral_second_last_col)
error_right_vs_integral = np.abs(thermal_sum_right) - np.abs(thermal_integral_last_row)
error_right_minus_vs_integral = np.abs(thermal_sum_right_minus) - np.abs(thermal_integral_second_last_row)

print(f"\nError statistics:")
print(f"  thermal_sum_left[-1,:] vs thermal_integral[:,-1]:")
print(f"    Max abs error: {np.max(np.abs(error_left_vs_integral)):.6e}")
print(f"    Mean abs error: {np.mean(np.abs(error_left_vs_integral)):.6e}")

print(f"\n  thermal_sum_left[-2,:] vs thermal_integral[:,-2]:")
print(f"    Max abs error: {np.max(np.abs(error_left_minus_vs_integral)):.6e}")
print(f"    Mean abs error: {np.mean(np.abs(error_left_minus_vs_integral)):.6e}")

print(f"\n  thermal_sum_right[-1,:] vs thermal_integral[-1,:]:")
print(f"    Max abs error: {np.max(np.abs(error_right_vs_integral)):.6e}")
print(f"    Mean abs error: {np.mean(np.abs(error_right_vs_integral)):.6e}")

print(f"\n  thermal_sum_right[-2,:] vs thermal_integral[-2,:]:")
print(f"    Max abs error: {np.max(np.abs(error_right_minus_vs_integral)):.6e}")
print(f"    Mean abs error: {np.mean(np.abs(error_right_minus_vs_integral)):.6e}")

# ============================================================================
# PLOTTING
# ============================================================================

# Create output directory if it doesn't exist
os.makedirs('Test_plots', exist_ok=True)

fig = plt.figure(figsize=(16, 16))

# Row 1: Left sum [-1,:] vs thermal_integral[:,-1]
ax1 = plt.subplot(4, 2, 1)
ax1.plot(evolution.time_grid, np.abs(thermal_sum_left), 'b-', linewidth=2, label='thermal_sum_left[-1,:]', alpha=0.7, marker='o', markevery=10, markersize=4)
ax1.plot(evolution.time_grid, np.abs(thermal_integral_last_col), 'r--', linewidth=2, label='thermal_integral[:,-1]', alpha=0.7, marker='^', markevery=10, markersize=4)
ax1.set_xlabel('t\' (time index)')
ax1.set_ylabel('Absolute value')
ax1.set_title('thermal_sum_left[-1,:] vs thermal_integral[:,-1]')
ax1.set_xlim(-5, 0)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(4, 2, 2)
ax2.semilogy(evolution.time_grid, np.abs(error_left_vs_integral) + 1e-20, 'k-', linewidth=2, alpha=0.7, marker='s', markevery=10, markersize=4)
ax2.set_xlabel('t\' (time index)')
ax2.set_ylabel('|Error|')
ax2.set_title('Error: |thermal_sum_left[-1,:] - thermal_integral[:,-1]|')
ax2.set_xlim(-5, 0)
ax2.set_ylim(1e-4, 1)
ax2.grid(True, alpha=0.3)

# Row 2: Left sum [-2,:] vs thermal_integral[:,-2]
ax3 = plt.subplot(4, 2, 3)
ax3.plot(evolution.time_grid, np.abs(thermal_sum_left_minus), 'b-', linewidth=2, label='thermal_sum_left[-2,:]', alpha=0.7, marker='o', markevery=10, markersize=4)
ax3.plot(evolution.time_grid, np.abs(thermal_integral_second_last_col), 'r--', linewidth=2, label='thermal_integral[:,-2]', alpha=0.7, marker='^', markevery=10, markersize=4)
ax3.set_xlabel('t\' (time index)')
ax3.set_ylabel('Absolute value')
ax3.set_title('thermal_sum_left[-2,:] vs thermal_integral[:,-2]')
ax3.set_xlim(-5, 0)
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(4, 2, 4)
ax4.semilogy(evolution.time_grid, np.abs(error_left_minus_vs_integral) + 1e-20, 'k-', linewidth=2, alpha=0.7, marker='s', markevery=10, markersize=4)
ax4.set_xlabel('t\' (time index)')
ax4.set_ylabel('|Error|')
ax4.set_title('Error: |thermal_sum_left[-2,:] - thermal_integral[:,-2]|')
ax4.set_xlim(-5, 0)
ax4.set_ylim(1e-4, 1)
ax4.grid(True, alpha=0.3)

# Row 3: Right sum [-1,:] vs thermal_integral[-1,:]
ax5 = plt.subplot(4, 2, 5)
ax5.plot(evolution.time_grid, np.abs(thermal_sum_right), 'b-', linewidth=2, label='thermal_sum_right[-1,:]', alpha=0.7, marker='o', markevery=10, markersize=4)
ax5.plot(evolution.time_grid, np.abs(thermal_integral_last_row), 'r--', linewidth=2, label='thermal_integral[-1,:]', alpha=0.7, marker='^', markevery=10, markersize=4)
ax5.set_xlabel('t\' (time index)')
ax5.set_ylabel('Absolute value')
ax5.set_title('thermal_sum_right[-1,:] vs thermal_integral[-1,:]')
ax5.set_xlim(-5, 0)
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(4, 2, 6)
ax6.semilogy(evolution.time_grid, np.abs(error_right_vs_integral) + 1e-20, 'k-', linewidth=2, alpha=0.7, marker='s', markevery=10, markersize=4)
ax6.set_xlabel('t\' (time index)')
ax6.set_ylabel('|Error|')
ax6.set_title('Error: |thermal_sum_right[-1,:] - thermal_integral[-1,:]|')
ax6.set_xlim(-5, 0)
ax6.set_ylim(1e-4, 1)
ax6.grid(True, alpha=0.3)

# Row 4: Right sum [-2,:] vs thermal_integral[-2,:]
ax7 = plt.subplot(4, 2, 7)
ax7.plot(evolution.time_grid, np.abs(thermal_sum_right_minus), 'b-', linewidth=2, label='thermal_sum_right[-2,:]', alpha=0.7, marker='o', markevery=10, markersize=4)
ax7.plot(evolution.time_grid, np.abs(thermal_integral_second_last_row), 'r--', linewidth=2, label='thermal_integral[-2,:]', alpha=0.7, marker='^', markevery=10, markersize=4)
ax7.set_xlabel('t\' (time index)')
ax7.set_ylabel('Absolute value')
ax7.set_title('thermal_sum_right[-2,:] vs thermal_integral[-2,:]')
ax7.set_xlim(-5, 0)
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot(4, 2, 8)
ax8.semilogy(evolution.time_grid, np.abs(error_right_minus_vs_integral) + 1e-20, 'k-', linewidth=2, alpha=0.7, marker='s', markevery=10, markersize=4)
ax8.set_xlabel('t\' (time index)')
ax8.set_ylabel('|Error|')
ax8.set_title('Error: |thermal_sum_right[-2,:] - thermal_integral[-2,:]|')
ax8.set_xlim(-5, 0)
ax8.set_ylim(1e-4, 1)
ax8.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'Test_plots/thermal_sum_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved comparison plot to {output_path}")
#plt.show()

# ============================================================================
# CONVERGENCE TEST: MAX ERROR VS DELTA_T
# ============================================================================

print(f"\n" + "="*70)
print("CONVERGENCE TEST: MAX ERROR VS DELTA_T")
print("="*70)

# Fixed parameters
T_max_fixed = 2 * np.pi * 5
T = 0.5

# Range of N_t values to test
N_t_values = np.arange(501, 1501 + 250, 250)
delta_t_values = []
max_errors_left = []
max_errors_right = []

for N_t_test in N_t_values:
    print(f"\nTesting N_t = {N_t_test}")

    # Create grid parameters
    grid_params_test = {
        'time_sampling': N_t_test,
        'time_duration': T_max_fixed,
        'omega_sampling': 201,
        'cutoff': 20.0,
        'eta': 0.1
    }

    system_params_test = {
        'critical_temperature': 1.0,
        'temperature': T,
        'eta': 0.1
    }

    # Create evolution object
    evolution_test = UsadelKeldyshEvolution(
        grid_parameters=grid_params_test,
        system_parameters=system_params_test
    )

    # Compute thermal integral and sum
    evolution_test.get_thermal_integral(T)
    evolution_test.get_thermal_sum(T)

    # Extract data (only [-1,:] elements)
    ts_left = evolution_test.thermal_sum_left.data[0, 0, -1, :]
    ts_right = evolution_test.thermal_sum_right.data[0, 0, -1, :]

    ti_last_row = evolution_test.thermal_integral.data[0, 0, -1, :]
    ti_last_col = evolution_test.thermal_integral.data[0, 0, :, -1]

    # Compute errors
    err_left = np.abs(ts_left) - np.abs(ti_last_col)
    err_right = np.abs(ts_right) - np.abs(ti_last_row)

    # Store delta_t and max errors (excluding last index)
    delta_t = T_max_fixed / (N_t_test - 1)
    delta_t_values.append(delta_t)
    max_errors_left.append(np.max(np.abs(err_left[:-1])))
    max_errors_right.append(np.max(np.abs(err_right[:-1])))

    print(f"  delta_t = {delta_t:.6f}")
    print(f"  Max error left[-1,:] (excl. last): {max_errors_left[-1]:.6e}")
    print(f"  Max error right[-1,:] (excl. last): {max_errors_right[-1]:.6e}")

# Convert to arrays
delta_t_values = np.array(delta_t_values)
max_errors_left = np.array(max_errors_left)
max_errors_right = np.array(max_errors_right)

# Plot convergence (single plot with both left and right)
fig_conv = plt.figure(figsize=(10, 6))

ax = plt.subplot(1, 1, 1)
ax.loglog(delta_t_values, max_errors_left, 'b-o', linewidth=2, label='Left sum [-1,:] vs thermal_integral[:,-1]', markersize=8)
ax.loglog(delta_t_values, max_errors_right, 'r-s', linewidth=2, label='Right sum [-1,:] vs thermal_integral[-1,:]', markersize=8)
ax.set_xlabel('Δt', fontsize=12)
ax.set_ylabel('Max |Error| (excluding last index)', fontsize=12)
ax.set_title('Thermal Sum: Max Error vs Δt', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
output_path_conv = 'Test_plots/thermal_sum_convergence.png'
plt.savefig(output_path_conv, dpi=150, bbox_inches='tight')
print(f"\n\nSaved convergence plot to {output_path_conv}")
