"""
Test script to verify analytical integral of log(|x|).

Compares numerical integration (sum) vs analytical expression for:
∫_{-30}^{x} ln(|t|) dt

as a function of the upper integration limit x.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("TESTING LOG(|t|) ANALYTICAL INTEGRAL VS INTEGRATION LIMIT")
print("="*70)

# Parameters
lower_limit = -30.0  # Fixed lower limit
x_min = -10.0  # Minimum upper limit
x_max = 0.0  # Maximum upper limit
N_x = 200  # Number of different integration limits to test
dt = 0.03  # Fixed grid spacing

print(f"\nParameters:")
print(f"  Fixed lower limit: {lower_limit}")
print(f"  Upper limit range: {x_min} to {x_max}")
print(f"  Number of x values: {N_x}")
print(f"  Fixed grid spacing dt: {dt}")
print()

# Create array of upper integration limits
x_limits = np.linspace(x_min, x_max, N_x)

# Storage for results
numerical_results = []
analytical_results = []

print("Computing integrals for different limits...")

for x in x_limits:
    # Create grid from lower_limit to x with fixed dt
    t_grid = np.arange(lower_limit, x + dt/2, dt)  # Add dt/2 to ensure we reach x

    # Numerical integration: ∫_{-10}^{x} ln(|t|) dt
    log_values = np.log(np.abs(t_grid))

    # Handle potential issues near t=0
    log_values[np.isinf(log_values)] = 0.0

    # Trapezoidal rule with midpoint corrections
    numerical_integral = np.sum(log_values) * dt
    numerical_integral -= 0.5 * log_values[0] * dt   # Correct first point
    numerical_integral -= 0.5 * log_values[-1] * dt  # Correct last point

    numerical_results.append(numerical_integral)

    # Analytical integration
    # Antiderivative of ln(|t|) is t*ln(|t|) - t
    # At t = x: x*ln(|x|) - x
    # At t = -30: (-30)*ln(30) - (-30) = -30*ln(30) + 30
    # Result: [x*ln(|x|) - x] - [-30*ln(30) + 30]
    #       = x*ln(|x|) - x + 30*ln(30) - 30

    if np.abs(x) < 1e-10:
        # At x=0: lim_{x→0} x*ln(|x|) - x = 0
        F_upper = 0.0
    else:
        F_upper = x * np.log(np.abs(x)) - x

    F_lower = (-30.0) * np.log(30.0) - (-30.0)  # = -30*ln(30) + 30

    analytical_integral = F_upper - F_lower

    analytical_results.append(analytical_integral)

numerical_results = np.array(numerical_results)
analytical_results = np.array(analytical_results)
difference = numerical_results - analytical_results

print("  Done!")
print()

# Print some sample values
print("Sample values:")
print("  x       | Numerical       | Analytical      | Difference")
print("  " + "-"*62)
for idx in [0, N_x//4, N_x//2, 3*N_x//4, N_x-1]:
    x_val = x_limits[idx]
    num_val = numerical_results[idx]
    ana_val = analytical_results[idx]
    diff_val = difference[idx]
    print(f"  {x_val:6.2f}  | {num_val:15.8f} | {ana_val:15.8f} | {diff_val:11.4e}")

print()

# Statistics
max_abs_diff = np.max(np.abs(difference))
max_rel_diff = np.max(np.abs(difference) / (np.abs(analytical_results) + 1e-12))

print("="*70)
print("STATISTICS")
print("="*70)
print(f"Maximum absolute difference: {max_abs_diff:.6e}")
print(f"Maximum relative difference: {max_rel_diff:.6e}")
print()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Top plot: Numerical vs Analytical
ax1 = axes[0]
ax1.plot(x_limits, numerical_results, 'b-', linewidth=2, label='Numerical', alpha=0.8)
ax1.plot(x_limits, analytical_results, 'r--', linewidth=2, label='Analytical', alpha=0.8)
ax1.set_xlabel("Upper integration limit x", fontsize=12)
ax1.set_ylabel("∫_{-30}^{x} ln(|t|) dt", fontsize=12)
ax1.set_title("Comparison: Numerical vs Analytical Integration", fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Bottom plot: Difference
ax2 = axes[1]
ax2.plot(x_limits, difference, 'g-', linewidth=2)
ax2.set_xlabel("Upper integration limit x", fontsize=12)
ax2.set_ylabel("Difference (Numerical - Analytical)", fontsize=12)
ax2.set_title("Difference vs Upper Integration Limit", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()

import os
save_path = 'Test_plots/log_integral_comparison.png'
os.makedirs('Test_plots', exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {save_path}")

plt.close()

print()
print("="*70)
print("TEST COMPLETE")
print("="*70)
print()
print("Conclusion:")
if max_rel_diff < 1e-4:
    print("  ✓ Numerical and analytical results agree to high precision")
    print(f"    Maximum relative error: {max_rel_diff*100:.4f}%")
else:
    print(f"  ✗ Discrepancy of {max_rel_diff*100:.4f}% detected")
