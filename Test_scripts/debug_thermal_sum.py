"""
Debug script to compare thermal_sum with thermal_integral.

Compares values for t=-1 and t=-2 vs t' to diagnose discrepancies.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usadel_keldysh_evolution import UsadelKeldyshEvolution

# Set up parameters
grid_parameters = {
    't_max': 10.0,
    'n_tpoints': 1000,
}

system_parameters = {
    'temperature': 0.1,
    'critical_temperature': 0.2,
    'bcs_coupling_constant': 1.0,
    'eta': 0.01,
}

# Create evolution object
print("Creating evolution object...")
evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)

# Compute thermal distributions
print("Computing thermal occupation...")
evolution.get_thermal_occupation(system_parameters['temperature'])

print("Computing thermal integral...")
evolution.get_thermal_integral(system_parameters['temperature'])

print("Computing thermal sum...")
evolution.get_thermal_sum(system_parameters['temperature'])

# Extract data for comparison
time_grid = evolution.time_grid
N_t = len(time_grid)

# Extract Pauli component 0 (tau_0) for all comparisons
pauli_idx = 0

# thermal_integral has shape (2,2,N_t,N_t)
# thermal_sum_left/right have shape (2,2,2,N_t)

print(f"\nData shapes:")
print(f"thermal_integral: {evolution.thermal_integral.data.shape}")
print(f"thermal_sum_left: {evolution.thermal_sum_left.data.shape}")
print(f"thermal_sum_right: {evolution.thermal_sum_right.data.shape}")

# For LEFT sum: integral from -2*T_max to t
# thermal_sum_left[t_idx, t_prime_idx] should equal integral from -inf to t for fixed t'
# Compare with thermal_integral which integrates from -inf to t' for fixed t

# Extract for t=-1 (index -1 in thermal_sum becomes index 1 since it has only 2 rows)
thermal_sum_left_t_minus1 = evolution.thermal_sum_left.data[pauli_idx, pauli_idx, 1, :]  # shape (N_t,)
thermal_sum_left_t_minus2 = evolution.thermal_sum_left.data[pauli_idx, pauli_idx, 0, :]  # shape (N_t,)

# thermal_integral[t, t'] = integral from -inf to t' of f(t, t'')
thermal_integral_t_minus1 = evolution.thermal_integral.data[pauli_idx, pauli_idx, -1, :]  # shape (N_t,)
thermal_integral_t_minus2 = evolution.thermal_integral.data[pauli_idx, pauli_idx, -2, :]  # shape (N_t,)

# For RIGHT sum: integral from -2*T_max to t'
thermal_sum_right_t_minus1 = evolution.thermal_sum_right.data[pauli_idx, pauli_idx, 1, :]  # shape (N_t,)
thermal_sum_right_t_minus2 = evolution.thermal_sum_right.data[pauli_idx, pauli_idx, 0, :]  # shape (N_t,)

# Compute direct sum from thermal_dist (no extended grid, just -T_max to 0)
print(f"\nComputing direct sums from thermal_dist (standard grid)...")
dt = evolution.delta_t

# For LEFT sum: sum f(t'', t') over t'' for each fixed t'
# CUMULATIVE: ∫_{-T_max}^{t'} f(t'', t') dt'' for each t'
direct_left_sum_minus1 = np.zeros(N_t, dtype=complex)
direct_left_sum_minus2 = np.zeros(N_t, dtype=complex)

for t_prime_idx in range(N_t):
    # Sum f(t'', t'=t_prime_idx) over t'' from 0 to t_prime_idx
    # Using trapezoidal rule
    weights = np.ones(t_prime_idx + 1)
    weights[0] = 0.5
    weights[-1] = 0.5
    # Sum over first axis (t'') for fixed second axis (t')
    direct_left_sum_minus1[t_prime_idx] = dt * np.sum(
        weights * evolution.thermal_dist.data[pauli_idx, pauli_idx, :t_prime_idx+1, t_prime_idx]
    )
    direct_left_sum_minus2[t_prime_idx] = dt * np.sum(
        weights * evolution.thermal_dist.data[pauli_idx, pauli_idx, :t_prime_idx+1, t_prime_idx]
    )

# FULL sum: ∫_{-T_max}^{0} f(t'', t') dt'' for each t' (sum over ALL t'')
full_left_sum = np.zeros(N_t, dtype=complex)

for t_prime_idx in range(N_t):
    # Sum f(t'', t'=t_prime_idx) over ALL t'' from 0 to N_t-1
    weights = np.ones(N_t)
    weights[0] = 0.5
    weights[-1] = 0.5
    full_left_sum[t_prime_idx] = dt * np.sum(
        weights * evolution.thermal_dist.data[pauli_idx, pauli_idx, :, t_prime_idx]
    )

# For RIGHT sum: sum f(t, t'') over t'' for each fixed t
# For t=-1: ∫_{-T_max}^{t'} f(t=-1, t'') dt'' for each t'
# thermal_dist[-1, :] gives f(t=-1, t'') for all t''
# We want cumulative sum over t'' dimension
direct_right_sum_minus1 = np.zeros(N_t, dtype=complex)
direct_right_sum_minus2 = np.zeros(N_t, dtype=complex)

for t_prime_idx in range(N_t):
    # Sum f(t=-1, t'') over t'' from 0 to t_prime_idx
    weights = np.ones(t_prime_idx + 1)
    weights[0] = 0.5
    weights[-1] = 0.5
    # thermal_dist[-1, :t_prime_idx+1] is f(t=-1, t''=0..t_prime_idx)
    direct_right_sum_minus1[t_prime_idx] = dt * np.sum(
        weights * evolution.thermal_dist.data[pauli_idx, pauli_idx, -1, :t_prime_idx+1]
    )
    direct_right_sum_minus2[t_prime_idx] = dt * np.sum(
        weights * evolution.thermal_dist.data[pauli_idx, pauli_idx, -2, :t_prime_idx+1]
    )

print(f"\nDirect sum values (from standard grid):")
print(f"direct_left_sum (cumulative, t=-1): max={np.max(np.abs(direct_left_sum_minus1)):.3e}")
print(f"full_left_sum (all t'', varying t'): max={np.max(np.abs(full_left_sum)):.3e}")
print(f"direct_right_sum (t=-1): max={np.max(np.abs(direct_right_sum_minus1)):.3e}")

# Check dt values
print(f"\nGrid spacing check:")
print(f"Original dt: {dt:.6f}")
dt_extended = 2*evolution.tmax / (2*evolution.ntpoints - 1)
print(f"Extended dt: {dt_extended:.6f}")
print(f"Ratio (dt/dt_extended): {dt/dt_extended:.6f}")

# Manually compute full sum on extended grid to compare with thermal_sum_right
print(f"\nManual extended grid computation:")
N_t_extended = 2 * N_t
extended_time_grid = np.linspace(-2*evolution.tmax, 0, N_t_extended)

# Compute f_extended[i, j] = f(extended_grid[i], time_grid[j])
manual_full_sum_extended = np.zeros(N_t, dtype=complex)

for t_prime_idx in range(N_t):
    t_prime = time_grid[t_prime_idx]
    # Sum over all extended grid points
    f_values = np.zeros(N_t_extended, dtype=complex)
    for i, t_double_prime in enumerate(extended_time_grid):
        tau = t_double_prime - t_prime
        if np.abs(tau) > 1e-6:
            f_values[i] = -1j * evolution.temperature / np.sinh(np.pi * tau * evolution.temperature)

    # Apply trapezoidal weights
    weights = np.ones(N_t_extended)
    weights[0] = 0.5  # First point
    weights[-1] = 0.5  # Last point

    manual_full_sum_extended[t_prime_idx] = dt * np.sum(weights * f_values)

print(f"Manual extended full sum: max={np.max(np.abs(manual_full_sum_extended)):.3e}")
print(f"thermal_sum_right (from get_thermal_sum): max={np.max(np.abs(thermal_sum_right_t_minus1)):.3e}")

# Now break down the contribution by grid region
print(f"\nBreakdown of contributions:")

# Contribution from extended part only (t'' from -2*T_max to -T_max)
extended_part_sum = np.zeros(N_t, dtype=complex)
# Contribution from standard part only (t'' from -T_max to 0)
standard_part_sum = np.zeros(N_t, dtype=complex)

for t_prime_idx in range(N_t):
    t_prime = time_grid[t_prime_idx]

    # Extended part: indices 0 to N_t-1 in extended grid
    f_values_extended = np.zeros(N_t, dtype=complex)
    for i in range(N_t):
        t_double_prime = extended_time_grid[i]
        tau = t_double_prime - t_prime
        if np.abs(tau) > 1e-6:
            f_values_extended[i] = -1j * evolution.temperature / np.sinh(np.pi * tau * evolution.temperature)

    weights_ext = np.ones(N_t)
    weights_ext[0] = 0.5
    weights_ext[-1] = 0.5
    extended_part_sum[t_prime_idx] = dt * np.sum(weights_ext * f_values_extended)

    # Standard part: indices N_t to 2*N_t-1 in extended grid
    f_values_standard = np.zeros(N_t, dtype=complex)
    for i in range(N_t, N_t_extended):
        t_double_prime = extended_time_grid[i]
        tau = t_double_prime - t_prime
        if np.abs(tau) > 1e-6:
            f_values_standard[i - N_t] = -1j * evolution.temperature / np.sinh(np.pi * tau * evolution.temperature)

    weights_std = np.ones(N_t)
    weights_std[0] = 0.5
    weights_std[-1] = 0.5
    standard_part_sum[t_prime_idx] = dt * np.sum(weights_std * f_values_standard)

print(f"Extended part only (t'' ∈ [-2*T_max, -T_max]): max={np.max(np.abs(extended_part_sum)):.3e}")
print(f"Standard part only (t'' ∈ [-T_max, 0]): max={np.max(np.abs(standard_part_sum)):.3e}")
print(f"Sum of parts: {np.max(np.abs(extended_part_sum + standard_part_sum)):.3e}")
print(f"Full extended grid: {np.max(np.abs(manual_full_sum_extended)):.3e}")

# Debug: check values at a specific point (t' = 0, last index)
print(f"\nDebug at t'=0 (last index):")
t_prime = 0.0
print(f"Standard grid thermal_dist sum at t'=0: {np.abs(full_left_sum[-1]):.3e}")
print(f"Extended grid standard part at t'=0: {np.abs(standard_part_sum[-1]):.3e}")

# Find where the maximum is
max_idx = np.argmax(np.abs(standard_part_sum))
print(f"\nMaximum is at index {max_idx}, t'={time_grid[max_idx]:.6f}")
print(f"Standard part sum at max: {np.abs(standard_part_sum[max_idx]):.3e}")
print(f"Full left sum at max: {np.abs(full_left_sum[max_idx]):.3e}")
print(f"Ratio: {np.abs(standard_part_sum[max_idx]) / np.abs(full_left_sum[max_idx]):.1f}×")

# Check f values at this problematic t'
t_prime_prob = time_grid[max_idx]
print(f"\nf values near t'={t_prime_prob:.6f}:")
print(f"  From standard grid (thermal_dist):")
for i in range(max(0, max_idx-2), min(N_t, max_idx+3)):
    t_pp = time_grid[i]
    tau = t_pp - t_prime_prob
    if np.abs(tau) > 1e-6:
        f_val_std = evolution.thermal_dist.data[pauli_idx, pauli_idx, i, max_idx]
        print(f"    t''={t_pp:.6f}, τ={tau:.6f}, |f|={np.abs(f_val_std):.3e}")

print(f"  From extended grid:")
for i in range(N_t_extended-5, N_t_extended):
    t_pp = extended_time_grid[i]
    tau = t_pp - t_prime_prob
    if np.abs(tau) > 1e-6:
        f_val = -1j * evolution.temperature / np.sinh(np.pi * tau * evolution.temperature)
        print(f"    t''={t_pp:.6f}, τ={tau:.6f}, |f|={np.abs(f_val):.3e}")

# Check a few individual f values
print(f"\nSample f values for t'=0:")
for i in [N_t, N_t + N_t//4, N_t + N_t//2, N_t_extended-1]:
    t_pp = extended_time_grid[i]
    tau = t_pp - 0.0
    if np.abs(tau) > 1e-6:
        f_val = -1j * evolution.temperature / np.sinh(np.pi * tau * evolution.temperature)
        print(f"  t''={t_pp:.3f}, τ={tau:.3f}, |f|={np.abs(f_val):.3e}")
    else:
        print(f"  t''={t_pp:.3f}, τ={tau:.3f}, |f|=0 (masked)")

print(f"\nValue ranges (Pauli component τ_{pauli_idx}):")
print(f"thermal_sum_left (t=-1): [{np.min(np.abs(thermal_sum_left_t_minus1)):.3e}, {np.max(np.abs(thermal_sum_left_t_minus1)):.3e}]")
print(f"thermal_integral (t=-1): [{np.min(np.abs(thermal_integral_t_minus1)):.3e}, {np.max(np.abs(thermal_integral_t_minus1)):.3e}]")
print(f"thermal_sum_right (t=-1): [{np.min(np.abs(thermal_sum_right_t_minus1)):.3e}, {np.max(np.abs(thermal_sum_right_t_minus1)):.3e}]")

print(f"\nRatio (sum/integral) for t=-1:")
ratio_left = np.abs(thermal_sum_left_t_minus1) / (np.abs(thermal_integral_t_minus1) + 1e-10)
ratio_right = np.abs(thermal_sum_right_t_minus1) / (np.abs(thermal_integral_t_minus1) + 1e-10)
print(f"Left sum (extended):  mean={np.mean(ratio_left):.2f}, max={np.max(ratio_left):.2f}")
print(f"Right sum (extended): mean={np.mean(ratio_right):.2f}, max={np.max(ratio_right):.2f}")

# Check early time behavior
print(f"\nEarly time analysis (first 5 points):")
for i in range(5):
    print(f"  t'={time_grid[i]:.4f}: thermal_sum_left={np.abs(thermal_sum_left_t_minus1[i]):.3e}, "
          f"thermal_integral={np.abs(thermal_integral_t_minus1[i]):.3e}, "
          f"ratio={ratio_left[i]:.1f}")

# Find where the maximum ratio occurs
max_ratio_idx = np.argmax(ratio_left)
print(f"\nMaximum ratio at index {max_ratio_idx}, t'={time_grid[max_ratio_idx]:.6f}:")
print(f"  thermal_sum_left: {np.abs(thermal_sum_left_t_minus1[max_ratio_idx]):.3e}")
print(f"  thermal_integral: {np.abs(thermal_integral_t_minus1[max_ratio_idx]):.3e}")
print(f"  Ratio: {ratio_left[max_ratio_idx]:.1f}")

# Add comparison between extended and standard grid sums before creating plots
print(f"\nComparison: extended grid vs standard grid sums (t=-1):")
print(f"LEFT sum - extended max: {np.max(np.abs(thermal_sum_left_t_minus1)):.3e}, standard max: {np.max(np.abs(direct_left_sum_minus1)):.3e}")
print(f"RIGHT sum - extended max: {np.max(np.abs(thermal_sum_right_t_minus1)):.3e}, standard max: {np.max(np.abs(direct_right_sum_minus1)):.3e}")

# Create plots - only imaginary parts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 1: t = -1
ax = axes[0, 0]
ax.plot(time_grid, np.imag(thermal_sum_left_t_minus1), label='thermal_sum_left (extended)', linewidth=2)
ax.plot(time_grid, np.imag(direct_left_sum_minus1), '-.', label='cumulative sum (standard)', linewidth=2)
ax.plot(time_grid, np.imag(full_left_sum), ':', label='full sum (all t")', linewidth=2)
ax.plot(time_grid, np.imag(thermal_integral_t_minus1), '--', label='thermal_integral', linewidth=2)
ax.set_xlabel("t'")
ax.set_ylabel("Imaginary part")
ax.set_title("LEFT sum: t=-1 vs t'")
ax.legend()
ax.grid(True)

ax = axes[0, 1]
ax.plot(time_grid, np.imag(thermal_sum_right_t_minus1), label='thermal_sum_right (extended)', linewidth=2)
ax.plot(time_grid, np.imag(direct_right_sum_minus1), '-.', label='direct sum (standard)', linewidth=2)
ax.plot(time_grid, np.imag(thermal_integral_t_minus1), '--', label='thermal_integral', linewidth=2)
ax.set_xlabel("t'")
ax.set_ylabel("Imaginary part")
ax.set_title("RIGHT sum: t=-1 vs t'")
ax.legend()
ax.grid(True)

# Row 2: t = -2
ax = axes[1, 0]
ax.plot(time_grid, np.imag(thermal_sum_left_t_minus2), label='thermal_sum_left (extended)', linewidth=2)
ax.plot(time_grid, np.imag(direct_left_sum_minus2), '-.', label='direct sum (standard)', linewidth=2)
ax.plot(time_grid, np.imag(thermal_integral_t_minus2), '--', label='thermal_integral', linewidth=2)
ax.set_xlabel("t'")
ax.set_ylabel("Imaginary part")
ax.set_title("LEFT sum: t=-2 vs t'")
ax.legend()
ax.grid(True)

ax = axes[1, 1]
ax.plot(time_grid, np.imag(thermal_sum_right_t_minus2), label='thermal_sum_right (extended)', linewidth=2)
ax.plot(time_grid, np.imag(direct_right_sum_minus2), '-.', label='direct sum (standard)', linewidth=2)
ax.plot(time_grid, np.imag(thermal_integral_t_minus2), '--', label='thermal_integral', linewidth=2)
ax.set_xlabel("t'")
ax.set_ylabel("Imaginary part")
ax.set_title("RIGHT sum: t=-2 vs t'")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('Test_plots/thermal_sum_vs_integral_comparison.png', dpi=150)
print(f"\nPlot saved to Test_plots/thermal_sum_vs_integral_comparison.png")

# Check all Pauli components for t=-1
print(f"\n=== All Pauli components for t=-1 ===")
for pauli in range(2):  # Only check tau_0 and tau_1 (thermal_sum only has 2 rows)
    sum_left = evolution.thermal_sum_left.data[pauli, pauli, 1, :]
    integral = evolution.thermal_integral.data[pauli, pauli, -1, :]
    max_sum = np.max(np.abs(sum_left))
    max_int = np.max(np.abs(integral))
    print(f"τ_{pauli}: thermal_sum_left max={max_sum:.3e}, thermal_integral max={max_int:.3e}, ratio={max_sum/(max_int+1e-10):.2f}")

plt.show()
