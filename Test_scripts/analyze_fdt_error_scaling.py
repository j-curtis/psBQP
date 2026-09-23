"""
Analyze FDT error scaling with delta_t.
Test with different grid sizes to confirm linear scaling.
"""

import numpy as np
import sys
sys.path.append('..')

from usadel_keldysh_evolution import UsadelKeldyshEvolution

print("="*80)
print("ANALYZING FDT ERROR SCALING WITH delta_t")
print("="*80)

T_max = 2 * np.pi * 5
temperature = 0.1

# Test with different grid sizes
N_t_values = [51, 101, 201, 401, 801]
results = []

for N_t in N_t_values:
    dt = T_max / (N_t - 1)
    
    grid_parameters = {
        'time_sampling': N_t,
        'time_duration': T_max,
        'eta': 0.2
    }
    
    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': temperature,
        'eta': 0.2
    }
    
    print(f"\n{'-'*80}")
    print(f"N_t = {N_t}, dt = {dt:.6f}")
    print(f"{'-'*80}")
    
    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
    initial_state, _, _, _ = evolution.generate_initial_state(Q=0.0)
    
    # Get thermal distributions
    evolution.get_thermal_occupation(temperature)
    evolution.get_thermal_integral(temperature)
    evolution.get_log_two_time(temperature)
    evolution.get_thermal_sum(temperature)
    
    # Check FDT
    gk_fdt, gk_actual, error, max_error = initial_state.check_fdt(
        f_thermal=evolution.thermal_dist,
        f_thermal_integral=evolution.thermal_integral,
        time_index=-1,
        thermal_sum_left=evolution.thermal_sum_left,
        thermal_sum_right=evolution.thermal_sum_right,
        log_two_time=evolution.log_two_time,
        tmax=evolution.tmax
    )
    
    # Extract tau_2 error (most important for gap)
    error_tau2 = error.trace(pauli_index=2) / 2
    max_error_tau2 = np.max(np.abs(error_tau2))
    
    results.append({
        'N_t': N_t,
        'dt': dt,
        'max_error': max_error,
        'max_error_tau2': max_error_tau2
    })
    
    print(f"  Max FDT error (all components): {max_error:.6e}")
    print(f"  Max FDT error (tau_2):          {max_error_tau2:.6e}")

print(f"\n{'='*80}")
print("SCALING ANALYSIS")
print("="*80)

print(f"\n{'N_t':>6} | {'dt':>10} | {'Max Error':>12} | {'Error_tau2':>12} | {'Error/dt':>12} | {'Error_tau2/dt':>15}")
print("-"*90)

for r in results:
    ratio = r['max_error'] / r['dt']
    ratio_tau2 = r['max_error_tau2'] / r['dt']
    print(f"{r['N_t']:>6} | {r['dt']:>10.6f} | {r['max_error']:>12.6e} | {r['max_error_tau2']:>12.6e} | {ratio:>12.6f} | {ratio_tau2:>15.6f}")

# Check if error/dt is approximately constant (linear scaling)
ratios = [r['max_error'] / r['dt'] for r in results]
ratios_tau2 = [r['max_error_tau2'] / r['dt'] for r in results]

mean_ratio = np.mean(ratios)
std_ratio = np.std(ratios)
mean_ratio_tau2 = np.mean(ratios_tau2)
std_ratio_tau2 = np.std(ratios_tau2)

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)

print(f"\nError / dt statistics (all components):")
print(f"  Mean: {mean_ratio:.6f}")
print(f"  Std:  {std_ratio:.6f}")
print(f"  Variation: {std_ratio/mean_ratio*100:.2f}%")

print(f"\nError_tau2 / dt statistics:")
print(f"  Mean: {mean_ratio_tau2:.6f}")
print(f"  Std:  {std_ratio_tau2:.6f}")
print(f"  Variation: {std_ratio_tau2/mean_ratio_tau2*100:.2f}%")

if std_ratio/mean_ratio < 0.1:
    print(f"\n✓ FDT error scales LINEARLY with dt (O(dt))")
    print(f"  Error ≈ {mean_ratio:.6f} * dt")
else:
    print(f"\n? FDT error scaling is not clearly linear")
    print(f"  May have dt^n scaling with n ≠ 1")

print("="*80)
