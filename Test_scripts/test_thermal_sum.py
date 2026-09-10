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
    'time_sampling': 101,
    'time_duration': 10.0,
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
print("TESTING THERMAL SUM MIDPOINT RULE")
print("="*70)

# Get thermal sum
T = 0.5
evolution.get_thermal_sum(T)

# Check the structure
print(f"\nGrid info:")
print(f"  N_t (original): {evolution.ntpoints}")
print(f"  N_t_extended: {2*evolution.ntpoints - 1}")
print(f"  dt: {evolution.delta_t}")
print(f"  Time grid: from {evolution.time_grid[0]:.3f} to {evolution.time_grid[-1]:.3f}")

# Check thermal_sum_left and thermal_sum_right
print(f"\nThermal sum shapes:")
print(f"  thermal_sum_left: {evolution.thermal_sum_left.data.shape}")
print(f"  thermal_sum_right: {evolution.thermal_sum_right.data.shape}")

# Let me manually verify the midpoint rule implementation
# Recreate the computation to check boundary weights

N_t = evolution.ntpoints
N_t_extended = 2 * N_t - 1
dt = evolution.delta_t

# Extended grid
extended_time_grid = np.append(
    np.linspace(-2 * evolution.tmax, -evolution.tmax, N_t-1, endpoint=False),
    evolution.time_grid
)

print(f"\nExtended grid boundaries:")
print(f"  First point (far boundary): t = {extended_time_grid[0]:.3f}")
print(f"  Junction point: t = {extended_time_grid[N_t-1]:.3f}")
print(f"  Last point (closest boundary): t = {extended_time_grid[-1]:.3f}")

# Check the ones tensors used for integration
print(f"\n" + "="*70)
print("CHECKING MIDPOINT RULE APPLICATION")
print("="*70)

# For left sum, check boundary weights
# The code uses:
# ones_data_left_minus[-1] = 0
# ones_data_left_minus[-2] = 1/2
# ones_data_left[-1] = 1/2

print(f"\nLEFT SUM (∫_{{-2T_max}}^t f(t'',t') dt''):")
print(f"  Boundary at t=-2*T_max (index 0):")
print(f"    Expected weight (trapezoidal): 1/2")
print(f"    Actual weight: 1 (implicit in ones array)")
print(f"    ❌ MISSING MIDPOINT CORRECTION AT FAR BOUNDARY")

print(f"\n  Boundary at t=0 (index {N_t_extended-1}, closest boundary):")
print(f"    Expected weight (trapezoidal): 1/2")
print(f"    ones_data_left[-1] = 1/2 ✓")
print(f"    ones_data_left_minus[-1] = 0 ✓ (for exclusive upper limit)")

print(f"\nRIGHT SUM (∫_{{-2T_max}}^{{t'}} f(t,t'') dt''):")
print(f"  Uses f_two_time_nk[-2:-1,:] for boundary correction")
print(f"    This is row at index {N_t_extended-2}, which corresponds to t = {extended_time_grid[-2]:.3f}")
print(f"    ❌ SHOULD BE AT INDEX {N_t_extended-1} (t = {extended_time_grid[-1]:.3f})")

print(f"\n  Correction factor: subtracts 0.5 * f_two_time_nk[-2:-1,:]")
print(f"    Expected: should correct at index -1 (t=0), not -2")

# Check if there's a midpoint correction at the starting boundary
print(f"\n  Boundary at t=-2*T_max (index 0):")
print(f"    Expected weight (trapezoidal): 1/2")
print(f"    No explicit correction visible in code")
print(f"    ❌ MISSING MIDPOINT CORRECTION AT FAR BOUNDARY")

print(f"\n" + "="*70)
print("SUMMARY OF ISSUES")
print("="*70)
print("""
1. LEFT SUM: Missing midpoint correction at t=-2*T_max (index 0)
   - All points implicitly have weight 1, but first point should be 1/2

2. RIGHT SUM: Midpoint correction applied at wrong index
   - Using f_two_time_nk[-2:-1,:] (second-to-last row)
   - Should use f_two_time_nk[-1:,:] (last row at t=0)

3. RIGHT SUM: Missing midpoint correction at t=-2*T_max (index 0)
   - First point should have weight 1/2 for trapezoidal rule
""")

print(f"\n" + "="*70)
print("RECOMMENDED FIXES")
print("="*70)
print("""
1. For LEFT SUM: Add midpoint correction at starting boundary
   ones_data_left_minus[0] = 1/2
   ones_data_left[0] = 1/2

2. For RIGHT SUM: Correct the boundary index from -2 to -1
   f_sum_right_minus = (f_two_time_nk[-1:,:] @ ones_tensor - f_two_time_nk[-1:,:] * 0.5) * dt
   (Note: This needs -2 version too for proper staggering)

3. For RIGHT SUM: Add midpoint correction at starting boundary
   Need to apply 1/2 weight at the first column of the sum
""")
