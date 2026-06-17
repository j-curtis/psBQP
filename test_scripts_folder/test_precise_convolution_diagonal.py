"""
Test precise_convolution_left and precise_convolution_right for thermal terms.

Properly compares:
- thermal_gr = (gr @ f) using precise_convolution_left
- thermal_ga = (f @ ga) using precise_convolution_right

As they appear in the normalization equation.
"""

import numpy as np
import sys
import os

# Add parent directory to path FIRST to get local modules
sys.path.insert(0, '/home/filip/Documents/Research/Superconducting_vortices/Keldysh-non-eq/psBQP-keldysh')

from nambu_keldysh_class import NambuKeldyshTensor
from usadel_keldysh_evolution import UsadelKeldyshEvolution
from state_object_class import StateObject

# Import from local data_analysis module
import importlib.util
spec = importlib.util.spec_from_file_location("data_analysis_local",
    "/home/filip/Documents/Research/Superconducting_vortices/Keldysh-non-eq/psBQP-keldysh/data_analysis.py")
data_analysis_local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_analysis_local)

load_job_data = data_analysis_local.load_job_data
_get_system_parameters = data_analysis_local._get_system_parameters
_get_grid_parameters = data_analysis_local._get_grid_parameters

# Load equilibrium data
if len(sys.argv) > 1:
    timestamp = sys.argv[1]
else:
    print("Please provide timestamp as command-line argument")
    print("Usage: python test_precise_convolution_diagonal.py <timestamp>")
    sys.exit(1)

job_index = 0

# Load data using same mechanism as data_analysis
input_kwargs, save_data = load_job_data(timestamp, job_index)
state = save_data['final_state']

# Setup evolution to get thermal distribution
system_params = _get_system_parameters(input_kwargs)
grid_params = _get_grid_parameters(save_data, input_kwargs)

evolution = UsadelKeldyshEvolution(grid_params, system_params)
evolution.get_thermal_occupation(system_params['temperature'])
evolution.get_thermal_integral(system_params['temperature'])

f_thermal = evolution.thermal_dist
F_thermal = evolution.thermal_integral
dt = evolution.time_grid[1] - evolution.time_grid[0]
N_t = len(evolution.time_grid)

print("\n" + "="*70)
print("THERMAL TERM COMPARISON: PRECISE CONVOLUTIONS")
print("="*70)

# Use final time
t_idx = -1
t_pos = N_t + t_idx
t_val = evolution.time_grid[t_idx]

print(f"\nAt t_idx = {t_idx} (t = {t_val:.2f})")
print("-" * 70)

# Extract rows
gr_row = state.gr[t_pos:t_pos+1, :]
f_row = f_thermal[t_pos:t_pos+1, :]
F_row = F_thermal[t_pos:t_pos+1, :]

# Get g^A
ga = state._r2a()

# Compute thermal terms as in normalization
print("\nComputing thermal convolutions...")
thermal_gr = gr_row.precise_convolution_left(f_thermal, F_thermal, dt, other_index=t_pos)
thermal_ga = ga.precise_convolution_right(f_row, F_row, dt, self_index=t_pos)

print(f"  thermal_gr = gr_row @ f  (using precise_convolution_left)")
print(f"  thermal_ga = f_row @ ga  (using precise_convolution_right)")
print(f"  Shapes: thermal_gr={thermal_gr.data.shape}, thermal_ga={thermal_ga.data.shape}")

# Apply τ₃ multiplication
tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
thermal_gr_final = thermal_gr * tau3 * 2  # (gr@f) * τ₃ * 2
thermal_ga_final = 2 * tau3 * thermal_ga  # 2 * τ₃ * (f@ga)

print(f"\nChecking g^A matrix elements near diagonal:")
print(f"  g^A should satisfy: g^A(t,t') ≠ 0 only for t' ≥ t (anti-causality)")
print(f"  and g^A = -[g^R]^† (Hermitian conjugate with sign)")

# Check ga[-1,-2] and ga[-2,-1]
ga_11 = ga.data[:, :, -1, -1]  # ga(t=-1, t'=-1) - diagonal
ga_12 = ga.data[:, :, -1, -2]  # ga(t=-1, t'=-2) - should be zero by anti-causality
ga_21 = ga.data[:, :, -2, -1]  # ga(t=-2, t'=-1) - should be non-zero

# Also check gr for comparison
gr_11 = state.gr.data[:, :, -1, -1]
gr_12 = state.gr.data[:, :, -1, -2]
gr_21 = state.gr.data[:, :, -2, -1]

print(f"\n  g^R[-1,-1] (diagonal):")
for p in range(4):
    val = NambuKeldyshTensor(gr_11).trace(p) / 2
    print(f"    τ_{p}: {val:.6e}")

print(f"\n  g^A[-1,-1] (diagonal):")
for p in range(4):
    val = NambuKeldyshTensor(ga_11).trace(p) / 2
    print(f"    τ_{p}: {val:.6e}")

print(f"\n  g^R[-1,-2] (above diagonal, should be 0 by causality):")
for p in range(4):
    val = NambuKeldyshTensor(gr_12).trace(p) / 2
    print(f"    τ_{p}: {val:.6e}")

print(f"\n  g^A[-1,-2] (below diagonal, should be 0 by anti-causality):")
for p in range(4):
    val = NambuKeldyshTensor(ga_12).trace(p) / 2
    print(f"    τ_{p}: {val:.6e}")

print(f"\n  g^R[-2,-1] (below diagonal, should be non-zero):")
for p in range(4):
    val = NambuKeldyshTensor(gr_21).trace(p) / 2
    print(f"    τ_{p}: {val:.6e}")

print(f"\n  g^A[-2,-1] (above diagonal, should be non-zero):")
for p in range(4):
    val = NambuKeldyshTensor(ga_21).trace(p) / 2
    print(f"    τ_{p}: {val:.6e}")

print(f"\nConvolution results (gr@f) and (f@ga) at t=-1 for various t':")
print(f"Row format: t=-1, t'=<value>")
print(f"\n{'t_idx':<8} {'t_val':<10} {'Comp':<6} {'(gr@f)':<25} {'(f@ga)':<25} {'difference':<25}")
print("-" * 100)

for idx in [-1, -2, -3, -4, -5]:
    t_val = evolution.time_grid[idx]

    elem_gr = thermal_gr.data[:, :, 0, idx]
    elem_ga = thermal_ga.data[:, :, 0, idx]

    for p in range(4):
        gr_val = NambuKeldyshTensor(elem_gr).trace(p) / 2
        ga_val = NambuKeldyshTensor(elem_ga).trace(p) / 2
        diff = gr_val - ga_val

        if p == 0:  # Print index and t_val only for first component
            print(f"{idx:<8} {t_val:<10.2f} τ_{p:<4} {gr_val:<25.6e} {ga_val:<25.6e} {diff:<25.6e}")
        else:
            print(f"{'':8} {'':10} τ_{p:<4} {gr_val:<25.6e} {ga_val:<25.6e} {diff:<25.6e}")
    print()  # Blank line between indices

print(f"\nAfter τ₃ multiplication:")
print(f"  term1 = (gr@f) * τ₃ * 2")
print(f"  term2 = 2 * τ₃ * (f@ga)")

# Check along the row at several t' values
print(f"\nPauli components along row (fixed t={t_val:.2f}, varying t'):")
print(f"{'t_idx':<8} {'t_val':<10} {'Channel':<8} {'term1':<20} {'term2':<20} {'sum':<20}")
print("-" * 90)

for offset in range(-5, 1):
    idx = offset
    t2_val = evolution.time_grid[idx]

    elem1 = thermal_gr_final.data[:, :, 0, idx]
    elem2 = thermal_ga_final.data[:, :, 0, idx]

    for pauli_idx in [0, 1]:  # Just τ₀ and τ₁
        p1 = NambuKeldyshTensor(elem1).trace(pauli_idx) / 2
        p2 = NambuKeldyshTensor(elem2).trace(pauli_idx) / 2
        psum = p1 + p2

        diag_marker = " (DIAG)" if idx == -1 else ""
        print(f"{idx:<8} {t2_val:<10.2f} τ_{pauli_idx:<6} {p1:<20.6e} {p2:<20.6e} {psum:<20.6e}{diag_marker}")

print("\n" + "="*70)
print("KEY QUESTION: Why do term1 and term2 become equal at diagonal?")
print("="*70)
