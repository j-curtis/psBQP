"""
Debug script to check asymptotic regularization in equilibrium g^K.
Plots g^K in energy space with and without asymptotic term.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.append('/home/filip/Documents/Research/BiPolaron/demler_tools')

from equilibrium_class import EquilibriumSolver
from nambu_keldysh_class import NambuKeldyshTensor

# Parameters matching the test output
grid_parameters = {
    'time_sampling': 1001,
    'time_duration': 12.5664,
    'omega_sampling': 1001,  # Need for equilibrium solver
    'cutoff': 30.0,  # Energy cutoff
    'eta': 0.2,  # Also needed in grid_parameters
}

system_parameters = {
    'critical_temperature': 1.0,
    'temperature': 0.5,
    'eta': 0.2,
}

print("Creating equilibrium solver...")
equilibrium_solver = EquilibriumSolver(
    grid_parameters,
    system_parameters,
    optimization_parameters=None,
    sigma_scatterings=None
)

# Compute equilibrium with gk
print("Computing equilibrium Green's functions...")
gr_eq, gk_eq = equilibrium_solver.compute_equilibrium_gr(
    temperature=0.5,
    Q=0.0,
    gr0=None,
    compute_gk=True
)

print(f"\nEquilibrium gap: gap_0 = {equilibrium_solver.gap_0}")
bcs_coupling = equilibrium_solver.usadel_solver.bcs_coupling
print(f"BCS coupling: λ = {bcs_coupling}")

omega_grid = equilibrium_solver.usadel_solver.w_arr
epsilon_grid = omega_grid  # In equilibrium, ε = ω

# Extract tau_2 component (imaginary gap channel)
# Before asymptotic is added, let's manually compute what we expect
print("\nExtracting g^K components...")

# Extract tau_2 component from g^K (off-diagonal imaginary)
# For g^K in tau_2: g^K[0,1] = -i*c and g^K[1,0] = i*c where c is the coefficient
# Use .data to access underlying JAX array
gk_2 = (gk_eq.data[1,0,:] - gk_eq.data[0,1,:]) / (2j)  # Extract imaginary coefficient

# Compute asymptotic term for tau_2
# For small ω: tanh(ω/2T) ≈ ω/2T, so 2iΔ/ω · tanh(ω/2T) → iΔ/T as ω→0
T = 0.5
tanh_omega = np.tanh(omega_grid / (2.0 * T))
asymptotic_gk2 = 2.0j * equilibrium_solver.gap_0 / (omega_grid + 1e-10) * tanh_omega

# Fix the value at ω=0 using L'Hôpital's rule / Taylor expansion
# lim_{ω→0} 2iΔ/ω · tanh(ω/2T) = iΔ/T
zero_idx = np.argmin(np.abs(omega_grid))
asymptotic_gk2 = np.array(asymptotic_gk2, dtype=complex)
asymptotic_gk2[zero_idx] = 1j * equilibrium_solver.gap_0 / T

# g^K regularized = g^K - asymptotic
gk_2_regularized = gk_2 + asymptotic_gk2

print("\nPlotting g^K in frequency space...")

# Compute gaps directly from frequency space
# Gap equation: Δ = -λ/4 * Tr[τ₋ g^K(τ=0)]
# In frequency space: Tr[τ₋ g^K(τ=0)] = ∫ dω/(2π) Tr[τ₋ g^K(ω)]
# For tau_2 component with coefficient c(ω): Tr[τ₋ · c(ω)·τ₂] = -ic(ω)

d_omega = omega_grid[1] - omega_grid[0]

# 1. Gap from g^K (with asymptotic) using bare coupling
tr_gk_tau0 = np.sum(-1j * gk_2 * d_omega) / (2 * np.pi)
gap_from_gk = -0.25 * bcs_coupling * tr_gk_tau0

# 2. Gap from g^K_regularized using renormalized coupling
Tc = 1.0
T = 0.5
bcs_coupling_prime = bcs_coupling / (0 - bcs_coupling * 1/(2*np.pi) * np.log(Tc/T))
tr_gk_reg_tau0 = np.sum(-1j * gk_2_regularized * d_omega) / (2 * np.pi)
gap_prime = -0.25 * bcs_coupling_prime * tr_gk_reg_tau0

print(f"\nGAP COMPUTATION (from frequency space):")
print(f"1. From g^K (with asymptotic), using bare λ = {bcs_coupling:.4f}:")
print(f"   Tr[τ₋ g^K(0)] = {tr_gk_tau0}")
print(f"   Δ = {gap_from_gk}")
print(f"   Δ/Δ₀ = {gap_from_gk / equilibrium_solver.gap_0}")

print(f"\n2. From g^K_regularized, using λ' = {bcs_coupling_prime:.4f}:")
print(f"   Tr[τ₋ g^K_reg(0)] = {tr_gk_reg_tau0}")
print(f"   Δ' = {gap_prime}")
print(f"   Δ'/Δ₀ = {gap_prime / equilibrium_solver.gap_0}")

# Create plots in frequency space
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: g^K(ε) in tau_2 channel (with asymptotic) - FULL RANGE
ax = axes[0, 0]
ax.plot(omega_grid, gk_2.imag, 'b-', linewidth=1.5, label='Im[g^K_2(ε)]')
ax.plot(omega_grid, asymptotic_gk2.imag, 'r--', linewidth=1.5, alpha=0.7,
        label='Asymptotic: 2iΔ/ε·tanh(ε/2T)')
ax.set_xlabel('ε (energy)', fontsize=14)
ax.set_ylabel('Im[g^K_2(ε)]', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('g^K in tau_2 channel (with asymptotic)', fontsize=14, fontweight='bold')

# Plot 2: g^K_regularized(ε) in tau_2 channel (without asymptotic) - FULL RANGE
ax = axes[0, 1]
ax.plot(omega_grid, gk_2_regularized.imag, 'g-', linewidth=1.5, label='Im[g^K_2_reg(ε)]')
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('ε (energy)', fontsize=14)
ax.set_ylabel('Im[g^K_2_reg(ε)]', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('g^K_regularized (asymptotic removed)', fontsize=14, fontweight='bold')

# Plot 3: Both on same plot for comparison - FULL RANGE
ax = axes[0, 2]
ax.plot(omega_grid, gk_2.imag, 'b-', linewidth=1.5, label='g^K_2 (with asymp)')
ax.plot(omega_grid, gk_2_regularized.imag, 'g-', linewidth=1.5, label='g^K_2_reg')
ax.plot(omega_grid, asymptotic_gk2.imag, 'r--', linewidth=1.5, alpha=0.7, label='Asymptotic')
ax.set_xlabel('ε (energy)', fontsize=14)
ax.set_ylabel('Im[g^K_2(ε)]', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('Comparison', fontsize=14, fontweight='bold')

# Plot 4: Zoom near gap edge - g^K
ax = axes[1, 0]
Delta = equilibrium_solver.gap_0
mask_gap = np.abs(omega_grid) < 3*Delta
ax.plot(omega_grid[mask_gap], gk_2.imag[mask_gap], 'b-', linewidth=2)
ax.axvline(x=Delta, color='r', linestyle='--', alpha=0.5, label=f'Δ = {Delta:.3f}')
ax.axvline(x=-Delta, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('ε (energy)', fontsize=14)
ax.set_ylabel('Im[g^K_2(ε)]', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('g^K near gap edge', fontsize=14, fontweight='bold')

# Plot 5: Zoom near gap edge - g^K_regularized
ax = axes[1, 1]
ax.plot(omega_grid[mask_gap], gk_2_regularized.imag[mask_gap], 'g-', linewidth=2)
ax.axvline(x=Delta, color='r', linestyle='--', alpha=0.5, label=f'Δ = {Delta:.3f}')
ax.axvline(x=-Delta, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('ε (energy)', fontsize=14)
ax.set_ylabel('Im[g^K_2_reg(ε)]', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('g^K_regularized near gap edge', fontsize=14, fontweight='bold')

# Summary statistics
ax = axes[1, 2]
ax.axis('off')
info_text = f"""
PARAMETERS:
────────────────────────────
Equilibrium gap: Δ₀ = {equilibrium_solver.gap_0:.4f}
BCS coupling: λ = {bcs_coupling:.4f}
Renormalized: λ' = {bcs_coupling_prime:.4f}
Temperature: T = {T}
Critical temp: Tc = {Tc}

ASYMPTOTIC FORM:
────────────────────────────
g^K_2(ε) → 2iΔ/ε · tanh(ε/2T)
            for |ε| → ∞

GAP FROM FREQUENCY SPACE:
────────────────────────────
From g^K (bare λ):
  Δ = {gap_from_gk.real:.4f}
  Δ/Δ₀ = {(gap_from_gk/equilibrium_solver.gap_0).real:.4f}

From g^K_reg (λ'):
  Δ' = {gap_prime.real:.4f}
  Δ'/Δ₀ = {(gap_prime/equilibrium_solver.gap_0).real:.4f}

PLOTS:
────────────────────────────
Top: Full frequency range
Bottom: Zoom near gap Δ₀
"""
ax.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.suptitle(f'Equilibrium g^K_2 components (gap_0 = {equilibrium_solver.gap_0:.4f})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(script_dir, 'Test_plots', 'debug_gk_omega.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot: {plot_path}")
plt.show()
