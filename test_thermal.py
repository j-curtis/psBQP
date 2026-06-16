"""
Test script to verify thermal integral computation.

Compares analytical thermal integral F(t,t') with numerical integration of f(t,t').
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
temperature = 0.1  # Temperature in energy units
tmax = 50.0        # Maximum time
dt = 0.1           # Time step

# Create time grid
time_grid = np.arange(-tmax, dt, dt)
N_t = len(time_grid)
print(f"Time grid: [{time_grid[0]:.2f}, {time_grid[-1]:.2f}], N_t = {N_t}")

# Use two time indices for comparison
t_indices = [-1, N_t // 2]  # Last and middle
t_labels = ['last', 'middle']
print(f"Will plot for two time indices:")

# ==============================================================================
# 1. Compute thermal distribution f(t,t')
# ==============================================================================
# f(t,t') = f(t-t') = -iT / sinh(π(t-t')T)

def compute_thermal_dist(t_vals, tp_vals, T):
    """Compute f(t,t') = -iT / sinh(π(t-t')T)"""
    t_mesh, tp_mesh = np.meshgrid(t_vals, tp_vals, indexing='ij')
    tau = t_mesh - tp_mesh  # Relative time

    result = np.zeros_like(tau, dtype=complex)
    mask = (np.abs(tau) > 1e-6)  # Avoid singularity at τ=0

    x = np.pi * tau[mask] * T
    result[mask] = -1j * T / np.sinh(x)

    return result

f_thermal = compute_thermal_dist(time_grid, time_grid, temperature)
print(f"Thermal distribution shape: {f_thermal.shape}")

# ==============================================================================
# 2. Compute analytical thermal integral F(t,t')
# ==============================================================================
# F(t,t') = ∫_{-∞}^{t} dt'' f(t'',t')
# Using: F(τ) = 1/2 - i/π · ln(tanh(πτT/2))

def compute_thermal_integral_analytical(t_vals, tp_vals, T):
    """Compute F(t,t') using analytical formula"""
    t_mesh, tp_mesh = np.meshgrid(t_vals, tp_vals, indexing='ij')

    # Upper limit: τ = t - t'
    tau_upper = t_mesh - tp_mesh

    # Lower limit: τ = -tmax - t' (finite domain approximation)
    tau_lower = -tmax - tp_mesh

    def F_full(tau):
        """Analytical antiderivative: F(τ) = -i/π · ln(tanh(πτT/2))"""
        result = np.zeros_like(tau, dtype=complex)
        mask = (np.abs(tau) > 1e-6)

        x = np.pi * tau[mask] * T
        tanh_half = np.tanh(x / 2.0)
        result[mask] = -1j/np.pi * np.log(tanh_half + 0j)

        return result

    # F(t,t') = F(t-t') - F(-tmax-t')
    # F must be purely imaginary since f(τ) is purely imaginary
    F_upper = F_full(tau_upper)
    F_lower = F_full(tau_lower)

    # Take only imaginary part and multiply by 1j to ensure purely imaginary result
    return 1j * np.imag(F_upper - F_lower)

F_analytical = compute_thermal_integral_analytical(time_grid, time_grid, temperature)
print(f"Analytical thermal integral shape: {F_analytical.shape}")

# ==============================================================================
# 3. Compute numerical thermal integral by cumulative sum
# ==============================================================================
# F(t,t') = ∫_{-∞}^{t} dt'' f(t'',t') ≈ dt · Σ_{t''=-tmax}^{t} f(t'',t')

F_numerical = np.zeros_like(f_thermal, dtype=complex)
for tp_idx in range(N_t):
    # For each t', integrate over t from -tmax to each t value
    F_numerical[:, tp_idx] = np.cumsum(f_thermal[:, tp_idx]) * dt

print(f"Numerical thermal integral shape: {F_numerical.shape}")

# ==============================================================================
# 4 & 5. Loop over both time indices and plot
# ==============================================================================

for plot_num, (t_idx, t_label) in enumerate(zip(t_indices, t_labels)):
    t_actual = time_grid[t_idx]
    print(f"  [{plot_num+1}] t_idx = {t_idx} ({t_label}), t = {t_actual:.2f}")

    # Extract data for this fixed t vs t'
    F_ana_row = F_analytical[t_idx, :]  # Analytical F(t_fixed, t')
    F_num_row = F_numerical[t_idx, :]   # Numerical F(t_fixed, t')
    f_row = f_thermal[t_idx, :]         # Thermal dist f(t_fixed, t')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Analytical F - Real part
    ax = axes[0, 0]
    ax.plot(time_grid, np.real(F_ana_row), 'b-', linewidth=2, label='Analytical')
    ax.plot(time_grid, np.real(F_num_row), 'r--', linewidth=1.5, alpha=0.7, label='Numerical')
    ax.set_xlabel("$t'$", fontsize=12)
    ax.set_ylabel("Re[F(t,t')]", fontsize=12)
    ax.set_title(f"Thermal Integral - Real Part (t={t_actual:.2f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Analytical F - Imaginary part
    ax = axes[0, 1]
    ax.plot(time_grid, np.imag(F_ana_row), 'b-', linewidth=2, label='Analytical')
    ax.plot(time_grid, np.imag(F_num_row), 'r--', linewidth=1.5, alpha=0.7, label='Numerical')
    ax.set_xlabel("$t'$", fontsize=12)
    ax.set_ylabel("Im[F(t,t')]", fontsize=12)
    ax.set_title(f"Thermal Integral - Imaginary Part (t={t_actual:.2f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Difference between analytical and numerical
    ax = axes[1, 0]
    diff = F_ana_row - F_num_row
    ax.plot(time_grid, np.real(diff), 'g-', linewidth=2, label='Real part')
    ax.plot(time_grid, np.imag(diff), 'm-', linewidth=2, label='Imag part')
    ax.plot(time_grid, np.abs(diff), 'k--', linewidth=1.5, alpha=0.7, label='Magnitude')
    ax.set_xlabel("$t'$", fontsize=12)
    ax.set_ylabel("F_analytical - F_numerical", fontsize=12)
    ax.set_title(f"Difference (Analytical - Numerical) (t={t_actual:.2f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Thermal distribution f(t,t')
    ax = axes[1, 1]
    ax.plot(time_grid, np.real(f_row), 'b-', linewidth=2, label='Real part')
    ax.plot(time_grid, np.imag(f_row), 'r-', linewidth=2, label='Imag part')
    ax.set_xlabel("$t'$", fontsize=12)
    ax.set_ylabel("f(t,t')", fontsize=12)
    ax.set_title(f"Thermal Distribution (t={t_actual:.2f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'test_thermal_integral_t{t_label}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Plot saved as '{filename}'")
    plt.close()

    # Print statistics for this t
    print(f"\n  Statistics for t={t_actual:.2f}:")
    print(f"    Max |diff|: Real={np.max(np.abs(np.real(diff))):.4e}, Imag={np.max(np.abs(np.imag(diff))):.4e}")
    print(f"    Mean |diff|: {np.mean(np.abs(diff)):.4e}")

# ==============================================================================
# 6. Final summary
# ==============================================================================
print("\n" + "="*70)
print("THERMAL INTEGRAL VERIFICATION COMPLETE")
print("="*70)
print(f"Temperature: {temperature}")
print(f"Time grid: [{time_grid[0]:.2f}, {time_grid[-1]:.2f}], dt = {dt}")
print(f"Generated {len(t_indices)} comparison plots")
print("="*70)
