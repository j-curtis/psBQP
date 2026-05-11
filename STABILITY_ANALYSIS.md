# Stability Analysis: τ₃ Error Accumulation

## Problem Statement

During time evolution, τ₃ components in off-diagonal elements g(t+Δt,t') accumulate exponentially, violating time-translation invariance in equilibrium. Observed error: g(t+Δt,t) differs from g(t,t-Δt) by ~0.02 in the τ₃ component.

## Root Cause: Exponential Instability

The bulk evolution formula in `_compute_new_gr_row` (line 323) is:

```python
gr_new = U_L * (gr_last_row + τ₃*gr_last_row*τ₃ - τ₃*gr_shifted*U_R_inv*τ₃)
```

This formula treats different Pauli components asymmetrically:

### Projection Properties

For off-diagonal elements (which should have only τ₁, τ₂ components):

**τ₂ component (correct behavior):**
```
τ₃*τ₂*τ₃ = -τ₂
∴ τ₂ + τ₃*τ₂*τ₃ = τ₂ - τ₂ = 0 ✓
```

**τ₃ component (unstable behavior):**
```
τ₃*τ₃*τ₃ = τ₃
∴ τ₃ + τ₃*τ₃*τ₃ = τ₃ + τ₃ = 2τ₃ ✗
```

### Amplification Mechanism

When equilibrium state has small τ₃ error (e.g., τ₃ = -0.04 from FFT grid mismatch):

1. **Bulk term:** `g + τ₃*g*τ₃` **doubles** the τ₃ component
   - Input: τ₃ = -0.04
   - Output: τ₃ = -0.16 (factor of 4 in term before U_L)

2. **Shifted term:** `τ₃*g_shifted*U_R_inv*τ₃` partially cancels
   - Contributes: τ₃ ≈ -0.016

3. **Net result:** `-0.16 - (-0.016) = -0.144`
   - After U_L multiplication: τ₃ ≈ -0.08

**Amplification factor: ~2.0 per timestep**

## Numerical Verification

Test with equilibrium state containing τ₃ = -0.04 error:

```
Input:  g(t,t') = τ₂ - 0.04*τ₃
        g(t,t'+Δt) = τ₂ - 0.04*τ₃

Output: g(t+Δt,t') = τ₂ - 0.0799*τ₃

Amplification: 1.998× per step
```

After N evolution steps: τ₃_error ≈ -0.04 × 2^N

## Why This Causes Problems

1. **Exponential growth:** With amplification ~2/step, errors grow catastrophically
   - After 10 steps: error × 1024
   - After 20 steps: error × 1,048,576

2. **Breaks time-translation invariance:** In equilibrium, g(t+Δt,t) should equal g(t,t-Δt), but:
   - g(t,t-Δt) has τ₃ = -0.04 (from FFT)
   - g(t+Δt,t) has τ₃ ≈ -0.08 (after one evolution)
   - Difference: ~0.04 (grows with more steps)

3. **Affects bulk, not just boundary:** Every off-diagonal element with τ₃ ≠ 0 gets amplified

## Sources of Initial τ₃ Error

### 1. FFT Grid Mismatch
In `equilibrium_class.py::_omega_to_two_time` (line 409):
- Evolution dt = T_max / (N_t - 1)
- FFT dtau = 2π / (dω × N_ω)
- Ratio: 0.9999106526374817 (small mismatch)
- Interpolation via `searchsorted` introduces τ₃ errors

### 2. FFT Regularization Artifacts
In `omega_to_one_time` (lines 264-270):
- Only τ₂ component is regularized for high-frequency behavior
- τ₃ component gets raw FFT with potential Gibbs oscillations
- No enforcement of τ₃ = 0 for off-diagonal elements after FFT

## Proposed Solutions

### Option 1: Force τ₃ = 0 in Equilibrium Grid (Quick Fix)
After computing equilibrium two-time GF, explicitly zero τ₃ for all off-diagonal elements:

```python
# In equilibrium_class.py after constructing gr_two_time
for i in range(ntpoints):
    for j in range(ntpoints):
        if i != j:  # Off-diagonal
            gr_two_time.data[0, 0, i, j] = 0.5 * gr_two_time.trace(2)[i, j]
            gr_two_time.data[1, 1, i, j] = 0.5 * gr_two_time.trace(2)[i, j]
```

**Pros:** Simple, eliminates initial τ₃ contamination
**Cons:** Doesn't fix evolution formula instability; roundoff errors will still grow

### Option 2: Stabilize Evolution Formula (Proper Fix)
Modify `_compute_new_gr_row` to explicitly project out τ₃ after each step:

```python
gr_new = unitary_propagator_L * (...)

# Project to pure τ₁, τ₂ subspace for off-diagonal
gr_new_projected = (NambuKeldyshTensor(gr_new.trace(1), pauli_channel=1) +
                    NambuKeldyshTensor(gr_new.trace(2), pauli_channel=2)) / 2
```

**Pros:** Prevents exponential growth, maintains correct structure
**Cons:** Adds computational overhead; masks potential physics issues

### Option 3: Fix FFT Grid Alignment (Root Cause)
Ensure dtau_fft = dt_evolution exactly by adjusting omega grid:

```python
# In equilibrium_class.py
dt = tmax / (ntpoints - 1)
d_omega = 2*np.pi / (dt * n_omega)
omega_grid = d_omega * (np.arange(n_omega) - n_omega // 2)
```

**Pros:** Eliminates grid mismatch, cleaner solution
**Cons:** Requires regenerating equilibrium data; may affect omega resolution

### Option 4: Damping Factor (Numerical Stability)
Add small damping to τ₃ component only:

```python
# After computing gr_new
tau3_component = gr_new.trace(3) * 0.99  # 1% damping per step
gr_new_stabilized = gr_new - NambuKeldyshTensor(0.01 * tau3_component, pauli_channel=3) / 2
```

**Pros:** Suppresses instability without changing physics
**Cons:** Ad-hoc; doesn't address underlying issue

## Recommended Action

**Combination of Options 1 + 3:**
1. Fix FFT grid alignment to eliminate initial τ₃ errors
2. Add explicit τ₃ = 0 enforcement in equilibrium grid as safety check
3. Monitor τ₃ magnitude during evolution; if it grows > 1e-6, flag a warning

This ensures both the initial condition and evolution preserve the physical constraint {g'^R(t,t'), τ₃} ≈ 0 for off-diagonal elements.

## Testing

After implementing fix, verify:
1. Equilibrium FFT has τ₃ < 1e-12 for all off-diagonal elements
2. Time evolution maintains τ₃ < 1e-10 after 100+ steps
3. Time-translation invariance: ||g(t+Δt,t) - g(t,t-Δt)|| < 1e-8

## References

- usadel_keldysh_evolution.py:323 (bulk evolution formula)
- equilibrium_class.py:409 (FFT grid mapping)
- Paper Eq. 2290-2297 (discrete evolution equations)
- Paper Eq. 2311 (constraint {g'^R(t,t), τ₃} = 0)
