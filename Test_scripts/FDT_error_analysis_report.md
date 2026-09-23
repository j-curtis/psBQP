# FDT Error Analysis Report: O(dt) Scaling and Root Causes

## Summary

**Finding:** FDT error scales linearly with dt: **Error ≈ 0.374 × dt** (±3.5% variation)

**Root Cause:** Midpoint rule applied to singular thermal functions at t=t' diagonal

---

## 1. Confirmed Linear Scaling

Test Results (N_t from 51 to 801):
```
N_t   |     dt      | Max Error  | Error/dt
------|-------------|------------|----------
  51  |  0.628319   | 2.187e-01  | 0.348
 101  |  0.314159   | 1.189e-01  | 0.378
 201  |  0.157080   | 5.990e-02  | 0.381
 401  |  0.078540   | 2.996e-02  | 0.381
 801  |  0.039270   | 1.499e-02  | 0.382
```

**Conclusion:** Error/dt ≈ constant → Error ∝ dt (linear scaling)

---

## 2. Root Cause: Singular Thermal Functions

### 2.1 Thermal Distribution f(τ)

**Definition (line 276):**
```python
f_two_time[mask] = -1j * temperature / np.sinh(np.pi * tau_matrix[mask] * temperature)
```

**Asymptotic behavior near τ=0:**
```
f(τ) = -iT / sinh(πτT) → -iT / (πτT) = -i/(πτ)  as τ→0
```
→ **1/τ singularity**

**Current handling (lines 273, 278-279):**
```python
mask = (np.abs(tau_matrix) > 1e-6)
# Diagonal (τ=0) remains zero (already initialized to zero)
```
→ **f(0) set to ZERO** (not the correct limiting behavior!)

### 2.2 Thermal Integral F(τ)

**Definition (lines 319-323):**
```python
x = np.pi * tau_vals[mask] * temperature
tanh_half = np.tanh(x / 2.0)
result[mask] = -1j/np.pi * np.log(tanh_half + 0j)
```

**Asymptotic behavior near τ=0:**
```
F(τ) = -i/π · ln(tanh(πτT/2)) → -i/π · ln(πτT/2)  as τ→0
```
→ **ln(τ) singularity**

**Current handling (lines 316, 324-325):**
```python
mask = (np.abs(tau_vals) > 1e-6)
# At τ = 0, set to 0 (principal value)
```
→ **F(0) set to ZERO** (diverges logarithmically!)

---

## 3. Why Midpoint Rule Gives O(dt) Error

### For Smooth Functions:
- Midpoint rule error: O(dt³) per interval
- Global error: O(dt²) ✓

### For Singular Functions:
Consider integral: ∫ f(t'') dt'' where f has 1/x singularity at x=0

**Midpoint rule near singularity:**
```
∫[ti, ti+1] (1/x) dx ≈ (1/xi_mid) · dt
```

But the correct integral is:
```
∫[ti, ti+1] (1/x) dx = ln|ti+1| - ln|ti| ≈ dt/ti  (for small ti)
```

**Error:**
```
Error = (1/xi_mid) · dt - dt/ti ≈ O(dt)  (not O(dt²)!)
```

The singularity **downgrades** the convergence order from dt² to dt.

---

## 4. How Setting f(0)=0 and F(0)=0 Causes Error

### 4.1 In Convolutions (precise_convolution)

**Formula (line 251):**
```python
result_std = (self @ other) * dt - 0.5 * dt * endpoint_corrections
```

This computes: dt · Σ self[i] · other[i]

**For diagonal/near-diagonal terms where |t-t'| < dt:**
- True: f(τ) ≈ -i/(πτ) (large!)
- Used: f(τ) = 0 (set to zero in mask)
- Missing contribution: O(1/dt) · dt = O(1)

This O(1) missing contribution appears as O(dt) error because:
- Number of affected points: constant (just diagonal ±few points)
- Error per affected convolution: O(1)
- Affects final result with weight dt
- Total error: O(dt)

### 4.2 In thermal_gap_term (line 595)

**Formula:**
```python
thermal_gap_term = (- f_thermal_integral * gap_tensor - gap_tensor * f_thermal_integral)
```

This is **element-wise multiplication** (not convolution), so:
```
thermal_gap_term[i,j] = -2 · F(ti - tj) · Δ(tj)
```

**Near diagonal (|ti - tj| ~ dt):**
- True: F(τ) ≈ -i/π · ln(πτT/2)
- Used: F(τ) = 0
- Missing: O(ln(dt)) term

This logarithmic term grows as dt→0, contributing O(dt) error.

---

## 5. Independent Errors in thermal_sum vs thermal_integral

From `test_thermal_sum.py`, you observed independent errors in:
- `thermal_sum_left` vs `thermal_integral[:,-1]`
- `thermal_sum_right` vs `thermal_integral[-1,:]`

**These should be equal because:**
```python
thermal_sum_right[t,:] = Σ f(t,t') = Σ f(t-t')
thermal_integral[t,:] = F(t,t') = F(t-t')
```

where F = ∫ f.

**But they differ because:**
1. **thermal_integral:** Computed analytically using F(τ) = -i/π·ln(tanh(...))
   - Sets F(0) = 0 for principal value
   - Has ln(τ) singularity discretization error

2. **thermal_sum:** Computed numerically via cumulative sum
   - Uses midpoint rule: dt · Σ f(τ)
   - f(0) = 0 creates artificial gap
   - Different discretization error pattern

**Result:** Both have O(dt) errors, but from different sources → independent errors.

---

## 6. Would Simpson's Rule Help?

**Short answer: Not for the singularity.**

**Simpson's rule:**
- For smooth f: Error = O(dt⁴) ✓✓✓
- For f with 1/τ singularity: Error = **O(dt)** ✗

**Why:**
Simpson's rule: ∫ f ≈ (dt/3)[f0 + 4f1 + f2]

For f(τ) = 1/τ near τ=0:
```
Simpson: (dt/3)[1/0 + 4/(dt/2) + 1/dt] → diverges!
```

Even with f(0)=0:
```
Simpson: (dt/3)[0 + 4/(dt/2) + 1/dt] = (dt/3)[8/dt + 1/dt] ≈ 3
True integral: ∫[0,dt] (1/τ)dτ → ∞ (diverges)
```

**Conclusion:** Higher-order quadrature doesn't help with singularities.

---

## 7. Proper Solution: Analytic Extraction of Singular Part

### Strategy:
1. **Split** f(τ) = f_sing(τ) + f_reg(τ)
   - f_sing(τ) = singular part (analytically integrable)
   - f_reg(τ) = regular part (smooth)

2. **Integrate separately:**
   - F_sing = analytical formula
   - F_reg = numerical integration (midpoint/Simpson)

3. **Combine:** F = F_sing + F_reg

### For thermal function:

**Exact form:**
```
f(τ) = -iT / sinh(πτT)
```

**Near τ=0 expansion:**
```
f(τ) = -iT / (πτT + O(τ³))
     = -i/(πτ) + O(τ)
     = f_sing(τ) + f_reg(τ)
```

where:
- f_sing(τ) = -i/(πτ)  (singular)
- f_reg(τ) = f(τ) - f_sing(τ)  (smooth at τ=0)

**Analytic integral of singular part:**
```
F_sing(τ) = ∫[-∞,τ] -i/(πτ') dτ' = -i/π · ln|τ| + const
```

**For convolution:** Replace numerical sum with analytic + smooth numerical.

---

## 8. Recommended Fixes

### Option 1: Proper Singular Extraction (Best, but complex)

Modify `get_thermal_occupation` and `get_thermal_integral`:

```python
# Split f(τ) = f_sing + f_reg
f_sing = -1j / (np.pi * tau_matrix + 1e-100)  # 1/τ part
f_reg = f_two_time - f_sing  # Smooth remainder

# Integrate separately
F_sing = -1j/np.pi * np.log(np.abs(tau_matrix) + 1e-100)  # Analytic
F_reg = cumulative_trapz(f_reg)  # Numerical (smooth, O(dt²))

F_total = F_sing + F_reg
```

Pros: Removes O(dt) error, achieves O(dt²)
Cons: Requires rewriting convolution formulas

### Option 2: Refined Grid Near Diagonal (Moderate effort)

Use adaptive spacing:
- Fine grid (spacing ~dt²) near |τ| < 10dt
- Regular grid (spacing dt) elsewhere

Pros: Reduces error without formula changes
Cons: Still O(dt) but with smaller constant

### Option 3: Use Principal Value + Regulator (Quick fix)

Instead of f(0)=0, use:
```python
epsilon = 0.1 * dt  # Small regulator
f_two_time[diagonal] = -1j * temperature / np.sinh(np.pi * epsilon * temperature)
```

Pros: Easy to implement
Cons: Error still O(dt), just smaller constant

### Option 4: Simpson's Rule for Thermal Sum (Moderate effort)

**Current implementation uses midpoint rule for thermal_sum.**

**Proposed:** Use Simpson's rule for smooth parts.

**Implementation:**
```python
# In get_thermal_sum():
# Current: thermal_sum = dt * cumsum(f)
# Proposed: thermal_sum = simpson_integrate(f)
```

**Effect:**
- For regions away from diagonal: O(dt²) → O(dt⁴) ✓
- For diagonal region: O(dt) → O(dt) (no change) ✗

**Verdict:** Helps overall but **doesn't fix the singularity error**.

---

## 9. Quantitative Predictions

**Current:** Error ≈ 0.374 × dt

**After Option 1 (proper extraction):** Error ≈ C × dt² (O(dt²))
- For dt=0.04: Error ~ 0.016 → 0.0016 (10x smaller)

**After Option 3 (regulator):** Error ≈ C' × dt (still O(dt))
- For dt=0.04: Error ~ 0.016 → 0.008 (2x smaller)

**After Option 4 (Simpson for thermal_sum):** Error ≈ 0.374 × dt (O(dt))
- Diagonal still dominates
- Maybe 20% reduction in constant

---

## 10. Recommendation

**For immediate improvement:**
- Implement Option 4 (Simpson for thermal_sum) - quick and gives marginal improvement
- Implement Option 3 (regulator at diagonal) - reduces constant factor

**For long-term solution:**
- Implement Option 1 (proper singular extraction)
- This is the only way to achieve O(dt²) and eliminate linear scaling

**Priority:** I recommend starting with Option 1, as it addresses the fundamental issue and provides the most significant improvement.
