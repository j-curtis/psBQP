# Analysis: O(1) Diagonal GK Error - Root Cause

## Summary

The O(1) error in GK normalization near the diagonal comes from **fundamental dimensional mismatch** in the thermal sum computation. The code computes 1D vectors when it should compute 2D time-dependent matrices.

---

## The Core Issue

### What the Code Should Compute (per docstring)

```python
# Lines 366-370 in usadel_keldysh_evolution.py
"""
Computes two-time matrices similar to get_thermal_occupation:
- SUM_right(t,t') = ∫_{-2*T_max}^{t'} f(t,t'') dt'' for all (t,t') pairs
- SUM_left(t,t') = ∫_{-2*T_max}^{t} f(t'',t') dt'' for all (t,t') pairs
"""
```

**Expected shape:** `(N_t, N_t)` - a full two-time matrix where each element depends on BOTH t and t'

### What the Code Actually Computes

**Lines 414-415:**
```python
f_sum_left = (ones_tensor_left @ f_two_time_nk) * dt
```
- `ones_tensor_left`: shape `(1, N_t_extended)` with all 1s (last element 0.5)
- `f_two_time_nk`: shape `(N_t_extended, N_t)` where rows=t'' (extended), cols=t' (standard)
- Matrix mult: `(1, N_t_extended) @ (N_t_extended, N_t) = (1, N_t)`
- **Computes:** `∑_{t''=-2*T_max}^{0} f(t'', t') * dt` for each t'
- **Independent of t!** This is a 1D vector, not a 2D matrix

**Lines 422-423:**
```python
ones_data = (row_indices <= col_indices).astype(complex)  # Shape (N_t, N_t)
f_sum_right = (f_two_time_nk[-1:,:] @ ones_tensor - f_two_time_nk[-1:,:] * 0.5) * dt
```
- `f_two_time_nk[-1:,:]`: Extracts ONLY the last row (t''=0), shape `(1, N_t)`
- `ones_tensor`: shape `(N_t, N_t)` with cumulative mask
- Matrix mult: `(1, N_t) @ (N_t, N_t) = (1, N_t)`
- **Computes:** `∑_{j≤k} f(0, t'_j)` for each k
- **Only uses t''=0 from extended grid!** Completely wrong.

**Final storage (lines 426-427):**
```python
self.thermal_sum_left = NambuKeldyshTensor(np.append(f_sum_right_minus.data[0,0],
                                                      f_sum_right.data[0,0], axis=0),
                                           pauli_channel=0)
self.thermal_sum_right = NambuKeldyshTensor(np.append(f_sum_left_minus.data[0,0],
                                                       f_sum_left.data[0,0], axis=0),
                                            pauli_channel=0)
```
- Appends two (1, N_t) vectors vertically → shape `(2, N_t)`
- Wrapped in NambuKeldyshTensor → final shape `(2, 2, 2, N_t)`
- **The "2" index is NOT a time index** - it's just two endpoint versions (minus/regular)

---

## How This Causes O(1) GK Error

### Usage in `check_keldysh_normalization` (line 351):

```python
thermal_gr = gr_row.precise_convolution_left(thermal_dist, thermal_integral, self.dt,
                                             other_index=t1_pos,
                                             precomputed_sum=thermal_sum_right) * tau3 * 2
```

### Inside `precise_convolution_left` (lines 274-283):

```python
positive_index_precomp = other_index % precomputed_sum.data.shape[2]  # shape[2] = 2
precomputed_sum_row = precomputed_sum[positive_index_precomp:positive_index_precomp+1, :]

result_fact = self[-1:,:] * precomputed_sum_row
result_anal = self[-1:,:] * other_integral_for_reg

return (result_std + (- result_fact + result_anal))
```

**The Problem:**

1. `positive_index_precomp = other_index % 2` maps time index t1_pos (range 0 to N_t-1) to just {0, 1}
   - **ALL times map to only 2 values!**
   - Time dependence completely lost

2. `precomputed_sum_row` is the SAME for ~500 consecutive time indices
   - other_index=0 → index 0
   - other_index=1 → index 1
   - other_index=2 → index 0 (modulo 2)
   - other_index=3 → index 1
   - ...
   - **Only 2 distinct values for 1000 time points!**

3. Regularization mixes incompatible terms:
   ```
   - result_fact (uses extended grid integral from -2*T_max)
   + result_anal (uses standard grid integral from -T_max)
   ```
   - Different integration domains don't cancel properly
   - Leaves O(1) residual error

---

## Near-Diagonal Behavior

The error is especially bad near the diagonal (t ≈ t') because:

1. **Thermal distribution diverges:** f(τ) = -iT/sinh(πτT) → ∞ as τ → 0
   - At τ = 5×10⁻⁶, we get |f| = 6.357×10⁴

2. **Extended grid creates near-diagonal points:**
   - Standard grid: spacing dt = 0.0100 (t'' = -0.0100, -0.0200, ...)
   - Extended grid: spacing dt = 0.0100 but offset (t'' = -0.010005, -0.020010, ...)
   - When t' = -0.010010 and t'' = -0.010005: τ = 5×10⁻⁶ → huge f value!

3. **Wrong precomputed sum near diagonal:**
   - The sum should account for the singular behavior at each (t, t') pair
   - But it's using a FIXED sum (independent of t)
   - Near diagonal where t ≈ t', the integral structure changes dramatically
   - Using t-independent sum gives completely wrong answer

4. **Endpoint weighting mismatch:**
   - `result_std` uses trapezoidal rule: 0.5 weight at endpoints
   - `precomputed_sum` has inconsistent weighting (0.5 at t''=0, but wrong structure overall)
   - Regularization fails to cancel endpoint errors near diagonal

---

## Why This Causes 0.4 FDT Error and O(1) Normalization Error

**FDT (Fluctuation-Dissipation Theorem):**
- Relates g^K to f(τ) through equilibrium constraint
- Formula: g^K = g^R ⊗ f - f ⊗ g^A + (thermal corrections)
- Error in thermal convolution directly violates FDT
- **0.4 error magnitude** comes from wrong thermal sum at diagonal

**GK Normalization (Keldysh constraint):**
- Formula: [g^R, g^K] + g^R ⊗ g^K + g^K ⊗ g^A + (thermal terms) = 0
- Thermal terms should exactly cancel parts of the convolutions
- Wrong `thermal_sum` → wrong cancellation → O(1) residual
- **Especially bad near diagonal** where thermal effects are strongest

---

## The Correct Fix

### Step 1: Compute True 2D Thermal Sums

Need to compute for ALL (t, t') pairs, not just 1D vectors:

```python
# For each t_i in standard grid:
#   For each t_j in standard grid:
#     SUM_left[i, j] = ∫_{-2*T_max}^{t_i} f(t'', t_j) dt''
#     SUM_right[i, j] = ∫_{-2*T_max}^{t_j} f(t_i, t'') dt''
```

**Shape:** `(N_t, N_t)` full matrix

### Step 2: Use Extended Grid Properly

Currently only uses t''=0 row for f_sum_right - should use FULL extended grid:

```python
# Extended grid: [-2*T_max, -T_max) ∪ [-T_max, 0]
# For SUM_left[i, j]: integrate f(t'', t_j) over t'' ∈ [-2*T_max, t_i]
# For SUM_right[i, j]: integrate f(t_i, t'') over t'' ∈ [-2*T_max, t_j]
```

### Step 3: Handle Near-Diagonal Singularity

At diagonal (t_i = t_j), τ = 0 → f diverges.

**Options:**
1. **Mask diagonal:** Set f(t, t) = 0 (already done with 1e-6 threshold)
2. **Analytic integration:** Use limiting value lim_{τ→0} ∫f(τ)dτ
3. **Regularized sum:** Exclude diagonal contribution explicitly

### Step 4: Ensure Consistent Endpoint Weighting

- Trapezoidal rule: 0.5 weight at BOTH endpoints
- Must match between `result_std` and `precomputed_sum`
- Current code has inconsistent weighting → regularization fails

---

## Diagnostic Evidence

From `debug_thermal_sum.py` output:

```
thermal_sum_left (t=-1): max=2.211e+00
thermal_integral (t=-1): max=2.603e+00
Ratio (sum/integral): mean=0.71, max=1.00
```

**Expected:** If computed correctly, thermal_sum should be LARGER than thermal_integral
(extended grid -2*T_max to 0 vs standard grid -T_max to 0)

**Actual:** thermal_sum is SMALLER - confirms it's computing the wrong thing

**Plot evidence:**
- thermal_sum_left (blue) has INVERTED shape
- Should rise cumulatively, instead drops to -2.2
- Confirms wrong integration structure

---

## Conclusion

The O(1) diagonal GK error is caused by:

1. **Dimensional collapse:** Computing 1D vectors instead of 2D (t,t') matrices
2. **Lost time dependence:** Modulo 2 arithmetic maps 1000 times to just 2 values
3. **Wrong integration domain:** Only using last row (t''=0) instead of full extended grid
4. **Incompatible mixing:** Subtracting extended-grid and standard-grid integrals
5. **Near-diagonal singularity:** Extended grid offset creates τ≈10⁻⁶ points with huge f values

**Fix requires:** Rewriting `get_thermal_sum()` to compute proper 2D cumulative integrals over extended grid for all (t, t') pairs.
