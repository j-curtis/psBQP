# Critical Bug: Wrong Integration Limits in thermal_integral

## The Smoking Gun

**File:** `usadel_keldysh_evolution.py`, line 316

```python
tau_lower = -self.tmax * 2 - t_j * 0 #* remove -t_j since assuming -infinity actually
```

**The bug:** `t_j` is multiplied by **0**, making the lower limit **constant** instead of t'-dependent!

---

## What the Code Computes vs What It Should Compute

### Docstring (lines 295-297) says:

```python
F(t, t') = ∫_{-T_max - t'}^{t - t'} f(τ) dτ
         = F_full(t - t') - F_full(-T_max - t')
```

**Lower limit:** `-T_max - t'` (depends on t')

### Code actually computes:

```python
tau_lower = -2*T_max  # CONSTANT (t_j multiplied by 0!)
F(t, t') = F_full(t - t') - F_full(-2*T_max)
         = ∫_{-2*T_max}^{t - t'} f(τ) dτ
```

**Lower limit:** `-2*T_max` (CONSTANT, independent of t')

---

## The Error

The difference between what's computed and what should be computed:

```
Error(t, t') = F_full(-T_max - t') - F_full(-2*T_max)
             = ∫_{-2*T_max}^{-T_max - t'} f(τ) dτ
```

**Key properties of this error:**

1. **Depends on t'** but is **independent of dt** (for fixed T_max)
   - Explains user observation: "disrepancy around zero seems to be delta_t independent"

2. **Especially large near t' = 0:**
   ```
   Error|_{t'=0} = ∫_{-2*T_max}^{-T_max} f(τ) dτ ≈ constant
   ```
   - Explains user observation: "this probably suggests that we are integrating with wrong limits"

3. **O(1) magnitude** because it's integrating over a finite interval of f(τ)
   - Explains the O(1) GK normalization error and 0.4 FDT error

---

## Why This Is Wrong

### Physical setup:
- Time grid: t'' ∈ [-T_max, 0], t' ∈ [-T_max, 0]
- Thermal distribution: f(t'', t') = f(t'' - t') depends only on τ = t'' - t'
- Need integral: ∫_{-T_max}^{t''} f(t'', t') d(t'')

### Change of variables to τ = t'' - t':
```
∫_{-T_max}^{t''} f(t'' - t') d(t'') = ∫_{-T_max - t'}^{t'' - t'} f(τ) dτ
```

**Lower limit in τ-space:** `-T_max - t'` (depends on t')

This matches the docstring but NOT the code!

---

## How This Causes dt-Independent Error Near t'=0

In the regularization scheme (precise_convolution), we compute:

```
result = result_std - result_fact + result_anal
```

where:
- `result_anal = g(t,t) * thermal_integral[t, t']`
- `thermal_integral[t, t']` should be ∫_{-T_max - t'}^{t - t'} f(τ) dτ
- But code uses ∫_{-2*T_max}^{t - t'} f(τ) dτ

The error in thermal_integral propagates directly to the final result.

**Near t' = 0 (end of time grid):**
```
Error = g(t,t) * [∫_{-2*T_max}^{t} - ∫_{-T_max}^{t}] f(τ) dτ
      = g(t,t) * ∫_{-2*T_max}^{-T_max} f(τ) dτ
      ≈ g(t,t) * constant
```

This is **dt-independent** and **O(1)** in magnitude!

---

## The Fix

### Option 1: Use t'-dependent lower limit (correct physics)

```python
# Line 316 - CURRENT (WRONG):
tau_lower = -self.tmax * 2 - t_j * 0 #* remove -t_j since assuming -infinity actually

# CORRECTED:
tau_lower = -self.tmax - t_j  # Proper lower limit for finite grid
```

This makes `thermal_integral` consistent with the docstring and the physics.

### Option 2: Use extended grid with t'-dependence

If we want to use -2*T_max for better -∞ approximation:

```python
tau_lower = -self.tmax * 2 - t_j  # Extended grid, still t'-dependent
```

**Must also update thermal_sum to use the SAME lower limit!**

---

## Why The Comment Is Misleading

The comment says:
```python
#* remove -t_j since assuming -infinity actually
```

This suggests that setting the lower limit to a constant (-2*T_max) is "assuming -infinity."

**This is wrong!** Even when approximating -∞, the integral limits in τ-space must respect the relationship τ = t'' - t':

```
∫_{-∞}^{t''} f(t'', t') d(t'') = ∫_{-∞}^{t'' - t'} f(τ) dτ
```

The lower limit -∞ in t''-space becomes -∞ in τ-space, which is fine.

But on a finite grid starting at -T_max (or -2*T_max), the lower limit in τ-space is:
```
τ_lower = t''_min - t' = -T_max - t'  (or -2*T_max - t')
```

**The t'-dependence is essential!** It cannot be removed by multiplying by 0.

---

## Verification

This bug explains ALL the observed symptoms:

✓ O(1) error in GK normalization
✓ 0.4 error in FDT
✓ Error is dt-independent
✓ Error is especially bad near t' = 0
✓ thermal_sum and thermal_integral don't match (they use different lower limits)
✓ Regularization fails because it mixes incompatible integrals

---

## Action Items

1. **Fix line 316** to include t_j dependence:
   ```python
   tau_lower = -self.tmax - t_j  # Standard grid
   # OR
   tau_lower = -self.tmax * 2 - t_j  # Extended grid (if using extended approach)
   ```

2. **Update thermal_sum** to use the same lower limit as thermal_integral

3. **Recompute thermal_integral** after the fix

4. **Verify** that GK normalization error and FDT error are reduced

5. **Check** that error now scales properly with dt (should go to 0 as dt → 0)
