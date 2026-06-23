# Issues with Diagonal g^K Element Evolution

## Critical Bugs Found

### 1. **CRITICAL: Boundary Condition Extrapolation Error**
**Location**: `usadel_keldysh_evolution.py:1432`

**Current Code**:
```python
gk_boundary = ( 2 * state.gk[-1, 0] - state.gk[-2, 1] )  # g^K(t, -infty)
```

**Problem**:
The extrapolation uses **inconsistent indices**:
- `state.gk[-1, 0]` = g^K(t_old, t_old - T_max)  [at t = t_old, t' = -∞]
- `state.gk[-2, 1]` = g^K(t_old - dt, t_old - T_max + dt)  [at t = t_old - dt, t' = -∞ + dt]

The formula mixes **different time slices** in the second index, creating an O(dt) error.

**Correct Fix**:
```python
gk_boundary = ( 2 * state.gk[-1, 0] - state.gk[-2, 0] )  # g^K(t, -infty)
```

This properly extrapolates along constant t' = -∞:
- Uses: g^K(t_old, -∞) and g^K(t_old - dt, -∞)
- Extrapolates to: g^K(t_new, -∞) ≈ 2·g^K(t_old, -∞) - g^K(t_old - dt, -∞)

**Impact**: This is an **O(dt) systematic error** in every timestep that accumulates during evolution.

---

### 2. **Diagonal Symmetrization Applied After Solver**
**Location**: `state_object_class.py:217`

**Code**:
```python
new_gk_diag = 1/2 * (new_gk_diag + new_gk_diag.involution())
```

**Analysis**:
- The diagonal g^K(t,t) should be **hermitian in Nambu space** by construction
- Applying `.involution()` performs: τ₃ @ g^† @ τ₃
- This symmetrization is applied **AFTER** the solver computes the diagonal

**Potential Issues**:
- If the solver is correct, this should be a no-op
- If this changes the result, it means the solver produces **non-hermitian diagonals**
- Could be **masking O(dt) errors** from the solver

**Recommendation**:
1. Check if `new_gk_diag - new_gk_diag.involution()` is non-zero
2. If non-zero, the solver has a bug that this symmetrization is hiding
3. This could be related to the boundary condition error in issue #1

---

### 3. **Missing Midpoint Rule Endpoint Corrections**
**Location**: Multiple places in `usadel_keldysh_evolution.py`

**Lines with Issues**:
- Lines 1114-1118 (g^R history convolution)
- Lines 1124-1127 (g^K history convolution, equation 1)
- Lines 1137-1142 (g^K history convolution, equation 2)
- Lines 1166-1176 (diagonal history convolutions)

**Problem Comments in Code**:
```python
#? Missing endpoint corrections: @ operator uses uniform weight for all points
#? For midpoint rule need: dt * (0.5*first + interior + 0.5*last)
#? Current: uses time+1:-1 to skip endpoints but doesn't apply 0.5 weight
```

**Issue**:
All convolution integrals using `@` operator apply **uniform dt weights**:
```
∫ f(t'') dt'' ≈ dt * Σ f(t'')
```

But for **midpoint/trapezoidal rule**, endpoints should have **0.5 weight**:
```
∫ f(t'') dt'' ≈ dt * (0.5·f(t_start) + Σ f(t_interior) + 0.5·f(t_end))
```

**Impact**: Each convolution has an **O(dt) error** at the endpoints.

---

## Summary of O(dt) Errors

| Issue | Location | Error Type | Impact |
|-------|----------|------------|--------|
| Boundary extrapolation | Line 1432 | O(dt) systematic | Accumulates every timestep |
| Diagonal symmetrization | state_object:217 | Masks solver bugs | Unknown, needs investigation |
| Midpoint corrections | Multiple lines | O(dt) per convolution | Multiple convolutions per step |

---

## Recommended Fixes

### Immediate (Critical):
1. **Fix boundary condition** in line 1432:
   ```python
   gk_boundary = 2 * state.gk[-1, 0] - state.gk[-2, 0]
   ```

### Diagnostic:
2. **Check diagonal symmetry**:
   - Before line 217, add:
   ```python
   asymmetry = np.max(np.abs((new_gk_diag - new_gk_diag.involution()).data))
   if asymmetry > 1e-10:
       print(f"WARNING: Diagonal asymmetry = {asymmetry:.4e}")
   ```

### Medium Priority:
3. **Add midpoint corrections** to all convolution integrals:
   - Modify convolutions to apply 0.5*dt weight at endpoints
   - This requires careful tracking of which points are endpoints in each integral

---

## Physical Validation Tests

After fixes, verify:
1. **FDT normalization**: Check `check_fdt()` errors decrease
2. **Energy conservation**: Gap evolution should be smoother
3. **Long-time behavior**: Errors shouldn't grow linearly with time
4. **dt convergence**: Gap(T) should converge as dt → 0 with expected rate

---

## Code References

- Diagonal evolution: `usadel_keldysh_evolution.py:1297` (`_compute_new_gk_complete`)
- Solver: `usadel_keldysh_evolution.py:1043` (`generalized_g_update_rule`)
- State update: `state_object_class.py:198` (`update_state_gk`)
- Main evolution: `usadel_keldysh_evolution.py:1444` (`_evolve_state_by_one_timestep`)
