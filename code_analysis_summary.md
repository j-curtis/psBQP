# Code Analysis Summary for psBQP-keldysh

**Analysis Date**: 2026-06-02
**Codebase**: Superconducting vortices - psBQP-Keldysh non-equilibrium solver
**Total Issues Analyzed**: 34

---

## Executive Summary

A comprehensive code review identified **34 potential issues** across indexing, sign errors, and incomplete implementations. After systematic verification with the developer:

- **4 Confirmed Bugs** requiring fixes (1 HIGH, 1 MEDIUM, 2 LOW priority)
- **4 Issues Needing Investigation** (physics verification needed)
- **26 Intentional Design Choices** (confirmed correct behavior)

**Key Finding**: Most asymmetries between retarded (g^R) and Keldysh (g^K) evolution are **intentional** due to different causality structures. The "weird τ₂ evolution" is most likely caused by τ₂ being solved via constraint (not evolution) equations for g^K, combined with potential convolution boundary issues.

---

## CATEGORY A: CONFIRMED BUGS (4 issues)

### Bug 1: Undefined Method in `__str__()`
**Priority: MEDIUM**

- **File**: `state_object_class.py`
- **Line**: 383-384
- **Status**: ✗ **BUG - Remove the call**

**Description**:
The `StateObject.__str__()` method calls `self.get_current_history()` which doesn't exist, causing AttributeError when trying to print or convert state to string.

**Impact**:
Affects debugging but not physics computation. Any code that calls `str(state)` or `print(state)` will crash.

**Fix**:
```python
# REMOVE lines 383-384:
current_history = self.get_current_history()
current_str = f"Current(t_final) = {current_history[-1]:.6f}"

# REPLACE with:
current_str = "Current tracking not yet implemented"
```

**User Comment**: Remove the call

---

### Bug 2: Undefined Variable `v_old_interior_4`
**Priority: HIGH**

- **File**: `usadel_keldysh_evolution.py`
- **Line**: 757
- **Status**: ✗ **BUG - Variable missing**

**Description**:
In the Type 6 operator assembly block (lines 723-778), line 757 attempts to use `v_old_interior_4` which is never defined. Only `v_old_interior_1` and `v_old_interior_3` are computed.

**Code**:
```python
# Line 755-757:
v_old_contribution = (v_old_boundary_1 + v_old_boundary_2 +
                     v_old_interior_1 +
                     v_old_interior_3 + v_old_interior_4)  # ← v_old_interior_4 undefined!
```

**Impact**:
Runtime NameError if Type 6 operators are ever used. May not be hit in practice if Type 6 is unused.

**Fix**:
1. Verify if Type 6 operators are used anywhere in the code
2. If used: Define `v_old_interior_4` following the pattern of `v_old_interior_1` and `v_old_interior_3`
3. If unused: Remove from sum and add comment explaining Type 6 is incomplete

**User Comment**: Variable missing

---

### Bug 3: Convolution Midpoint Rule Unverified
**Priority: HIGH**

- **File**: `nambu_keldysh_class.py`
- **Lines**: 200, 266
- **Status**: ✗ **BUG - Not verified**

**Description**:
Both `precise_convolution_left()` and `precise_convolution_right()` methods have TODO comments questioning whether the midpoint rule for the last term is correctly implemented:

```python
# Line 200:
def precise_convolution_left(self, other, other_integral, dt, other_index=-1):
    #TODO: check this is implemented appropriately, are we subtracting the last term in the sum?

# Line 266:
def precise_convolution_right(self, other, other_integral, dt, self_index = -1):
    #TODO: check the midpoint rule is implemented appropriately, are we subtracting the last term in the sum?
```

**Impact**:
Potential error in boundary treatment of thermal collision integrals and electromagnetic coupling terms. These convolutions are used extensively throughout evolution.

**Fix**:
1. Write unit tests for `precise_convolution_left/right` with analytical test cases
2. Verify boundary behavior matches theoretical expectations
3. Check that thermal FDT (fluctuation-dissipation theorem) is satisfied
4. Update or remove TODO comments based on findings

**User Comment**: Not verified - needs testing

---

### Bug 4: Misleading Comment
**Priority: LOW**

- **File**: `usadel_keldysh_evolution.py`
- **Lines**: 1261-1262
- **Status**: ✗ **BUG - Comment misleading**

**Description**:
Comment says "SAME SIGN as Type 1" but the signs are actually opposite:

```python
'type1_gap': {'L': -1j * gap_tensor},
'type2_gap': {'R': 1j * gap_tensor},  # SAME SIGN as Type 1  ← MISLEADING
```

Type1 has `-1j` while Type2 has `+1j` (opposite signs).

**Impact**:
Confuses code readers about intentional behavior. Documentation error only, no physics impact.

**Fix**:
```python
# Change line 1262 comment to:
'type2_gap': {'R': 1j * gap_tensor},  # Opposite sign vs Type 1 (matches g^R pattern)
```

**User Comment**: Comment misleading

---

## CATEGORY B: NEEDS INVESTIGATION (4 issues)

### Issue 1: V_old Derivative Sign Pattern in g^K
**Status: ? UNCLEAR - Need to check**

- **File**: `usadel_keldysh_evolution.py`
- **Lines**: 1306-1308

**Description**:
The sign pattern in the code is `(+) (-) (+) (+)`:
```python
v_old_deriv = ((1j/2) * tau3 * gk_last_row.shift(-1, axis=1)     # +
               - (1j/2) * gk_last_row.shift(-1, axis=1) * tau3    # -
               + (1j/2) * tau3 * gk_last_row                      # +
               + (1j/2) * gk_last_row * tau3)                     # +
```

But the comment (lines 1303-1305) says:
```python
# V_old correction (both terms MINUS)
# Keldysh: -(i/2)τ₃·g^K(t-δt,t') - (i/2)g^K(t-δt,t')·τ₃
#          -(i/2)τ₃·g^K(t-δt,t'-δt) - (i/2)g^K(t-δt,t'-δt)·τ₃  [MINUS on both]
```

**Next Steps**:
- Derive Crank-Nicolson discretization for ∂ₜg^K on paper
- Compare term-by-term with code implementation
- Resolve discrepancy between comment and code

**User Comment**: Need to verify if code or comment is correct

---

### Issue 2: Convolution Tensor Slicing
**Status: ? UNCLEAR - Need to trace**

- **File**: `usadel_keldysh_evolution.py`
- **Lines**: 1043, 1047

**Description**:
g^R and g^K use different tensor slicing in convolution terms:

```python
if g_type == 'r':
    # Line 1043
    convolution_term_1 += (left_term * solution_tensor[:-1]) @ right_term[time+1:-1, time]
elif g_type == 'k':
    # Line 1047
    convolution_term_1 += (left_term * solution_tensor[1:]) @ right_term[:time, time]
```

g^R excludes last element `[:-1]` while g^K excludes first element `[1:]`.

**Next Steps**:
- Trace through one timestep of evolution
- Verify `solution_tensor` indices align with `right_term` time indices
- Document indexing convention in comments

**User Comment**: Need to trace through indexing logic to verify alignment

---

### Issue 3: EM Local Coupling Sign
**Status: ? UNCLEAR - Check physics**

- **File**: `usadel_keldysh_evolution.py`
- **Lines**: 1144, 1266

**Description**:
Type2 electromagnetic local coupling has opposite signs between g^R and g^K:

```python
# g^R (line 1144):
'type2_em_local': {'R': -1j * (A_tensor * A_tensor) * tau3},

# g^K (line 1266):
'type2_em_local': {'R': 1j * (A_tensor * A_tensor) * tau3},  # SAME SIGN ← Comment says SAME but they're opposite!
```

**Next Steps**:
- Check electromagnetic Hamiltonian coupling terms in physics paper/derivation
- Verify sign conventions for A²·τ₃ term in Keldysh formalism
- Update comment to clarify if signs are correct or fix the code

**User Comment**: Need to verify against electromagnetic coupling equations

---

### Issue 4: EM-Thermal Loop Recent Fix
**Status: ? UNCLEAR - Review again**

- **File**: `usadel_keldysh_evolution.py`
- **Lines**: 1350, 1406

**Description**:
Comment indicates recent bug fix: `"NEW: Add missing dt_shift loop"`:

```python
for dt_prime_shift in [0, 1]:
    for dt_shift in [0, 1]:  # NEW: Add missing dt_shift loop
        dt_end = None if dt_shift == 0 else -dt_shift
        # ... thermal coupling terms ...
```

If this loop was missing before, the code may have been skipping thermal coupling terms.

**Next Steps**:
- Manually verify all 4 corners are covered: (t, t'), (t, t'+δt), (t-δt, t'), (t-δt, t'+δt)
- Check `dt_end` logic produces correct slice ranges
- Add assertion tests to verify 4-corner Crank-Nicolson averaging

**User Comment**: Should review this section more carefully

---

## CATEGORY C: CONFIRMED INTENDED BEHAVIOR (26 issues)

All of the following are **INTENTIONAL DESIGN CHOICES** and should NOT be changed:

### 1. Hardcoded Gap for Testing
- **Lines**: 1119, 1224
- **Status**: ✓ **INTENDED**
- **Behavior**: Gap is overridden with constant value 1.4563 for testing
- **User Comment**: Intentional for testing evolution with fixed gap value

### 2. Sandwich Terms Only in Evolution Equations
- **Lines**: 998-1002
- **Status**: ✓ **INTENDED**
- **Behavior**: Sandwich terms only added to matrix_row_1 and matrix_row_2 (evolution), not rows 3-4 (constraint)
- **Implication**: For g^K, τ₁ and τ₂ components (solved via constraint) don't receive sandwich terms
- **User Comment**: By design, sandwich terms only go into evolution equations

### 3. Sign Asymmetries Between g^R and g^K
All intentional due to different causality structures (retarded vs Keldysh):

#### Type2 Damping
- **Lines**: 1142 (g^R), 1264 (g^K)
- g^R: `-1j * self.eta * tau3`
- g^K: `+1j * self.eta * tau3`
- **Status**: ✓ **INTENDED - Different for R vs K**

#### M_R Derivative Correction
- **Lines**: 1160 (g^R), 1297 (g^K)
- g^R: `R1 = R1 - (1j/2) * tau3`
- g^K: `R1 = R1 + (1j/2) * tau3`
- **Status**: ✓ **INTENDED - R vs K differ**

#### Type5 EM Coupling R Term
- **Lines**: 1148 (g^R), 1277 (g^K)
- g^R: `'R': -1j * A_tensor * tau3`
- g^K: `'R': 1j * A_tensor * tau3`
- **Status**: ✓ **INTENDED - R vs K differ**

### 4. Shift Index Arithmetic
- **Lines**: 513, 545, 550, etc.
- **Status**: ✓ **INTENDED**
- **Behavior**: `shift(shift_index-1)` produces `shift(0)` for g^K and `shift(-2)` for g^R
- **User Comment**: Asymmetry is correct due to retarded vs Keldysh causality

### 5. Boundary Correction with Shift
- **Line**: 1175
- **Status**: ✓ **INTENDED**
- **Code**: `NambuKeldyshTensor([np.append(np.zeros(self.ntpoints-1),[1.0])]).shift(-1, axis=1)`
- **User Comment**: "shift zeroes out other elements, check nambu_keldysh_class implementation for details"

### 6. Return Statement Asymmetry
- **Lines**: 1092-1095
- **Status**: ✓ **INTENDED**
- **Behavior**: g^R appends then removes last element `[:-1]`, g^K returns as-is
- **User Comment**: g^R and g^K intentionally have different output sizes

### 7. Constraint Equation Sign Asymmetry
- **Lines**: 939-941
- **Status**: ✓ **INTENDED**
- **Code**:
  ```python
  left_matrix = tau3 * expansion_tensor + ...
  right_matrix = -tau3 * expansion_tensor + ...
  ```
- **User Comment**: Sign difference is correct for [tau3, gK] commutator structure

### 8. Current Computation Disabled
- **Line**: 1516
- **Status**: ✓ **INTENDED**
- **Code**: `current_new = 0  # state.get_current_at_time_t(...)`
- **User Comment**: Current is disabled until Stage 2 of the project

### 9. Loop Boundaries and Time Grid

#### Loop End = -1
- **Line**: 973
- **Status**: ✓ **INTENDED**
- **Code**: `loop_end = -1` with `range(ntpoints-2, -1, -1)`
- **User Comment**: Correct range - includes index 0

#### Backward Time Grid
- **Line**: 114
- **Status**: ✓ **INTENDED**
- **Code**: `self.time_grid = np.linspace(-self.tmax, 0, self.ntpoints)`
- **User Comment**: Backward time is correct convention for this formalism

### 10. Diagonal Correction Timing
- **Lines**: 1061-1074
- **Status**: ✓ **INTENDED**
- **Behavior**: Triggers at `time == ntpoints-1` for g^K
- **User Comment**: Correctly handles the last diagonal element

### 11. Matrix Slicing Patterns
- **Lines**: 550, 555
- **Status**: ✓ **INTENDED**
- **Code**: `l_operator[-2:-1, :-2]` vs `l_operator[-1:, :-1]`
- **User Comment**: Different slices correct for boundary vs interior terms

### 12. Thermal Distribution Slicing
- **Lines**: 1338-1339
- **Status**: ✓ **INTENDED**
- **Code**: `dt_end = None if dt_shift == 0 else -dt_shift`
- **User Comment**: Correctly implements 4-corner Crank-Nicolson averaging

### 13. Thermal-Gap Coupling Signs
- **Line**: 1340
- **Status**: ✓ **INTENDED**
- **Pattern**: `cn_factor * -2 * (-1j * ... + 1j * ...)`
- **User Comment**: Correct commutator/anticommutator structure

### 14. Old Comments and TODOs
- **Lines 1318-1331**: ✓ **INTENDED - Old comments** (outdated, code is correct)
- **Self-energy stubs**: ✓ **INTENDED - Future work**
- **Normalization check TODOs (223, 264)**: ✓ **INTENDED - Not used**
- **Unused returns gr_tau, gk_tau (218)**: ✓ **INTENDED - Future use**

---

## Implementation Priorities

### Priority 1: Immediate Fixes (Runtime Errors)
1. **Bug 2**: Fix undefined `v_old_interior_4` (1 line fix)
2. **Bug 1**: Remove `get_current_history()` call (2 line fix)

### Priority 2: High Priority (Physics Accuracy)
3. **Bug 3**: Verify convolution midpoint rule (testing + possible fix)
4. **Issue 2**: Verify convolution tensor slicing (verification)
5. **Issue 4**: Review EM-thermal loop fix (verification)

### Priority 3: Medium Priority (Verification)
6. **Issue 1**: Resolve V_old sign pattern vs comment (verification)
7. **Issue 3**: Check EM coupling signs (physics verification)

### Priority 4: Low Priority (Documentation)
8. **Bug 4**: Update misleading comment (1 line fix)

---

## Root Cause Analysis: "Weird τ₂ Evolution"

Based on the systematic analysis, the most likely causes are:

### For g^K (gk2):
1. **Constraint vs Evolution Structure** (INTENDED but noteworthy):
   - τ₂ component is solved via **constraint equation** (matrix_row_4), not evolution equation
   - Sandwich terms are **NOT applied** to constraint equations (only rows 1-2)
   - User confirms this is correct design, but it may explain different behavior

2. **Potential Physics Issues**:
   - **Bug 3**: Convolution midpoint rule - if wrong, affects all thermal coupling terms
   - **Issue 1**: V_old sign pattern - if incorrect, affects time derivatives specifically
   - **Issue 2**: Convolution slicing - if misaligned, creates indexing errors

### For g^R (gr2):
- τ₂ component IS solved via evolution equation (matrix_row_2)
- Should receive all terms including sandwich terms
- Less likely to have structural issues compared to gK

### Recommended Diagnostic Approach:
1. Test with sandwich terms disabled → see if τ₂ behavior changes
2. Verify FDT (fluctuation-dissipation theorem) is satisfied for τ₂ component
3. Check τ₂ evolution against analytical test case with known solution
4. Compare τ₂ vs τ₁ evolution to isolate asymmetric behavior
5. Fix Bug 3 (convolution midpoint) and retest τ₂ evolution

---

## Conclusion

The codebase is largely well-structured with intentional design choices for the Keldysh formalism. The main action items are:

**Immediate:**
- Fix 2 runtime bugs (Bug 1, Bug 2)

**High Priority:**
- Verify convolution boundary treatment (Bug 3)
- Investigate 4 unclear issues requiring physics verification

**Most Likely Source of τ₂ Issues:**
- Convolution midpoint rule (Bug 3) combined with constraint equation structure for g^K

**Code Quality:**
- Update misleading comments
- Consider adding more comprehensive tests for edge cases
- Document indexing conventions more explicitly
