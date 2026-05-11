# Bug Report: _compute_new_gr_row Function

## Errors Found (2026-05-07)

### 1. Incorrect Pauli matrix structure (Lines 316-317)
**Issue:** Unitary propagators use only τ₁, which only works for real gaps.

**Current:**
```python
unitary_propagator_L = cos(|Δ|Δt) exp(-ηΔt) τ₀ - i sin(|Δ|Δt) τ₁ exp(-ηΔt)
```

**Correct (Eq. 1964 from paper):**
```python
rotation_axis = (Δᵢ τ₂ - Δᵣ τ₁) / |Δ|
U_L = exp(-ηΔt/2) [cos(|Δ|Δt) τ₀ + i sin(|Δ|Δt) rotation_axis]
```

**Fix:**
```python
tau2 = NambuKeldyshTensor(1.0, pauli_channel=2)
Delta_r = np.real(gap_history[-1])
Delta_i = np.imag(gap_history[-1])
crt_gap_magnitude = np.abs(gap_history[-1])

if crt_gap_magnitude > 1e-12:
    rotation_axis = (Delta_i * tau2 - Delta_r * tau1) / crt_gap_magnitude
else:
    rotation_axis = NambuKeldyshTensor(0.0, pauli_channel=0)

unitary_propagator_L = (np.exp(-self.eta * self.delta_t / 2) *
                        (np.cos(crt_gap_magnitude * self.delta_t) * tau0 +
                         1j * np.sin(crt_gap_magnitude * self.delta_t) * rotation_axis))
```

### 2. Missing factor of 1/2 in η exponential (Lines 316-317)
**Issue:** exp(-ηΔt) should be exp(-ηΔt/2)

**Current:** `np.exp(-self.eta * self.delta_t)`
**Correct:** `np.exp(-self.eta * self.delta_t / 2)` (from Eq. 1964)

### 3. Wrong sign in U_L (Line 316)
**Issue:** Has -i but should be +i

**Current:** `- 1j * np.sin(...)`
**Correct:** `+ 1j * np.sin(...)` (from Eq. 1964)

### 4. Wrong sign in U_R^(-1) (Line 317)
**Issue:** Has +i but should be -i

**Current:** `+ 1j * np.sin(...)`
**Correct:** `- 1j * np.sin(...)` (from Eq. 1995)

**Fix:**
```python
unitary_propagator_R_inv = (np.exp(-self.eta * self.delta_t / 2) *
                             (np.cos(crt_gap_magnitude * self.delta_t) * tau0 -
                              1j * np.sin(crt_gap_magnitude * self.delta_t) * rotation_axis))
```

### 5. Wrong diagonal element formula (Line 310)
**Issue:** Should include unitary evolution

**Current:** `gr_diagonal_new = -gap_tensor[-1]`
**Correct:** `gr_diagonal_new = 2 * unitary_propagator_L * gap_tensor[-1]` (from Eq. 2325)

**Note:** This fix must be applied AFTER computing the correct unitary_propagator_L

## Root Cause
The unitary transformation equations in the paper had an error in the Pauli matrix commutation relations (τ₃τ₂ = -iτ₁, not +iτ₁). This was corrected on 2026-05-07, leading to the proper form:

τ₃Δ(t) = i(Δᵢ τ₂ - Δᵣ τ₁)

The code needs to be updated to reflect this correction.

## References
- Eq. 1964: U_L explicit form
- Eq. 1995: U_R^(-1) explicit form
- Eq. 2325: Diagonal element evolution
- Eq. 2290-2297: General discrete evolution
