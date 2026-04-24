# Testing Strategy for psBQP Keldysh Implementation

## Status Overview

### ✅ Completed & Tested
- **NambuKeldyshTensor** - User confirmed working

### ⚠️ Implemented but Untested
- **StateObject** - Partially complete, needs testing
- **EquilibriumSolver** - Wrapper around old code, needs testing
- **UsadelKeldyshEvolution** - Core evolution implemented, needs testing

### ❌ Not Implemented
- **SelfEnergy classes** - All methods are `pass` (Stage 4 feature)

---

## Outstanding TODOs

### Critical TODOs (Block Testing)
1. **`state_object_class.py:79`** - Check factor of 4 in gap equation
2. **`usadel_keldysh_evolution.py:381,386`** - Implement thermal convolution (term6, term7)

### Minor TODOs (Documentation/Optimization)
3. **`state_object_class.py:120`** - Keep better track of rows and columns
4. **`equilibrium_class.py:75`** - Can load pre-computed equilibrium (optimization)

---

## Testing Strategy - Bottom-Up Approach

### Phase 1: StateObject Testing (Foundation)
**Dependencies:** NambuKeldyshTensor ✓

**Test 1.1: Basic Initialization**
- Create StateObject with random gr, gk tensors
- Verify shapes are stored correctly
- Check grid parameter extraction (T_max, dt)
- Expected: No errors, correct attribute values

**Test 1.2: Advanced Green's Function**
- Create StateObject with known gr
- Call `state._r2a()`
- Verify g^A = -(g^R)^† using involution
- Expected: Correct transformation

**Test 1.3: Gap Extraction**
- Create StateObject with known gk diagonal
- Call `state.get_gap_history()`
- **BLOCKER:** Verify factor of -λ/4 is correct (TODO line 79)
- Expected: Gap values extracted from diagonal

**Test 1.4: Update State Object**
- Create initial StateObject
- Generate new row, diagonal for gr and gk
- Call `state.update_state_object()`
- Verify sliding window works (oldest removed, newest added)
- Check causality: gr column is zero
- Check gk column computed via τ₃ transformation
- Expected: State updated correctly, shapes preserved

**Test 1.5: String Representation**
- Create StateObject
- Call `str(state)`
- **ISSUE:** `get_current_history()` returns None, will crash
- Expected: Fix or skip current in __str__ for now

---

### Phase 2: EquilibriumSolver Testing (Foundation)
**Dependencies:** NambuKeldyshTensor ✓, Old equilibrium code

**Test 2.1: Initialization**
- Create EquilibriumSolver with grid/system parameters
- Verify UsadelEvolution (old code) is created
- Check omega grid is passed correctly
- Expected: No errors, old solver initialized

**Test 2.2: Compute Equilibrium g^R**
- Call `eq_solver.compute_equilibrium_gr(T=0.3, Q=0)`
- Verify returns NambuTensor (old class)
- Check gap converges to expected BCS value
- Expected: Equilibrium solution found

**Test 2.3: Thermal Occupation**
- Call `eq_solver._get_thermal_occupation(T=0.3)`
- Verify f(ω) = tanh(ω/2T)
- Check shape matches omega grid
- Expected: Correct thermal distribution

**Test 2.4: Fourier Transform to Two-Time**
- Compute equilibrium gr_eq, gk_eq
- Call `eq_solver.fourier_transform_to_two_time(gr_eq, gk_eq)`
- Verify output shape is (2, 2, N_t, N_t) for t < 0
- Check translational invariance: g(t,t') = g(t-t')
- Expected: Correct two-time representation

---

### Phase 3: UsadelKeldyshEvolution Testing (Integration)
**Dependencies:** StateObject ✓, EquilibriumSolver ✓

**Test 3.1: Initialization and Grid Setup**
- Create UsadelKeldyshEvolution with parameters
- Verify time grid: `[-T_max, 0]` with N_t points
- Verify omega grid from FFT of extended time domain
- Check BCS constants (gap_constant ≈ 1.134, ratio ≈ 1.764)
- Check BCS coupling λ computed correctly
- Expected: Grids and constants correct

**Test 3.2: Thermal Distribution**
- Call `evolution.get_thermal_occupation(T=0.3)`
- Verify shape is (N_t, N_t)
- Check antisymmetry: F(τ) = -F(-τ)
- Check F(0) is purely imaginary
- Expected: Correct thermal distribution in two-time

**Test 3.3: Initial State Generation**
- Call `evolution.generate_initial_state(Q=0)`
- Verify StateObject returned
- Check gr, gk shapes are (2, 2, N_t, N_t)
- Verify equilibrium gap is reasonable (Δ ≈ 1.764 * T_c * sqrt(1 - T/T_c))
- Check normalization: gr @ gr ≈ -1
- Check Keldysh relation: gk ≈ gr @ f - f @ ga
- Expected: Valid equilibrium initial state

**Test 3.4: Single Time Step Evolution (gr)**
- Create mock StateObject
- Call `evolution._compute_new_gr_column(state)`
- Verify returns (gr_new, gr_diag_new)
- Check shapes are correct
- Check no NaN or Inf values
- Expected: New gr computed without errors

**Test 3.5: Single Time Step Evolution (gk)**
- Create mock StateObject
- Initialize thermal_dist
- Call `evolution._compute_new_gk_column(state)`
- Verify returns (gk_new, gk_diag_new)
- **BLOCKER:** term6 and term7 are zero (TODO lines 381, 386)
- Expected: New gk computed (note thermal terms missing)

**Test 3.6: Combined Time Step**
- Create initial StateObject
- Call `evolution._evolve_state_by_one_timestep(state, 0)`
- Verify state is updated in-place
- Check gap and current returned
- Expected: State evolves one timestep

**Test 3.7: Multi-Step Evolution**
- Generate initial state from equilibrium
- Call `evolution.real_time_evolution(initial_state, num_timesteps=10)`
- Verify gap time series has length 10
- Check gap remains approximately constant (no external field)
- Check for stability (no divergence)
- Expected: Evolution runs for multiple steps without crash

---

## Critical Issues to Address Before Full Testing

### Issue 1: Gap Equation Normalization (TODO line 79)
**Location:** `state_object_class.py:78`
**Problem:** Factor of -λ/4 needs verification
**Action Required:**
- Derive correct normalization from BCS theory
- Verify against equilibrium gap equation
- Update formula if needed

### Issue 2: Thermal Convolution Terms (TODO lines 381, 386)
**Location:** `usadel_keldysh_evolution.py:381, 386`
**Problem:** term6 and term7 in gk evolution are set to zero
**Impact:** Keldysh evolution will not include thermal scattering
**Action Required:**
- Implement convolution: Σ_k F[i,k] * g^A[k,j]
- Implement convolution: Σ_k g^R[i,k] * F[k,j]
- Use @ operator for proper time-index contraction

### Issue 3: Current Calculation Not Implemented
**Location:** `state_object_class.py:83-85`
**Problem:** `get_current_history()` returns None
**Impact:** `__str__` will crash, Stage 2 feature unavailable
**Action Required:**
- Either implement current calculation
- Or fix `__str__` to skip current for now

### Issue 4: SelfEnergy Classes Empty
**Location:** `self_energy_class.py` (entire file)
**Problem:** All methods are `pass`
**Impact:** Cannot use scattering mechanisms (Stage 4)
**Action Required:** None for now (Stage 4 feature)

---

## Recommended Testing Order

1. **Fix Issue 3** (Current in __str__) - 5 minutes
   - Quick fix to prevent crashes

2. **Test Phase 1** (StateObject) - 30 minutes
   - Fundamental class, must work first
   - Resolve Issue 1 (gap normalization) during testing

3. **Test Phase 2** (EquilibriumSolver) - 1 hour
   - Depends on old code working
   - Critical for initial states

4. **Test Phase 3.1-3.3** (Evolution setup & initial state) - 1 hour
   - Test everything before time evolution

5. **Address Issue 2** (Thermal convolution) - 1-2 hours
   - Required for correct gk evolution

6. **Test Phase 3.4-3.7** (Time evolution) - 1 hour
   - Full integration test

---

## Success Criteria

### Minimal Working Model (Stage 1)
- ✅ NambuKeldyshTensor operations work
- [ ] StateObject stores and updates Green's functions
- [ ] EquilibriumSolver generates valid initial states
- [ ] Time evolution runs for 10+ steps without crash
- [ ] Gap stays approximately constant in equilibrium
- [ ] No NaN or Inf values appear

### Validation Checks
- [ ] Normalization: g^R @ g^R ≈ -1
- [ ] Keldysh relation: g^K ≈ g^R @ f - f @ g^A (equilibrium)
- [ ] Gap at T=0 ≈ 1.764 * T_c
- [ ] Gap at T=T_c ≈ 0
- [ ] Thermal distribution antisymmetry: F(-τ) = -F(τ)

---

## Notes

- All tests assume no external field (external_field=None)
- Vector potential evolution (Stage 2) not included yet
- Self-energy scattering (Stage 4) not included yet
- Focus on equilibrium dynamics first
