# Keldysh Non-Equilibrium Implementation TODO

Implementation plan for psBQP (pseudo-spin Bogoliubov-Quasiparticle) Keldysh solver.

## Status: Stage 1 - Minimal Working Model

### ✅ Completed
- [x] NambuKeldyshTensor class fully implemented
  - [x] Multiplication operators with broadcasting
  - [x] Convolution operator (@)
  - [x] Arithmetic operations
  - [x] Matrix operations (conj, transpose, complete_transpose, involution, determinant)
  - [x] Pauli trace and string representation

---

## Stage 1: Minimal Working Model

**Goal:** Get a basic simulation running end-to-end without vector potential or self-energies.

### 1.1 StateObject Class
**File:** `state_object_class.py`

- [ ] **`__init__(self, gr, gk, temporal_grid_params, cutoff)`**
  - Store gr (retarded Green's function) - NambuKeldyshTensor
  - Store gk (Keldysh Green's function) - NambuKeldyshTensor
  - Store temporal_grid_params (dict with time grid info)
  - Store cutoff (energy cutoff for gap normalization)
  - Initialize internal storage for computed gaps

- [ ] **`r2a(self)` - Compute Advanced Green's Function**
  - Compute g^A from g^R using involution
  - Formula: `g^A = -(g^R)^†`
  - Implementation: `return -self.gr.conj().complete_transpose()`
  - Return: NambuKeldyshTensor

- [ ] **`copy(self)` - Deep Copy**
  - Return deep copy of StateObject with all data copied

- [ ] **`_update_state_object(self, gr_column, gk_column, time_index)`**
  - **Purpose:** Insert new gr and gk data at specified time_index
  - **Inputs:**
    - gr_column: Row/column vector of new gr data
    - gk_column: Row/column vector of new gk data
    - time_index: Which time point to update
  - **Implementation:**
    - Update `self.gr.data[:, :, time_index, :]` with gr_column
    - Update `self.gr.data[:, :, :, time_index]` with gr_column (symmetric)
    - Same for gk
  - **Key:** Recycles same StateObject instance (memory efficient)
  - No return (modifies in-place)

- [ ] **`get_gap(self)`**
  - **Purpose:** Extract superconducting gap with proper normalization
  - **Formula:** `Delta(t) = (i/2) * Tr[tau_y * g^K(t,t)] / normalization`
  - **Normalization:** Account for cutoff
  - **Steps:**
    1. Extract diagonal: `gk_diag = self.gk[:, :, range(N_t), range(N_t)]`
    2. Compute Y-trace: `gap_raw = gk_diag.trace(pauli_index=2)`
    3. Apply cutoff normalization: `gap = gap_raw / norm_factor`
  - Return: Array of gap values at each time

- [ ] **`check_normalization(self)`**
  - **Purpose:** Verify normalization: `g^R @ g^R = -1`
  - **Implementation:**
    - Compute: `gr_squared = self.gr @ self.gr`
    - Check: `gr_squared ≈ -Identity`
    - Return: Maximum deviation from -1

- [ ] **`check_keldysh_relation(self)`**
  - **Purpose:** Verify: `g^K = g^R @ f - f @ g^A`
  - **Steps:**
    1. Compute g^A: `ga = self.r2a()`
    2. Need thermal f (store in StateObject or pass as argument)
    3. Compute: `gk_check = self.gr @ f - f @ ga`
    4. Compare with `self.gk`
  - Return: Maximum deviation
  - **Note:** May need to store f in StateObject for this check

- [ ] **`__str__(self)`**
  - **Purpose:** Print gap and current at all times
  - **Format:**
    ```
    StateObject:
      Time points: N_t
      Gap values: [Δ(t0), Δ(t1), ...]
      Currents: [I(t0), I(t1), ...] (Stage 2+)
    ```

**Testing:** Create StateObject with dummy data, verify r2a, normalization

---

### 1.2 EquilibriumSolver Class
**File:** `equilibrium_class.py`

- [ ] **`__init__(self, grid_parameters, system_parameters)`**
  - Store grid_parameters (omega grid, cutoff)
  - Store system_parameters (T_c, coupling, temperature)
  - Create omega grid array

- [ ] **`compute_self_cons_equilibrium(self, temperature, Q=0.0)`**
  - **Purpose:** Compute equilibrium gr, gk for given T and phase gradient Q
  - **Steps:**
    1. Generate thermal occupation: `f_eq(omega) = tanh(omega / 2T)`
    2. Initial guess for gap: `Delta_0 = 1.764 * T_c * sqrt(1 - T/T_c)` (BCS)
    3. Call `_iterate_self_consistency(Delta_0, f_eq, ...)`
    4. Compute equilibrium gr from converged Delta
    5. Compute equilibrium gk: `g^K = g^R @ f - f @ g^A`
  - Return: (gr_eq, gk_eq) as NambuKeldyshTensor objects

- [ ] **`_solve_gap_equation(self, gr, f, delta_guess)`**
  - **Purpose:** Solve BCS gap equation
  - **Formula:** `Delta = λ * integral[ Tr_Y[g^R @ f] ]` (simplified)
  - Use iterative solver or fixed-point iteration
  - Return: Converged Delta (complex scalar)

- [ ] **`_iterate_self_consistency(self, delta, f, max_iterations=100, tolerance=1e-6)`**
  - **Purpose:** Iteratively solve for self-consistent gr and Delta
  - **Loop:**
    1. Construct h^R from current delta (use solver._construct_hr)
    2. Solve for gr: `g^R = -h^R / sqrt(h^R @ h^R)` (or matrix inversion)
    3. Compute new delta from gap equation
    4. Check convergence: `|delta_new - delta_old| < tolerance`
    5. Repeat
  - Return: (gr_converged, delta_converged)

**Testing:**
- Equilibrium gap at T=0 should be ≈ 1.764 * T_c
- Gap at T=T_c should vanish
- Self-consistency should converge in < 50 iterations

---

### 1.3 UsadelKeldyshEvolution - Grid Setup
**File:** `usadel_keldysh_evolution.py`

- [ ] **`__init__(self, grid_parameters, system_parameters, optimization_parameters=None)`**
  - Store grid_parameters dict
  - Store system_parameters dict
  - Call `_generate_time_grid()`
  - Create omega grid from grid_parameters
  - Store eta (broadening parameter)
  - Store critical_temperature, temperature

- [ ] **`_generate_time_grid(self)`**
  - **Purpose:** Create time array for evolution
  - **Reads from grid_parameters:**
    - `time_sampling`: Number of time points
    - `time_duration`: Total time duration
  - **Creates:**
    - `self.time_grid = np.linspace(0, time_duration, time_sampling)`
    - `self.dt = time_grid[1] - time_grid[0]`
    - `self.temporal_grid_params = {...}` (dict with time info)

- [ ] **Static BCS helper methods**
  - **`get_bcs_gap_constant() -> float`**
    - Return: `2 * np.exp(np.euler_gamma) / np.pi ≈ 1.134`
  - **`get_bcs_ratio() -> float`**
    - Return: `Delta(0)/T_c ≈ 1.764` (BCS ratio)
  - **`_get_BCS_coupling(self) -> float`**
    - Compute BCS coupling constant λ from critical_temperature
    - Return: λ

**Testing:** Verify time grid, check BCS constants match theory

---

### 1.4 Hamiltonian and Gap Tensor
**File:** `usadel_keldysh_evolution.py`

- [ ] **`compute_gap_tensor(self, gap_number)`**
  - **Purpose:** Convert scalar gap to Nambu tensor
  - **Input:** gap_number (complex scalar or array)
  - **Formula:** Gap enters as tau_y component: `-i * Delta * tau_y`
  - **Implementation:**
    ```python
    gap_matrix = -1j * gap_number * get_pauli_matrix('y')
    # Handle broadcasting if gap_number is array
    return NambuKeldyshTensor(gap_matrix)
    ```

- [ ] **`_construct_hr(self, Q, delta, sigma_r=None)`**
  - **Purpose:** Build retarded Hamiltonian h^R (Stage 1: without vector potential)
  - **Formula:** `h^R = (omega - Q) * tau_3 - Delta * tau_y + i*eta`
  - **Steps:**
    1. Kinetic term: `(omega - Q) * tau_3`
    2. Gap term: Use `compute_gap_tensor(delta)`
    3. Broadening: `+i*eta * Identity`
    4. (sigma_r for Stage 4)
  - **Note:** Some terms diagonal in omega, some may need convolution structure
  - Return: h^R as NambuKeldyshTensor

- [ ] **`_get_thermal_occupation(self, temperature)`**
  - **Purpose:** Generate thermal distribution tensor
  - **Formula:** `f(omega) = tanh(omega / 2T)`
  - **Output shape:** Must match state dimensions for Keldysh relation
  - **Implementation:**
    ```python
    f_array = np.tanh(omega_grid / (2 * temperature))
    # Create as Nambu tensor (identity in Nambu space)
    return NambuKeldyshTensor(f_array, pauli_channel=0)
    ```

**Testing:** Verify h^R structure, gap tensor has Y-component only, f(omega) correct

---

### 1.5 Initial State Generation
**File:** `usadel_keldysh_evolution.py`

- [ ] **`_generate_initial_state(self)`**
  - **Purpose:** Generate translationally invariant equilibrium for t < 0
  - **Steps:**
    1. Create EquilibriumSolver instance
    2. Call `equilibrium_solver.compute_self_cons_equilibrium(T, Q=0)`
    3. Get equilibrium (gr_eq, gk_eq) - time-independent tensors
    4. **Generate full time-dependent arrays:**
       - Target shape: `(2, 2, N_t, N_t)`
       - For translational invariance: `gr(t, t') = gr(t - t')`
       - Fill lower triangular part (t > t') with equilibrium
       - Upper triangular is zero (causality)
    5. **Truncate to t, t' < 0:**
       - Take lower-left quarter of full time grid
       - This gives initial conditions for evolution
    6. Create StateObject with initial data
  - **Comment for Stage 4:** "Can load pre-computed initial states if available"
  - Return: StateObject

**Testing:**
- Verify shape is (2, 2, N_t/2, N_t/2) or similar
- Check translational invariance: `gr[i,j,t,tp] = gr[i,j,t-tp,0]`
- Verify normalization and Keldysh relation hold

---

### 1.6 Time Evolution - Derivatives
**File:** `usadel_keldysh_evolution.py`

- [ ] **`_compute_dtgr(self, state, time_index, external_field=None)`**
  - **Purpose:** Compute time derivative ∂_t g^R at given time
  - **Equation:** `∂_t g^R = i [h^R, g^R]` (commutator)
  - **Steps:**
    1. Extract current gr from state at time_index
    2. Construct h^R (no external_field in Stage 1)
    3. Compute commutator: `i * (hr * gr - gr * hr)` using `*` operator
    4. Extract the row/column corresponding to time_index
  - **Return:** Row vector (1D slice) to update gr matrix
  - **Note:** May need special handling for causality structure

- [ ] **`_compute_dtgk(self, state, time_index, external_field=None)`**
  - **Purpose:** Compute time derivative ∂_t g^K at given time
  - **Equation:** `∂_t g^K = i [h^R, g^K]` (Stage 1: no collision terms)
  - **Steps:**
    1. Extract current gk from state
    2. Construct h^R
    3. Compute commutator with gk
    4. Extract row/column
  - **Return:** Row vector to update gk matrix
  - **Note:** Collision integrals added in Stage 4 with self-energies

**Testing:** Verify dtgr and dtgk have correct shapes, check signs

---

### 1.7 Time Evolution - Integration
**File:** `usadel_keldysh_evolution.py`

- [ ] **`_evolve_state_by_one_timestep(self, state, time_index, external_field=None)`**
  - **Purpose:** Update state by one timestep
  - **Integration scheme:** Start with Forward Euler, upgrade to RK4 if needed
  - **Steps:**
    1. Compute `dtgr = _compute_dtgr(state, time_index, external_field)`
    2. Compute `dtgk = _compute_dtgk(state, time_index, external_field)`
    3. Update:
       - `gr_new = gr_old + self.dt * dtgr`
       - `gk_new = gk_old + self.dt * dtgk`
    4. Call `state._update_state_object(gr_new, gk_new, time_index)`
  - **Note:** For stability, may need RK4 or adaptive stepping
  - Return: Updated StateObject (modified in-place)

- [ ] **`real_time_evolution(self, initial_state, num_timesteps, external_field=None)`**
  - **Purpose:** Main evolution loop
  - **Steps:**
    1. Initialize storage: `gaps = []`, `currents = []` (currents Stage 2+)
    2. **Loop:** `for i in range(num_timesteps):`
       - Call `_evolve_state_by_one_timestep(state, i, external_field)`
       - Extract gap: `gap_i = state.get_gap()`
       - Append to gaps array
    3. Return: `(final_state, np.array(gaps), np.array(currents))`
  - **Note:** external_field function signature: `field(t) -> value`

**Testing:**
- Run for 10 timesteps, verify state updates
- Check energy conservation (if no external field)
- Verify no NaN or Inf values

---

### 1.8 Complete Workflow Test
**File:** Create `test_stage1.py` or notebook

- [ ] **Minimal working example**
  ```python
  import numpy as np
  from usadel_keldysh_evolution import UsadelKeldyshEvolution

  # Define grids
  grid_params = {
      'omega_sampling': 51,
      'cutoff': 10.0,
      'time_sampling': 100,
      'time_duration': 10.0
  }

  # System parameters
  system_params = {
      'critical_temperature': 1.0,
      'temperature': 0.5,
      'eta': 0.01
  }

  # Initialize solver
  solver = UsadelKeldyshEvolution(grid_params, system_params)

  # Generate initial state
  print("Generating initial equilibrium state...")
  initial_state = solver._generate_initial_state()
  print(f"Initial gap: {initial_state.get_gap()}")

  # Run evolution (no external field in Stage 1)
  print("Running time evolution...")
  final_state, gaps, currents = solver.real_time_evolution(
      initial_state,
      num_timesteps=50,
      external_field=None
  )

  # Check results
  print(f"Final gap: {gaps[-1]}")
  print(f"Gap should be constant: std = {np.std(gaps)}")

  # Plot
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(solver.time_grid[:len(gaps)], gaps)
  plt.xlabel('Time')
  plt.ylabel('Gap Δ(t)')
  plt.title('Stage 1: Equilibrium Evolution')
  plt.show()
  ```

- [ ] **Verify expected behavior:**
  - Gap should remain approximately constant (equilibrium)
  - No divergences or instabilities
  - Normalization check passes
  - Keldysh relation approximately satisfied

**Success criteria:** Code runs without errors, gap stays near equilibrium value

---

## Stage 2: Vector Potential Evolution

**Goal:** Add external vector potential A(t) and current calculation.

### 2.1 Current Calculation
**File:** `state_object_class.py`

- [ ] **`get_current(self, Q=None)`**
  - **Purpose:** Compute supercurrent from Green's functions
  - **Formula:** TBD (involves derivatives or phase gradients)
  - **Implementation:** Extract from appropriate traces of gk
  - Return: Array of current values at each time

- [ ] **Update `__str__(self)` to include currents**

**File:** `usadel_keldysh_evolution.py`

- [ ] **Update `real_time_evolution()` to track currents**
  - Add `current_i = state.get_current(Q)` in loop
  - Return currents array

---

### 2.2 Vector Potential in Hamiltonian
**File:** `usadel_keldysh_evolution.py`

- [ ] **Modify `_construct_hr()` to include vector potential**
  - Add term: `+ A(t) * ...` (gauge coupling)
  - Handle A as time-dependent parameter

- [ ] **Update `_compute_dtgr()` and `_compute_dtgk()`**
  - Include A(t) in h^R construction
  - Handle external_field as A(t) function

- [ ] **Test with oscillating A(t)**
  - Define `external_field(t) = A_0 * sin(omega * t)`
  - Verify gap and current oscillate with field

**Success criteria:** System responds to external vector potential

---

## Stage 3: Self-Consistent Vector Potential

**Goal:** Make A(t) evolve self-consistently with the system.

### 3.1 Coupled Evolution
**File:** `usadel_keldysh_evolution.py`

- [ ] **Add `_compute_dtA()` method**
  - **Equation:** `∂_t A = -J(t) + ...` (depends on current)
  - Use `state.get_current()` to get J
  - Return: dA/dt

- [ ] **Modify `_evolve_state_by_one_timestep()`**
  - Evolve both state and A(t)
  - Update A: `A_new = A_old + dt * dtA`
  - Use updated A in next timestep

- [ ] **Add A(t) storage in StateObject or Evolution class**

- [ ] **Test self-consistent evolution**
  - Start with small perturbation
  - Verify A and current equilibrate
  - Check energy/current conservation laws

**Success criteria:** Self-consistent A(t) evolution, stable dynamics

---

## Stage 4: Self-Energy Additions

**Goal:** Add scattering mechanisms (elastic, Dynes, phonon).

### 4.1 SelfEnergy Classes
**File:** `self_energy_class.py`

- [ ] **Implement `ElasticScattering` class**
  - `__init__(scattering_rate, theta_arr, omega_arr)`
  - `_sigma_r(system_state)` - Angle-averaged elastic scattering
  - `_sigma_k(system_state)`
  - `_sigma_shape()`, `_get_sigma_indicies()`

- [ ] **Implement `DynesScattering` class**
  - `__init__(scattering_rate, theta_arr, omega_arr)`
  - `_sigma_r(system_state)` - Phenomenological: `-i*Gamma*tau_3`
  - `_sigma_k(system_state)`

- [ ] **Implement `PhononScattering` class**
  - `__init__(scattering_rate, omega_arr, temperature)`
  - `_sigma_r(gr, f)` - Energy-dependent kernel
  - `_sigma_k(gr, f)`

---

### 4.2 Self-Energy Integration
**File:** `usadel_keldysh_evolution.py`

- [ ] **`_generate_sigma_objects(self)`**
  - **Purpose:** Create self-energy objects from dictionary
  - **Reads from:** `self.sigma_scatterings` (optional parameter in __init__)
  - **Example:**
    ```python
    sigma_scatterings = {
        'elastic': {'rate': 0.1},
        'dynes': {'rate': 0.01}
    }
    ```
  - Create instances of SelfEnergy classes
  - Store in `self.sigma_objects` list

- [ ] **Modify `_construct_hr()` to include sigma_r**
  - Add term: `- Sigma^R`
  - May require convolution depending on self-energy type

- [ ] **Modify `_compute_dtgk()` to include collision integrals**
  - Add: `+ collision_integral[Sigma^K, g^R, g^K, f]`
  - Implement collision integral formula

- [ ] **Test with different scattering mechanisms**
  - Verify gap broadening with Dynes parameter
  - Check elastic scattering effects
  - Phonon scattering at finite temperature

**Success criteria:** Self-energies modify gap and dynamics correctly

---

## Stage 5: Optimization & Documentation (Future)

- [ ] Profile code for bottlenecks
- [ ] Optimize convolution operations (FFT, caching)
- [ ] Add comprehensive docstrings
- [ ] Type hints throughout
- [ ] Save/load for StateObject
- [ ] Parallel time evolution (if needed)
- [ ] Visualization tools
- [ ] Comparison with analytical limits

---

## Key Physics Notes

### Keldysh Formalism
- **Two Green's functions:** g^R (retarded), g^K (Keldysh)
- **Causality:** g^R upper triangular in time
- **Advanced GF:** g^A = -(g^R)^† (involution in Nambu-Keldysh)
- **Keldysh relation:** g^K = g^R @ f - f @ g^A (equilibrium)

### Normalization
- **Convolution:** `g^R @ g^R = -1` (normalization condition)
- **Gap extraction:** Must normalize by cutoff integral
- **Broadening eta:** Necessary for numerical stability

### Time Evolution
- **Equations of motion:** `i ∂_t g^R = [h^R, g^R]`
- **Initial conditions:** Equilibrium state for t < 0
- **Causality structure:** Only evolve t > t' part

---

## Implementation Strategy

1. **Stage by stage:** Complete Stage 1 fully before Stage 2
2. **Test incrementally:** Unit test each method
3. **Keep it simple:** Optimize later, correctness first
4. **Memory efficiency:** Recycle StateObject, in-place updates

---

## Common Pitfalls

- **Dimension mismatches:** Always verify tensor shapes
- **Time ordering:** Respect causality in convolutions
- **Normalization factors:** Don't forget cutoff in gap
- **Broadening eta:** Too small → numerical instability
- **Involution vs transpose:** g^A uses complete_transpose, not just transpose

---

## Current Status: Ready for Stage 1 Implementation

**Next task:** Implement StateObject.__init__() and basic structure.
