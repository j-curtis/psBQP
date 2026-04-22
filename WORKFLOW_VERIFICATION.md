# Code Completeness Verification

## Required Workflow

```python
# Step 1: Initialize solver with grid
solver = UsadelKeldyshEvolution(grid_parameters, system_parameters)

# Step 2: Generate initial equilibrium state
initial_state = solver._generate_initial_state()

# Step 3: Perform real-time evolution
final_state, gaps, currents = solver.real_time_evolution(
    initial_state,
    num_timesteps=1000,
    external_field=external_field_function
)
```

## Method Dependencies Check

### ✓ UsadelKeldyshEvolution
- [x] `__init__(grid_parameters, system_parameters, optimization_parameters, sigma_scatterings)`
- [x] `_generate_time_grid()` - Called by __init__
- [x] `_generate_sigma_objects()` - Called by __init__
- [x] `_generate_initial_state()` - **Main entry: Step 2**
- [x] `real_time_evolution(initial_state, num_timesteps, external_field)` - **Main entry: Step 3**
- [x] `_evolve_state_by_one_timestep(state, time_index, external_field)`
- [x] `_evolve_gr_by_one_timestep(state, time_index, external_field)`
- [x] `_evolve_gk_by_one_timestep(state, time_index, external_field)`
- [x] `_construct_hr(Q, delta, sigma_r)`
- [x] `_get_thermal_occupation(temperature)`
- [x] `get_bcs_gap_constant()` [static]
- [x] `get_bcs_ratio()` [static]
- [x] `_get_BCS_coupling()`
- [x] `_calc_dtQ_prefactor_new(gr, temperature)`

### ✓ EquilibriumSolver
- [x] `__init__(grid_parameters, system_parameters)`
- [x] `compute_self_cons_equilibrium(temperature, Q)` - Called by _generate_initial_state()
- [x] `_solve_gap_equation(gr, f, delta_guess)`
- [x] `_iterate_self_consistency(delta, f, max_iterations, tolerance)`

### ✓ StateObject
- [x] `__init__(gr, gk)`
- [x] `_r2a()` - Compute g^A from g^R
- [x] `get_gap()` - **Called by real_time_evolution()**
- [x] `get_current(Q)` - **Called by real_time_evolution()**
- [x] `_update_state_object(new_gr, new_gk, time_index)` - **Called by _evolve_state_by_one_timestep()**
- [x] `copy()`
- [x] `check_normalization()`
- [x] `check_keldysh_relation()`
- [x] `save(filepath)`
- [x] `load(filepath)` [static]
- [x] `__str__()`, `__repr__()`, `__del__()`

### ✓ NambuKeldyshTensor
- [x] `__init__(data_in, pauli_channel)`
- [x] `__mul__(other)` - Nambu matrix product
- [x] `__matmul__(other)` - Convolution
- [x] `__add__()`, `__sub__()`, `__truediv__()`, `__neg__()`
- [x] `__rmul__()`, `__radd__()`, `__rtruediv__()`
- [x] `__getitem__()`, `__str__()`
- [x] `_is_scalar()`, `_is_single_time()`, `_broadcast_to_shape()`, `_make_compatible()`
- [x] `_conj()`, `_transpose()`, `_involution()`, `_determinant()`
- [x] `_trace(pauli_index)`
- [x] `_flatten_nambu_object()`, `_flatten_nambu_object_to_complex()`, `_unflatten_nambu_object()` [static]
- [x] `_join_nambu_list()` [static]

### ✓ SelfEnergy (Abstract + Implementations)
- [x] `SelfEnergy.__init__(scattering_rate, omega_arr)` [ABC]
- [x] `_sigma_r(gr, f)` [abstract]
- [x] `_sigma_k(gr, f)` [abstract]
- [x] `_sigma_shape()` [abstract]
- [x] `_get_sigma_indicies()` [abstract]
- [x] `ElasticScattering`, `DynesScattering`, `PhononScattering` implementations

## Call Chain Verification

```
UsadelKeldyshEvolution.__init__(grid_params, system_params)
    ├─> _generate_time_grid()
    └─> _generate_sigma_objects()

UsadelKeldyshEvolution._generate_initial_state()
    └─> EquilibriumSolver.compute_self_cons_equilibrium(T, Q)
        ├─> _get_thermal_occupation(T)
        ├─> _solve_gap_equation(gr, f, delta_guess)
        └─> _iterate_self_consistency(delta, f, ...)

UsadelKeldyshEvolution.real_time_evolution(initial_state, num_steps, field)
    Loop for each timestep:
        ├─> _evolve_state_by_one_timestep(state, i, field)
        │   ├─> _evolve_gr_by_one_timestep(state, i, field)
        │   │   └─> _construct_hr(Q, delta, sigma_r)
        │   ├─> _evolve_gk_by_one_timestep(state, i, field)
        │   └─> state._update_state_object(new_gr, new_gk, i)
        ├─> state.get_gap()
        └─> state.get_current(Q)
    Return: final_state, gaps[], currents[]
```

## Status: ✅ COMPLETE

All necessary components are present for the full workflow:
1. ✅ Grid initialization
2. ✅ Initial state generation (via equilibrium solver)
3. ✅ Real-time evolution loop
4. ✅ Observable extraction (gap, current)

## Next Steps for Implementation

When implementing, fill in the `pass` statements in this order:

### Phase 1: Foundation
1. `NambuKeldyshTensor.__init__()` and basic operations
2. `get_pauli_matrix()`
3. `StateObject.__init__()`

### Phase 2: Grid Setup
4. `UsadelKeldyshEvolution.__init__()`
5. `UsadelKeldyshEvolution._generate_time_grid()`

### Phase 3: Equilibrium
6. `EquilibriumSolver.__init__()`
7. `EquilibriumSolver._iterate_self_consistency()`
8. `EquilibriumSolver._solve_gap_equation()`
9. `EquilibriumSolver.compute_self_cons_equilibrium()`

### Phase 4: Initial State
10. `UsadelKeldyshEvolution._construct_hr()`
11. `UsadelKeldyshEvolution._get_thermal_occupation()`
12. `UsadelKeldyshEvolution._generate_initial_state()`

### Phase 5: Observables
13. `StateObject.get_gap()`
14. `StateObject.get_current()`
15. `StateObject._update_state_object()`

### Phase 6: Time Evolution
16. `UsadelKeldyshEvolution._evolve_gr_by_one_timestep()`
17. `UsadelKeldyshEvolution._evolve_gk_by_one_timestep()`
18. `UsadelKeldyshEvolution._evolve_state_by_one_timestep()`
19. `UsadelKeldyshEvolution.real_time_evolution()`

## Example Usage

```python
import numpy as np
from usadel_keldysh_evolution import UsadelKeldyshEvolution

# Define grids
grid_parameters = {
    'omega_sampling': 101,
    'cutoff': 10.0,
    'time_sampling': 200,
    'time_duration': 20.0
}

# System parameters
system_parameters = {
    'critical_temperature': 1.0,
    'temperature': 0.5,
    'eta': 0.01
}

# Initialize solver
solver = UsadelKeldyshEvolution(grid_parameters, system_parameters)

# Generate initial equilibrium state (t < 0)
initial_state = solver._generate_initial_state()

# Define external field (optional)
def external_field(t):
    return 0.1 * np.sin(0.5 * t)

# Run real-time evolution (t >= 0)
final_state, gaps, currents = solver.real_time_evolution(
    initial_state,
    num_timesteps=200,
    external_field=external_field
)

# Analyze results
import matplotlib.pyplot as plt
time_grid = solver.time_grid

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(time_grid, gaps)
plt.xlabel('Time')
plt.ylabel('Gap Δ(t)')

plt.subplot(132)
plt.plot(time_grid, currents)
plt.xlabel('Time')
plt.ylabel('Current I(t)')

plt.subplot(133)
plt.plot(time_grid, external_field(time_grid))
plt.xlabel('Time')
plt.ylabel('External Field')

plt.tight_layout()
plt.show()
```
