# NambuKeldyshTensor Operations Guide

## Data Structure

`NambuKeldyshTensor` represents 2×2 Nambu matrices with additional dimensions:
- First two dimensions (2, 2): Nambu space (particle-hole)
- Additional dimensions: time, frequency, angle, etc.

Example shapes:
- `(2, 2)`: Single Nambu matrix (scalar-like)
- `(2, 2, Nt)`: Nambu vector over time/frequency
- `(2, 2, Nt, Nt)`: Two-time Green's function g(t,t')
- `(2, 2, 1, Nt)`: Row vector (e.g., g(t_fixed, t'))

## Core Operations

### 1. `*` (Nambu Matrix Multiplication)

Contracts over Nambu indices, element-wise on other dimensions.

```python
# (2,2,a,b) * (2,2,a,b) -> (2,2,a,b)
A = NambuKeldyshTensor(...)  # shape (2, 2, Nt)
B = NambuKeldyshTensor(...)  # shape (2, 2, Nt)
C = A * B  # Nambu product at each time point
```

**Formula**: `C[i,m,n] = Σ_j A[i,j,n] B[j,m,n]` (sum over j, Nambu contraction)

**Use case**: `τ₃ * gap_tensor` where τ₃ is a Pauli matrix

### 2. `@` (Convolution/Inner Product)

Contracts over Nambu index AND one shared dimension.

```python
# (2,2,1,Nt) @ (2,2,Nt,Nt) -> (2,2,1,Nt)
gr_row = state.gr[-1:, :]  # shape (2, 2, 1, Nt)
gk_col = state.gk[:, -1:]   # shape (2, 2, Nt, 1)
result = gr_row @ gk_col    # shape (2, 2, 1, 1)
```

**Formula**: Contracts last dimension of left with first extra dimension of right:
- `(2,2,a,b) @ (2,2,b,c)` → `(2,2,a,c)`
- Einstein notation: `ij{left_extra}, jk{right_extra} -> ik{result_extra}`

**Use case**: Time integration `∫ g^R(t,t'') g^K(t'',t') dt''`

### 3. `precise_convolution_left(other, other_integral, dt)`

Regularized convolution: `self @ other` with suppression of Gibbs oscillations.

```python
# Compute: gr @ f with regularization on f
gr_row = state.gr[-1:, :]          # (2, 2, 1, Nt)
f_thermal = thermal_dist[-1:, :]    # (2, 2, 1, Nt)
f_integral = thermal_integral[-1:, :] # (2, 2, 1, Nt)

result = gr_row.precise_convolution_left(f_thermal, f_integral, dt)
```

**Formula**:
```
result = dt · (self @ other)           [standard convolution]
       - dt · (self * (ones @ other))  [factored term]
       + (self * other_integral)       [analytic term]
```

Where `ones` is a row vector of ones (identity in Nambu space).

**Use case**: FDT relation `g^R @ f` where f has singularities

### 4. `precise_convolution_right(other, other_integral, dt, self_index)`

Regularized convolution: `other @ self` with suppression of Gibbs oscillations.

```python
# Compute: f @ ga with regularization on f
ga = state._r2a()                      # (2, 2, Nt, Nt)
f_row = thermal_dist[-1:, :]           # (2, 2, 1, Nt)
f_integral = thermal_integral[-1:, :]  # (2, 2, 1, Nt)

result = ga.precise_convolution_right(f_row, f_integral, dt, self_index=-1)
```

**Formula**:
```
result = dt · (other @ self)           [standard convolution]
       - dt · ((other @ ones) * self_row) [factored term]
       + (other_integral * self_row)      [analytic term]
```

Where `ones` is a column vector of ones.

**Special handling**: When `other` is a row (shape 2,2,1,Nt), extracts `self[self_index, :]` for regularization terms. The `self_index` parameter specifies which row to use.

**Use case**: FDT relation `f @ g^A` where f has singularities

## Shape Compatibility Rules

### Multiplication `*`

Broadcasting follows these rules:
1. Nambu dimensions (first two) must be (2, 2) for both operands
2. Extra dimensions broadcast element-wise
3. Missing dimensions are added via broadcasting

Examples:
- `(2,2,Nt) * (2,2,Nt)` → `(2,2,Nt)` ✓
- `(2,2,1,Nt) * (2,2,Nt,Nt)` → broadcasts to `(2,2,Nt,Nt)` ✓
- `(2,2) * (2,2,Nt)` → broadcasts to `(2,2,Nt)` ✓

### Matmul `@`

Contraction requires:
1. Both must have Nambu structure (2, 2)
2. Last dimension of left = first extra dimension of right (after Nambu)

Examples:
- `(2,2,Nt,Nt) @ (2,2,Nt,1)` → `(2,2,Nt,1)` ✓
- `(2,2,1,Nt) @ (2,2,Nt)` → `(2,2,1)` ✓
- `(2,2,5,10) @ (2,2,10,3)` → `(2,2,5,3)` ✓
- `(2,2,5,10) @ (2,2,3,10)` → Error: 10 ≠ 3 ✗

### Indexing `[i, j]`

Indexing always preserves Nambu dimensions:
```python
g = NambuKeldyshTensor(...)  # shape (2, 2, Nt, Nt)
g_row = g[-1:, :]            # shape (2, 2, 1, Nt) - last row
g_col = g[:, -1:]            # shape (2, 2, Nt, 1) - last column
g_diag = g[-1, -1]           # shape (2, 2) - diagonal element
```

**Note**: `g[-1:, :]` uses slice notation to preserve the row structure (dimension = 1 instead of removed).

## Common Patterns in Evolution Code

### Pattern 1: Computing RHS with precise convolution

```python
# Compute: ∫ g^R(t,t'') f(t'',t') dt'' with regularization
rhs = gr[-1:,:].precise_convolution_left(
    thermal_dist.shift(1, axis=1),
    thermal_integral.shift(1, axis=1),
    dt
)
```

### Pattern 2: Two-time thermal distribution

```python
# Generate thermal distribution f(t-t')
evolution.get_thermal_occupation(temperature)
f = evolution.thermal_dist  # shape (2, 2, Nt, Nt)

# Extract row for specific time
f_row = f[-1:, :]  # shape (2, 2, 1, Nt)
```

### Pattern 3: Gap extraction from g^K

```python
# Gap equation: Δ = -λ/4 · Tr[τ₋ g^K(t,t)]
gk_traced = state.gk.trace(pauli_index='-')  # shape (Nt, Nt)
gk_diag = np.diagonal(gk_traced)              # shape (Nt,)
gap_history = -0.25 * lambda_bcs * gk_diag
```

## Bug Fix: Negative Index Handling

**Issue**: When `self_index = -1` in `precise_convolution_right`, the slice `self[-1:0, :]` becomes empty because `-1 + 1 = 0`.

**Solution**: Convert negative indices to positive before slicing:
```python
N_t = self.data.shape[2]
if self_index < 0:
    positive_index = N_t + self_index  # -1 becomes Nt-1
else:
    positive_index = self_index
self_for_reg = self[positive_index:positive_index+1, :]
```

This ensures `self[-1]` correctly extracts the last row as `self[Nt-1:Nt, :]`.
