# Discretization Instability Analysis

## The Fundamental Problem

Your current discretization uses an **asymmetric stencil**:
- **Forward difference** in first argument: `g(t+Δt, t') - g(t, t')`
- **Backward difference** in second argument: `g(t, t'+Δt) - g(t, t')` [via shift(-1)]

This asymmetry **breaks time-translation invariance** in equilibrium.

## Why Current Method Fails

### 1. Time-Translation Invariance Requires Symmetry

In equilibrium, g(t,t') = g(τ) where τ = t - t'. This means:
```
g(t+Δt, t) should equal g(t, t-Δt)  [both are g(τ=Δt)]
```

Your evolution computes g(t+Δt, t') from:
- g(t, t') and g(t, t'+Δt)

But it SHOULD relate g(t+Δt, t') to g(t, t'-Δt) by symmetry. The current stencil points in the wrong direction!

### 2. τ₃ Component is Physical, Not Error

Equilibrium g^R(ω) has:
```
g₃(ω) ∝ ω  (frequency appears in diagonal elements)
```

After FFT to time domain, τ₃(τ) has specific structure determined by physics. The evolution must **preserve this structure**, not amplify it.

### 3. Amplification is Structure-Violating

The term `g + τ₃*g*τ₃`:
- Preserves τ₁, τ₂ subspace (pure anomalous pairing) ✓
- **Doubles** τ₀, τ₃ subspace (normal + energy) ✗

This would be fine if g started with EXACTLY the right τ₃, but any deviation amplifies exponentially.

## Why This Happens

The discrete equation integrates over [t, t+Δt] using approximations like:
```
∫ U_L(t₁) ∂_{t₁} h(t₁, t') dt₁ ≈ U_L(t) [h(t+Δt, t') - h(t, t')]
```

This assumes U_L(t₁) ≈ U_L(t) inside the integral (error O(Δt²) claimed in paper line 2160).

**BUT:** Even though truncation error is O(Δt²), the **algebraic structure** of the discrete equation doesn't preserve time-translation invariance. The error accumulates over many steps.

## Proposed Solutions

### Option 1: Crank-Nicolson (Midpoint/Trapezoidal)

Use **symmetric** time differencing:
```python
# Current (asymmetric):
∂_t g ≈ [g(t+Δt,t') - g(t,t')]/Δt
∂_{t'} g ≈ [g(t,t'+Δt) - g(t,t')]/Δt

# Symmetric:
∂_t g ≈ [g(t+Δt,t') - g(t-Δt,t')]/(2Δt)
∂_{t'} g ≈ [g(t,t'+Δt) - g(t,t'-Δt)]/(2Δt)
```

**Pros:**
- Preserves time-translation invariance
- 2nd order accurate (vs 1st order current)
- Standard for parabolic PDEs

**Cons:**
- Implicit method → need to solve linear system each step
- More complex implementation

### Option 2: Projection Method (Easiest Fix)

Keep current evolution, but **enforce constraints** after each step:

```python
def project_to_equilibrium_manifold(gr_new, gr_equilibrium_reference, tau):
    """
    Project evolved GF back to equilibrium manifold.

    In equilibrium, g(t,t') should depend only on τ = t-t'.
    Use reference equilibrium g_eq(τ) to correct errors.
    """
    # Extract current Pauli components
    g0_now = gr_new.trace(0) / 2
    g1_now = gr_new.trace(1) / 2
    g2_now = gr_new.trace(2) / 2
    g3_now = gr_new.trace(3) / 2

    # Get reference equilibrium values at this τ
    g0_eq = gr_equilibrium_reference.trace(0) / 2
    g1_eq = gr_equilibrium_reference.trace(1) / 2
    g2_eq = gr_equilibrium_reference.trace(2) / 2
    g3_eq = gr_equilibrium_reference.trace(3) / 2

    # Damping factor for τ₃ overgrowth
    damping = 0.5  # Reduce amplification from 2.0 to 1.0

    # Reconstruct with damped τ₃
    gr_projected = (NambuKeldyshTensor(g1_now, pauli_channel=1) +
                    NambuKeldyshTensor(g2_now, pauli_channel=2) +
                    NambuKeldyshTensor(damping * g3_now, pauli_channel=3)) / 2

    return gr_projected
```

**Pros:**
- Simple to implement
- Guarantees stability
- Can tune damping factor

**Cons:**
- Ad-hoc correction
- Not systematic
- Might affect physics in non-equilibrium

### Option 3: Strang Splitting (Structure-Preserving)

Split evolution into parts that preserve different structures:

```python
# Split Hamiltonian evolution
H_gap = Δ(t) * (τ₁ + iτ₂)  # Gap term (preserves anomalous pairing)
H_damping = -iη/2 * τ₀       # Damping term

# Evolution as composition:
g(t+Δt) = exp(-iH_damping Δt/2) * exp(-iH_gap Δt) * exp(-iH_damping Δt/2) * g(t)
```

Each piece can be solved exactly, preserving its geometric structure.

**Pros:**
- Preserves multiple conservation laws simultaneously
- 2nd order accurate
- No linear solve needed (explicit)

**Cons:**
- More complex to derive for two-time functions
- Need to identify correct splitting

### Option 4: Exponential Integrator

For linear systems like Usadel, use exact time evolution of linear part:

```python
# Exact solution of linear part:
g_linear(t+Δt) = exp(L*Δt) * g(t)

# Where L is the linear operator (can be computed via matrix exponential)
# Then add nonlinear corrections perturbatively
```

**Pros:**
- Exact for linear part
- Automatically stable
- Preserves structure of continuous equation

**Cons:**
- Complex implementation
- Expensive matrix exponentials

## Recommended Approach

**For immediate fix (while writing paper):**
Use **Option 2 (Projection)** with damping factor 0.5 to cancel the 2× amplification.

**For proper long-term solution:**
Implement **Option 1 (Crank-Nicolson)** as it's standard, well-tested, and preserves the physics correctly.

## Testing Protocol

After implementing any fix, verify:

1. **Time-translation invariance in equilibrium:**
   ```
   ||g(t+Δt, t) - g(t, t-Δt)|| < 1e-10
   ```

2. **Stability over long evolution:**
   ```
   max|τ₃(t)| / max|τ₃(0)| ≈ 1.0  (not growing)
   ```

3. **Convergence with Δt:**
   ```
   Error ∝ Δt^p where p ≥ 2 for good schemes
   ```

4. **Physical self-consistency:**
   - Gap equation still satisfied
   - Normalization preserved
   - Current conservation holds

## References

- Hairer, Lubich, Wanner: "Geometric Numerical Integration" (structure-preserving methods)
- Ascher & Petzold: "Computer Methods for ODEs and DAEs" (Crank-Nicolson, exponential integrators)
- Leimkuhler & Reich: "Simulating Hamiltonian Dynamics" (symplectic integrators for quantum systems)
