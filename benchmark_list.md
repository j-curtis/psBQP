# Benchmarking Guide: psBQP-Keldysh Code

Literature-based benchmarks for validating the Keldysh-Usadel implementation of non-equilibrium dirty superconductors.

**Last Updated:** 2026-06-15

---

## Overview

This document organizes theoretical predictions and numerical results that can be used to validate the psBQP-Keldysh code. Benchmarks are ordered from simplest (equilibrium) to most complex (driven non-equilibrium).

---

# BENCHMARK SCENARIOS

## LEVEL 1: Equilibrium & Static Properties

### Benchmark 1.1: BCS Gap vs Temperature

**What to compute:** Equilibrium gap Δ(T) as a function of temperature

**Validation:**
- BCS mean-field theory: Δ(T)/Δ(0) ≈ 1.74√(1 - T/T_c) near T_c
- Full BCS gap equation (solve self-consistently)
- Gap ratio: 2Δ(0)/k_B T_c = 3.52 for weak-coupling BCS

**References:**
- Standard BCS theory (Bardeen, Cooper, Schrieffer 1957)
- Any superconductivity textbook (Tinkham, de Gennes)

**Implementation:**
- Use `equilibrium_sweep.py` function `compute_equilibrium_gap_vs_temperature()`
- Compare with analytical BCS formula
- Check convergence with time grid resolution

---

### Benchmark 1.2: Mattis-Bardeen Optical Conductivity

**What to compute:** Optical conductivity σ(ω) at equilibrium (zero current, finite temperature)

**Observables:**
- Real part: Re[σ(ω)] - absorption
- Imaginary part: Im[σ(ω)] - dispersion
- Coherence peak at ω ≈ 2Δ
- Sum rule: ∫ Re[σ(ω)]dω conserved

**Key Physics:**
- Below gap (ω < 2Δ): exponentially suppressed Re[σ]
- At gap edge (ω = 2Δ): coherence peak
- Above gap (ω > 2Δ): quasiparticle absorption

**Primary References:**
1. **Mattis & Bardeen (1958)** - "Theory of the Anomalous Skin Effect in Normal and Superconducting Metals", Phys. Rev. 111, 412
   - Original dirty limit calculation

2. **Weydt et al. (2024)** - [Optical conductivity of a dirty current-carrying superconductor](https://arxiv.org/pdf/2512.06943)
   - Modern Keldysh sigma model derivation
   - Extends to finite supercurrent
   - **MOST RELEVANT** for your implementation

3. **Pracht et al. (2018)** - [Complete electrodynamics of a BCS superconductor with µeV energy scales](https://arxiv.org/pdf/1803.02736)
   - Precision measurements for comparison

**Implementation:**
- Use equilibrium state from `generate_initial_state(Q=0)`
- Compute `state.energy_time_representation('gr')`
- Extract conductivity from Kubo formula or directly from g^R

**Expected Results:**
- Coherence peak height depends on disorder (Dynes parameter)
- Low-temperature limit: Re[σ(ω < 2Δ)] → 0 exponentially

---

### Benchmark 1.3: Current vs Vector Potential (Equilibrium)

**What to compute:** Equilibrium current J as a function of phase gradient Q (or vector potential A)

**Observables:**
- J(Q) curve
- Depairing current: J_depair where gap goes to zero
- Critical current: J_c (maximum of J(Q))

**Key Physics:**
- Small Q: J ∝ sin(Q) (Josephson-like)
- Depairing: J_depair ∝ Δ/ξ (ξ = coherence length)
- Gap suppression: Δ(Q) decreases with current

**Primary References:**
1. **Weydt et al. (2024)** - [Optical conductivity of a dirty current-carrying superconductor](https://arxiv.org/pdf/2512.06943)
   - Section on finite Q effects
   - Keldysh-Usadel calculation

2. **Larkin-Ovchinnikov** - Theory of nonequilibrium superconductivity (1986)
   - Classic reference for current-carrying states

**Implementation:**
- Use `equilibrium_sweep.py` function `compute_equilibrium_current_and_gap_vs_vector_potential()`
- Sweep Q from 0 to depairing value
- Check gap suppression Δ(Q)

---

## LEVEL 2: Quench Dynamics (No External Fields)

### Benchmark 2.1: Gap Oscillations After Instantaneous Quench

**What to compute:** Time evolution of Δ(t) after sudden change in pairing interaction

**Scenario:**
1. Start in equilibrium at gap Δ_initial
2. Instantaneously change interaction strength λ → λ'
3. System evolves toward new equilibrium Δ_final
4. Gap oscillates and relaxes: Δ(t) → Δ_final

**Key Physics:**
- Oscillation frequency: ω ≈ 2Δ_asymptotic (Higgs mode)
- Relaxation: Power-law decay in collisionless limit
- Single-band: Δ(t) - Δ_∞ ∝ t^(-1/2)
- Two-band: Δ(t) - Δ_∞ ∝ t^(-3/2) (faster dephasing)

**Primary References:**
1. **Mori et al. (2019)** - [Post-quench gap dynamics of two-band superconductors](https://arxiv.org/pdf/1908.06125), Phys. Rev. B 100, 144513
   - **BEST STARTING BENCHMARK**
   - Exact analytical solution for single-band BCS
   - Volkov-Kogan perturbative method for two-band
   - Power-law relaxation laws derived
   - Beating patterns for multi-band

2. **Yuzbashyan & Dzero (2010)** - [Dynamics of a quantum quench in an ultra-cold atomic BCS superfluid](https://arxiv.org/pdf/1006.4579)
   - Exact solutions using integrability

**Secondary References:**
- **Maraga et al. (2023)** - [Lindblad master equation approach to dissipative quench dynamics](https://arxiv.org/pdf/2308.08264)
  - Includes dissipation effects

**Implementation:**
1. Generate equilibrium state at initial λ
2. Suddenly change BCS coupling constant in `system_parameters`
3. Evolve with `real_time_evolution()` with `driving_field=None`
4. Extract gap time series
5. Fit to: Δ(t) = Δ_∞ + A·cos(2Δ_∞ t)·t^(-α)

**Expected Results:**
- Oscillation period: T ≈ π/Δ_∞
- Decay exponent: α = 1/2 (single-band, collisionless)
- No external field needed - pure intrinsic dynamics

**Why This Is The Best First Benchmark:**
✓ Exact analytical solution exists
✓ No external fields to implement
✓ Tests time evolution machinery
✓ Clear signature (oscillations + power law)
✓ Tests gap self-consistency

---

### Benchmark 2.2: Relaxation Rates & Damping

**What to compute:** Relaxation timescales τ₁ (energy) and τ₂ (phase coherence)

**Observables:**
- τ₁: Population relaxation (energy distribution)
- τ₂: Dephasing time (coherence decay)
- Damping of gap oscillations

**Primary References:**
1. **Amin et al. (2024)** - [Measuring intrinsic relaxation rates in superconductors using nonlinear response](https://arxiv.org/html/2510.07398)
   - Recent paper on extracting τ₁ and τ₂
   - Connection to nonlinear conductivity

2. **Schwarz & Manske (2018)** - [Impact of damping on superconducting gap oscillations induced by intense Terahertz pulses](https://arxiv.org/pdf/1802.09711)
   - Phenomenological model with T₂ relaxation

**Implementation:**
- Fit gap oscillations to: Δ(t) ∝ exp(-t/τ₂)·cos(ωt)
- Compare τ₂ with disorder scattering rate
- Check if Δt ≪ ℏ/Δ (adiabatic limit)

---

## LEVEL 3: THz Driving & Pump-Probe

### Benchmark 3.1: Higgs Oscillations Under THz Pump

**What to compute:** Gap dynamics under monochromatic THz driving

**Scenario:**
- Drive superconductor with A(t) = A₀·cos(ω_drive t)
- For ω_drive ≈ Δ: resonant excitation of Higgs mode
- Gap oscillates at 2ω_drive (second harmonic)

**Key Physics:**
- Higgs mode: amplitude oscillations of order parameter
- Resonance when ω_drive ≈ 2Δ
- Nonlinear coupling: A² term drives Higgs

**Primary References:**
1. **Schwarz & Manske (2018)** - [Impact of damping on superconducting gap oscillations induced by intense Terahertz pulses](https://arxiv.org/pdf/1802.09711)
   - Gap oscillation amplitude vs THz intensity
   - Damping rate extraction
   - Comparison with NbN experiments

2. **Shimano & Tsuji (2020)** - [Higgs Mode in Superconductors](https://arxiv.org/pdf/1906.09401)
   - Comprehensive review of Higgs mode
   - Theory and experimental signatures

3. **Matsunaga & Shimano (2020)** - [Non-equilibrium phenomena in superconductors probed by femtosecond time-domain spectroscopy](https://arxiv.org/pdf/2003.11503)
   - Experimental context

**Implementation:**
- Use `driving_field_utils.py` to create THz pulse:
  ```python
  field_params = {
      'amplitude': A0,
      'frequency': omega_drive / (2*pi),  # In THz
      'phase': 0.0
  }
  ```
- Run `evolve_keldysh_state()` with `field_type='oscillatory'`
- Extract gap oscillations at 2ω_drive

**Expected Results:**
- Gap oscillation frequency: 2ω_drive
- Amplitude ∝ A₀² (nonlinear response)
- Damping time τ₂

---

### Benchmark 3.2: Third-Harmonic Generation (THG)

**What to compute:** Nonlinear optical response at 3ω from THz pump at ω

**Observables:**
- THG signal: A(3ω) / A³(ω)
- Phase of THG relative to pump
- Temperature dependence

**Primary References:**
1. **Chu et al. (2024)** - [Tracing the dynamics of superconducting order via transient terahertz third-harmonic generation](https://www.science.org/doi/10.1126/sciadv.adi7598), Science Advances
   - Optical pump - THz-THG probe protocol
   - La₂₋ₓSrₓCuO₄ experiments
   - Transient order parameter recovery

2. **Silaev (2021)** - [Phase signatures in third-harmonic response of Higgs and coexisting modes](https://arxiv.org/pdf/2107.01445)
   - Theory of THG phase

**Implementation:**
- Fourier transform current J(t) or A(t)
- Extract component at 3ω
- Compare THG(T) with experiments

---

### Benchmark 3.3: Time-Resolved Optical Conductivity

**What to compute:** σ(ω, t) - conductivity as function of both frequency and time after pump

**Scenario:**
- Pump at t=0 with optical/THz pulse
- Probe with weak pulse at various delays
- Extract σ(ω) at each time

**Primary References:**
1. **Mori (2020)** - [Time-resolved optical conductivity and Higgs oscillations in two-band dirty superconductors](https://ar5iv.labs.arxiv.org/html/2012.07674)
   - Theory for σ(ω, t)
   - Two-band effects

2. **Topp et al. (2021)** - [Theory of optical responses in clean multi-band superconductors](https://www.nature.com/articles/s41467-021-21905-x), Nature Communications
   - Multi-band extensions

**Implementation:**
- At each timestep during evolution, compute `energy_time_representation('gr')`
- Extract σ(ω, t) from Kubo formula
- Track spectral weight transfer: ∫ω·Re[σ(ω,t)]dω

---

## LEVEL 4: Advanced Non-Equilibrium

### Benchmark 4.1: Occupation Function Dynamics

**What to compute:** Distribution function f(ε, t) evolution - **YOUR CODE'S UNIQUE STRENGTH**

**Observables:**
- Non-thermal distribution f(ε, t) ≠ f_thermal(ε)
- FDT violation: g^K ≠ g^R @ f - f @ g^A
- Relaxation toward thermal equilibrium

**Key Physics:**
- Quasiparticle redistribution under driving
- Hot quasiparticles above gap
- Pair-breaking vs recombination

**Primary References:**
1. **Yokoyama et al. (2024)** - [Emergence of Larkin-Ovchinnikov-type superconducting state in a voltage-driven superconductor](https://arxiv.org/pdf/2401.04684), Phys. Rev. B
   - Non-equilibrium f(ε) under dc bias
   - Gap suppression from distribution

2. **Bergeret & Volkov (2013)** - [In-plane Fulde-Ferrel-Larkin-Ovchinnikov instability under nonequilibrium quasiparticle distribution](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.174502)
   - FFLO instability from non-thermal f

**Implementation:**
- Enable occupation tracking: `track_every_n=10` in simulation
- Access `save_data['f_energy_time']`
- Plot f(ε) at different times using `plot_energy_time_representation()`
- Compare with thermal: f_th(ε) = tanh(ε/2T)

**Unique Aspect:**
Your code tracks f(t,t') explicitly - most theories assume quasi-equilibrium f(ε)!

---

### Benchmark 4.2: DC Bias & Steady-State Non-Equilibrium

**What to compute:** Non-equilibrium steady state under constant current drive

**Scenario:**
- Apply constant dc bias: A(t) = A_dc·t (linear ramp)
- System reaches non-equilibrium steady state
- Gap suppressed, distribution non-thermal

**Primary References:**
1. **Yokoyama et al. (2024)** - [Emergence of Larkin-Ovchinnikov state in voltage-driven superconductor](https://arxiv.org/pdf/2401.04684)

2. **Adachi & Ikeda (2019)** - [Dissipative conductivity of a dirty superconductor with Dynes subgap states under dc bias](https://arxiv.org/abs/1912.01822)

**Implementation:**
- Use `field_type='DC'` with `field_params={'amplitude': dA_dt}`
- Run long enough to reach steady state
- Check gap suppression vs current

---

### Benchmark 4.3: Transient Superconductivity Enhancement

**What to compute:** Light-induced enhancement of gap Δ(t) > Δ_equilibrium

**Scenario:**
- Drive with THz field
- Gap transiently increases above equilibrium value
- Metastable enhanced state

**Primary References:**
1. **Lannig et al. (2020)** - [Temporarily enhanced superconductivity from magnetic fields](https://arxiv.org/pdf/2010.09759)

2. **Dehghani & Mitra (2018)** - [Transient Floquet engineering of superconductivity](https://arxiv.org/pdf/1808.07450)
   - Floquet theory of driven superconductors

**Implementation:**
- Short THz pulse with optimal frequency
- Check if Δ(t) > Δ_eq during pulse
- Measure enhancement lifetime

---

# ADDITIONAL THEORETICAL REFERENCES

References that provide important background but don't directly correspond to specific benchmarks.

## Foundational Theory

### Quasiclassical Formalism

- **Belzig et al. (1999)** - [Quasiclassical Green's function approach to mesoscopic superconductivity](https://arxiv.org/pdf/cond-mat/9812297)
  Comprehensive review (cond-mat/9812297)

- **Mori (2022)** - [Theory of disordered superconductors with applications to nonlinear current response](https://academic.oup.com/ptep/article/2022/3/033I03/6533532)
  Progress of Theoretical and Experimental Physics

- **Bergeret et al. (2018)** - [Colloquium: Nonequilibrium effects in superconductors with a spin-splitting field](https://arxiv.org/pdf/1706.08245)
  Rev. Mod. Phys. comprehensive colloquium

### Keldysh Technique

- **Rammer & Smith** - [Many-body theory of non-equilibrium systems](https://arxiv.org/pdf/cond-mat/0412296)
  Foundations of Keldysh formalism

- **Eschrig et al. (2009)** - [The scattering problem in non-equilibrium quasiclassical theory](https://arxiv.org/pdf/0907.2345)
  Boundary conditions

- **Eschrig et al. (2015)** - [General boundary conditions for quasiclassical theory in the diffusive limit](https://iopscience.iop.org/article/10.1088/1367-2630/17/8/083037)
  New Journal of Physics

## Numerical Methods & Codes

- **Abuaf & Maccione (2022)** - [A finite element method for the quasiclassical theory of superconductivity](https://arxiv.org/pdf/2205.09001)
  Phys. Rev. B - FEM implementation

- **Linder et al. (2016)** - [General solution of 2D and 3D superconducting quasiclassical systems](https://www.nature.com/articles/srep22765)
  Scientific Reports

- **Linder** - [GitHub - jabirali/usadel](https://github.com/jabirali/usadel)
  Open-source Julia implementation

- **Bishop-Van Horn et al. (2023)** - [pyTDGL: Time-dependent Ginzburg-Landau in Python](https://arxiv.org/pdf/2302.03812)
  TDGL code for comparison

## Higgs Mode (Additional)

- **Shimano & Tsuji (2020)** - [Higgs Mode in Superconductors](https://www.annualreviews.org/content/journals/10.1146/annurev-conmatphys-031119-050813)
  Annual Review of Condensed Matter Physics

- **Krull et al. (2020)** - [Classification and characterization of nonequilibrium Higgs modes in unconventional superconductors](https://www.nature.com/articles/s41467-019-13763-5)
  Nature Communications

- **Seibold et al. (2019)** - [Spin currents driven by the Higgs mode in magnetic superconductors](https://arxiv.org/pdf/1907.00539)

- **Tsuji & Nomura (2016)** - [Amplitude Higgs mode and admittance in superconductors with a moving condensate](https://arxiv.org/pdf/1607.01373)

- **Chu et al. (2021)** - [Light quantum control of persisting Higgs modes in iron-based superconductors](https://www.nature.com/articles/s41467-020-20350-6)
  Nature Communications

- **Giorgianni et al. (2024)** - [Kapitza-Dirac interference of Higgs waves in superconductors](https://arxiv.org/pdf/2511.10954)

## Nonlinear Response

- **Amin et al. (2024)** - [Nonequilibrium nonlinear response theory of amplitude-dependent dissipative conductivity](https://arxiv.org/pdf/2509.09766)

- **Golubov et al. (2024)** - [Plasmon excitations and their attenuation in dirty superconductors](https://arxiv.org/pdf/2511.23431)

## Spin Effects & Spin-Orbit Coupling

- **Chen et al. (2024)** - [Spin-galvanic response to non-equilibrium spin injection in superconductors with spin-orbit coupling](https://arxiv.org/pdf/2512.23536)

- **Espedal et al. (2021)** - [Spin injection and spin relaxation in odd-frequency superconductors](https://arxiv.org/pdf/2108.10313)

## Time-Dependent Ginzburg-Landau

- **Sadovskyy et al. (2020)** - [Time-Dependent Ginzburg-Landau Simulations of Superconducting Vortices in Three Dimensions](https://arxiv.org/pdf/2001.07971)

- **Kopnin** - [The Time-dependent Ginzburg-Landau Theory](https://academic.oup.com/book/4832/chapter/147224729)
  Oxford Academic book chapter

## Experimental Context

- [Probing Superconducting Gap Dynamics with THz Pulses](https://opg.optica.org/abstract.cfm?uri=cleo_si-2015-SM3H.3) - CLEO conference

- Experimental data on NbN, PCCO cuprates, La₂₋ₓSrₓCuO₄ from THz spectroscopy groups

---

# VALIDATION CHECKLIST

## Code Consistency (Internal Checks)

- [x] **Normalization of g^R**: Check via `check_normalizations()`
  - ∫ dt' g^R(t,t') g^R(t',t'') + [τ₃, g^R(t,t'')] = 0

- [x] **Normalization of g^K**: Check via `check_normalizations()`
  - ∫ dt' (g^R g^K + g^K g^A) + [τ₃, g^K] - FDT terms = 0

- [x] **FDT relation**: Check via `check_fdt()`
  - g^K = g^R @ f - f @ g^A (equilibrium)
  - Violations indicate non-equilibrium

- [x] **Causality**: g^R(t,t') = 0 for t < t'
  - Automatically enforced by code structure

- [x] **Involution**: g^A = -τ₃ (g^R)^† τ₃
  - Used in `state._r2a()`

- [x] **Gap self-consistency**: Δ = -λ/4 Tr[τ₋ g^K(t,t)]
  - Check convergence of gap equation

## Physics Benchmarks (External Validation)

### Equilibrium (Level 1)
- [ ] **Gap vs T**: Compare with BCS Δ(T)/Δ(0)
- [ ] **Mattis-Bardeen σ(ω)**: Check coherence peak
- [ ] **J(Q) curve**: Check depairing current

### Quench (Level 2)
- [ ] **Gap oscillations**: Frequency ω ≈ 2Δ_final
- [ ] **Power-law decay**: t^(-1/2) for single-band
- [ ] **Relaxation time τ₂**: Compare with disorder rate

### THz Driving (Level 3)
- [ ] **Higgs resonance**: Enhanced response at ω ≈ 2Δ
- [ ] **THG signal**: Third-harmonic generation
- [ ] **σ(ω,t) dynamics**: Time-resolved conductivity

### Advanced (Level 4)
- [ ] **Occupation f(ε,t)**: Non-thermal distribution
- [ ] **DC steady state**: Gap suppression under current
- [ ] **Transient enhancement**: Δ(t) > Δ_eq possible?

---

# RECOMMENDED WORKFLOW

## Step 1: Code Validation (Internal)
Run all consistency checks:
```python
from data_analysis import check_normalizations, check_fdt

# Check g^R and g^K normalization
norm_results = check_normalizations(timestamp, job_index)
print(f"g^R max error: {norm_results['gr_max_error']}")
print(f"g^K max error: {norm_results['gk_max_error']}")

# Check FDT relation
fdt_results = check_fdt(timestamp, job_index)
print(f"FDT max error: {fdt_results['max_error']}")
```

## Step 2: Equilibrium Benchmarks
1. Run equilibrium gap vs T sweep
2. Compare with BCS theory
3. Compute Mattis-Bardeen σ(ω)

## Step 3: First Dynamics Benchmark
**Start with quench** - simplest non-equilibrium case:
1. Set up initial equilibrium state
2. Change coupling constant λ
3. Run evolution (no external field)
4. Compare gap oscillations with [Mori et al. 2019]

## Step 4: THz Benchmarks
Once quench works, add external fields:
1. Implement THz pulse
2. Check Higgs oscillations
3. Compare with [Schwarz & Manske 2018]

## Step 5: Exploit Unique Features
Your occupation function tracking is unique - use it:
1. Plot f(ε,t) evolution
2. Check FDT violations
3. Study non-thermal distributions

---

**Compiled by:** Claude Code
**Project:** psBQP-Keldysh (Superconducting Vortices - Keldysh non-equilibrium)
**Date:** June 15, 2026
