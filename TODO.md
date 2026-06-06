# Keldysh Non-Equilibrium Implementation TODO


TEST for static state
	normalization_test
		- computes the gr@gr normalization by calling the appropriate state_object functions
		- computes the gk normalization by calling the appropriate state_object functions
	FDT test
		- computes the fdt comparison by calling the  appropriate state_object functions
	Equilibrium state for different system sizes
 		- and plot of results vs delta_t. do 1000, 1250, 1500,1750 and 2000 points
	Current:
		- compute current for the equilibrated state should be zero for zero vector potential
	Time translation invariance:
		- check that the last 10 rows shifted by 1 time index all match eachother. in principle the equilibrated state result should only depend on t-t' 

Evolution tests: -- Implement with Demler tools framework
	long-time stability test
		- start from an equilibrated state, evolve for long time and make sure it stays in equilibrium
	gap vs T test
		- start from an equilibrated state at a certain T and quench the temperature to a new value - reproduce gap(T) i.e. do this test for 10 different temperatures, extract the finally converged gap and plot vs temperature
		- note the gk should be properly changed at the beggining in accordinace to f(T) changing 
		- check for stability of temperature quench dynamics
	Normal state equilibrium:
		- Quench the temperature to T>T_c and see if the result is correct and if any numerical issues appear

Response tests:
	Current vs DC vector potential test
		- start from an equilibrated state with zero vector potential and equilibrate to different DC values of the vector potential
		- 
		

2) Direct Response functions
Test1: DC linear response (current computation) --> correct superfluid density
Test2: AC linear response
Test3: DC non-linear J_S(Q), critical current and breakdown and decay of gap with increasing A potential --> compare with equilibrium code
Test4: Return to equilibrium after a weak pulse has been sent into the system
Test5: Return to equilibrium after a strong pulse has been sent into the system
Test6: strong vector potential FDT 
Test7:  Matching of equilibrium and code g^r for finite A value! start from equilibrium of finite A, propagate and nothing happens A --> DC value --> Hard if we dont know the gap! 

3) Involved tests
Test1: Non-equilibrium continuous drive test --> increasing gap up to zero temperature value with microwave drive
--

4) External pulse response tests
Test1: Transmitivity under linear response

 
## Stage 4: Increased Accuracy & Evolution Methods

### Performance Optimization

- [ ] Profile bottlenecks
- [ ] Optimize convolutions (FFT, caching)
- [ ] Implement parallel time evolution

### Code Quality

- [ ] Add docstrings + type hints
- [ ] Implement save/load for StateObject
- [ ] Create visualization tools

### Validation & Consistency

- [ ] Compare with analytical limits
- [ ] Check all derivations are correct and labeling is consistent
- [ ] Fix all the eta factors throughout the document to be eta not eta/2

---

## Stage 5: Self-Energies

### Implementation Tasks

- [ ] Implement `ElasticScattering` class
- [ ] Implement `DynesScattering` class
- [ ] Implement `PhononScattering` class
- [ ] Implement `_generate_sigma_objects()`
- [ ] Update `_construct_hr()` and `_compute_dtgk()` for self-energies

