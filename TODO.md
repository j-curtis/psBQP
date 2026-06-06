# Keldysh Non-Equilibrium Implementation TODO



TEST for static state

Evolution tests: -- Implement with Demler tools framework
	
Response tests:
	Current vs DC vector potential test
		- start from an equilibrated state with zero vector potential and equilibrate to different DC values of the vector potential
		- 
		
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

