# Test / TODO list (psBQP-keldysh)

- [ ] test and verify energy and occupation function tracking functions
- [ ] check the time labels on the gap, A and J
- [ ] circuit mode (current pulse) normal-state test on the cluster: T > T_c, field_type='current_gaussian' (amplitude ~0.05, FWHM 5), circuit_params {'Z_T': 50, 'R_n': 170, 'J_DP_prime': 1, 'R_c': 0 and 20}; saved currents / input_pulse must be 2 Z_T / (2 Z_T + R_n + R_c) = 0.3704 (R_c = 0) and 0.3448 (R_c = 20) at all times; compare currents[k] with the input at t_(k+1) and scan a time shift. Local N_t = 128 run (dt = 0.157): mean ratio agrees to 0.2 % (R_c = 0) and 0.5 % (R_c = 20), peak transmission 0.373 / 0.348, but the pointwise ratio varies between 0.24 and 0.48 on the pulse flanks (I_in > 5 % of peak), i.e. a time offset of about one to two steps or an error of a few % of the peak J; not resolved
- [ ] circuit mode: slow pulse in the superconducting state compared with equilibrium_python.real_time_evolve (adiabatic limit)
- [ ] verify the reconstructed energy representation (postprocessing 'energy_representation') on an equilibrium state: g^R = g'^R + tau_3 should give the BCS/Dynes DOS, g^K = g'' + 2 tau_3 tanh + T(eps) should equal tanh(eps/2T) (g^R - g^A), the thermal gap term should be about -2i Delta tanh(eps/2T)/eps, and n should be about 0
