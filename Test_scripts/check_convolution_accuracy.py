"""
Check precise_convolution accuracy by comparing with frequency-domain computation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from usadel_keldysh_evolution import UsadelKeldyshEvolution
from equilibrium_class import EquilibriumSolver
from nambu_keldysh_class import NambuKeldyshTensor
from nambu_class import NambuTensor

def check_convolution_accuracy(T_max=2*np.pi*5, N_t=201, temperature=0.1, 
                               check_type='both', save_plots=True):
    """
    Compare precise_convolution with frequency-domain calculation.
    """
    
    print("="*80)
    print(f"CHECKING CONVOLUTION ACCURACY: {check_type}")
    print("="*80)
    print(f"Parameters: N_t={N_t}, T_max={T_max:.3f}, T={temperature}")
    
    dt = T_max / (N_t - 1)
    print(f"Time step: dt = {dt:.6f}")
    
    # Create evolution object
    grid_parameters = {
        'time_sampling': N_t,
        'time_duration': T_max,
        'eta': 0.2
    }
    
    system_parameters = {
        'critical_temperature': 1.0,
        'temperature': temperature,
        'eta': 0.2
    }
    
    print("\nGenerating initial state...")
    evolution = UsadelKeldyshEvolution(grid_parameters, system_parameters)
    initial_state, _, _, _ = evolution.generate_initial_state(Q=0.0)
    
    # Get thermal distributions
    evolution.get_thermal_occupation(temperature)
    evolution.get_thermal_integral(temperature)
    evolution.get_thermal_sum(temperature)
    
    gap = initial_state.get_gap_history()[-1]
    print(f"Gap: Δ = {gap:.6f}")
    
    # Create equilibrium solver to get frequency-domain functions
    print("\nCreating equilibrium solver for frequency domain...")
    grid_params_with_omega = grid_parameters.copy()
    grid_params_with_omega['omega_grid'] = evolution.omega_grid
    grid_params_with_omega['energy_cutoff'] = evolution.energy_cutoff
    
    eq_solver = EquilibriumSolver(
        grid_params_with_omega,
        system_parameters,
        evolution.optimization_parameters,
        evolution.sigma_scatterings
    )

    # Get Green's functions and thermal distribution in frequency domain
    print("Getting frequency-domain Green's functions...")
    gr_omega = eq_solver.compute_equilibrium_gr(temperature)
    ga_omega = eq_solver._compute_advanced(gr_omega)

    # Get thermal distribution from equilibrium solver (already in frequency domain)
    print("Getting thermal distribution in frequency domain...")
    f_omega = eq_solver._get_thermal_occupation(temperature)

    # Get the actual omega grid used by the equilibrium solver
    omega_grid = eq_solver.usadel_solver.w_arr
    d_omega = omega_grid[1] - omega_grid[0]

    # Time grid for comparison
    time_grid = evolution.time_grid
    
    print(f"  gr_omega shape: {gr_omega.data.shape}")
    print(f"  f_omega shape: {f_omega.data.shape}")
    omega_grid = eq_solver.usadel_solver.w_arr  # Get actual omega grid from solver
    print(f"  omega_grid from solver length: {len(omega_grid)}")
    print(f"  omega_grid from evolution length: {len(evolution.omega_grid)}")
    print(f"  omega_grid range: [{omega_grid[0]:.3f}, {omega_grid[-1]:.3f}]")
    print(f"  d_omega: {d_omega:.6f}")
    
    results = {}
    
    # CHECK 1: gr @ f
    if check_type in ['gr', 'both']:
        print("\n" + "-"*80)
        print("CHECK 1: gr @ f convolution")
        print("-"*80)
        
        # Frequency domain: GR(ω) * F(ω) (element-wise multiplication of full tensors)
        # Multiply the data arrays element-wise (broadcasting over (2,2) Nambu indices)
        product_gr_f_data = gr_omega.data * f_omega.data  # Shape: (2, 2, n_omega)
        # Extract tau_2 component after multiplication
        product_tau2 = (product_gr_f_data[0, 1, :] - product_gr_f_data[1, 0, :]) / 2j  # tau_2 = (σ_y ⊗ I)

        # Inverse FFT
        product_tau_raw = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(product_tau2)))
        product_tau_raw = product_tau_raw * d_omega / (2.0 * np.pi)

        # Convert to two-time
        # The tau grid goes from -T_max to T_max with n_omega points
        n_omega = len(omega_grid)
        tau_grid_fft = np.linspace(-T_max, T_max, n_omega)
        dt_fft = tau_grid_fft[1] - tau_grid_fft[0]

        t_i, t_j = np.meshgrid(time_grid, time_grid, indexing='ij')
        tau_matrix = t_i - t_j
        tau_idx_matrix = np.round((tau_matrix - tau_grid_fft[0]) / dt_fft).astype(int)
        tau_idx_matrix = np.clip(tau_idx_matrix, 0, n_omega - 1)
        
        gr_f_freq = product_tau_raw[tau_idx_matrix]
        
        # Time domain: precise_convolution_left
        gr_row = initial_state.gr[-1:, :]
        gr_f_time_row = gr_row.precise_convolution_left(
            evolution.thermal_dist, evolution.thermal_integral, dt, 
            other_index=-1, precomputed_sum=evolution.thermal_sum_right, gap_tensor=None
        )
        
        gr_f_time = gr_f_time_row.trace(pauli_index=2) / 2
        gr_f_time = np.array(gr_f_time[0, :])
        
        # Compare
        gr_f_freq_row = gr_f_freq[-1, :]
        error_gr = np.abs(gr_f_freq_row - gr_f_time)
        
        print(f"  Max error: {np.max(error_gr):.6e}")
        print(f"  Mean error: {np.mean(error_gr):.6e}")
        print(f"  Error/dt: {np.max(error_gr)/dt:.6f}")
        
        results['gr_f'] = {
            'freq': gr_f_freq_row,
            'time': gr_f_time,
            'error': error_gr,
            'max_error': np.max(error_gr)
        }
    
    # CHECK 2: f @ ga
    if check_type in ['ga', 'both']:
        print("\n" + "-"*80)
        print("CHECK 2: f @ ga convolution")
        print("-"*80)
        
        # Frequency domain: F(ω) * GA(ω) (element-wise multiplication)
        # Need to extract data arrays and multiply element-wise
        f_omega_2 = np.array(f_omega._trace(2)) / 2  # Extract tau_2 component
        ga_omega_2 = np.array(ga_omega._trace(2)) / 2  # Extract tau_2 component
        product_tau2 = f_omega_2 * ga_omega_2  # Element-wise multiplication in frequency

        # Inverse FFT
        product_tau_raw = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(product_tau2)))
        product_tau_raw = product_tau_raw * d_omega / (2.0 * np.pi)

        # Convert to two-time
        # The tau grid goes from -T_max to T_max with n_omega points
        n_omega = len(omega_grid)
        tau_grid_fft = np.linspace(-T_max, T_max, n_omega)
        dt_fft = tau_grid_fft[1] - tau_grid_fft[0]

        t_i, t_j = np.meshgrid(time_grid, time_grid, indexing='ij')
        tau_matrix = t_i - t_j
        tau_idx_matrix = np.round((tau_matrix - tau_grid_fft[0]) / dt_fft).astype(int)
        tau_idx_matrix = np.clip(tau_idx_matrix, 0, n_omega - 1)
        
        f_ga_freq = product_tau_raw[tau_idx_matrix]
        
        # Time domain: precise_convolution_right
        ga = initial_state._r2a()
        f_row = evolution.thermal_dist[-1:, :]
        F_row = evolution.thermal_integral[-1:, :]
        thermal_sum_left_row = evolution.thermal_sum_left[-1:, :]
        
        f_ga_time_row = ga.precise_convolution_right(
            f_row, F_row, dt,
            self_index=-1, precomputed_sum=thermal_sum_left_row, gap_tensor=None
        )
        
        f_ga_time = f_ga_time_row.trace(pauli_index=2) / 2
        f_ga_time = np.array(f_ga_time[0, :])
        
        # Compare
        f_ga_freq_row = f_ga_freq[-1, :]
        error_ga = np.abs(f_ga_freq_row - f_ga_time)
        
        print(f"  Max error: {np.max(error_ga):.6e}")
        print(f"  Mean error: {np.mean(error_ga):.6e}")
        print(f"  Error/dt: {np.max(error_ga)/dt:.6f}")
        
        results['f_ga'] = {
            'freq': f_ga_freq_row,
            'time': f_ga_time,
            'error': error_ga,
            'max_error': np.max(error_ga)
        }
    
    # Plotting
    if save_plots:
        import os
        os.makedirs('Test_plots', exist_ok=True)
        
        for key, title in [('gr_f', 'gr @ f'), ('f_ga', 'f @ ga')]:
            if key not in results:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Real
            ax = axes[0, 0]
            ax.plot(time_grid, np.real(results[key]['freq']), 'b-', lw=2, 
                   label='Frequency', alpha=0.7, marker='o', markevery=20)
            ax.plot(time_grid, np.real(results[key]['time']), 'r--', lw=2,
                   label='Time (precise_conv)', alpha=0.7, marker='^', markevery=20)
            ax.set_xlabel("t'")
            ax.set_ylabel(f'Real[({title})_2]')
            ax.set_title(f'{title}: Real (N_t={N_t}, dt={dt:.5f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Imag
            ax = axes[0, 1]
            ax.plot(time_grid, np.imag(results[key]['freq']), 'b-', lw=2,
                   label='Frequency', alpha=0.7, marker='o', markevery=20)
            ax.plot(time_grid, np.imag(results[key]['time']), 'r--', lw=2,
                   label='Time (precise_conv)', alpha=0.7, marker='^', markevery=20)
            ax.set_xlabel("t'")
            ax.set_ylabel(f'Imag[({title})_2]')
            ax.set_title(f'{title}: Imag')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Error (log)
            ax = axes[1, 0]
            ax.semilogy(time_grid, results[key]['error'] + 1e-20, 'k-', lw=2,
                       marker='s', markevery=20, ms=4)
            ax.set_xlabel("t'")
            ax.set_ylabel('|Error|')
            ax.set_title(f'{title}: Error (max={np.max(results[key]["error"]):.3e})')
            ax.grid(True, alpha=0.3)
            
            # Error (linear, zoomed)
            ax = axes[1, 1]
            ax.plot(time_grid, results[key]['error'], 'k-', lw=2,
                   marker='s', markevery=20, ms=4)
            ax.set_xlabel("t'")
            ax.set_ylabel('|Error|')
            ax.set_title(f'{title}: Error (linear)')
            ax.set_xlim(-5, 0)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fname = f'Test_plots/convolution_check_{key}.png'
            plt.savefig(fname, dpi=150)
            print(f"  Saved {fname}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for key, name in [('gr_f', 'gr @ f'), ('f_ga', 'f @ ga')]:
        if key in results:
            print(f"\n{name}:")
            print(f"  Max error: {results[key]['max_error']:.6e}")
            print(f"  Error/dt: {results[key]['max_error']/dt:.6f}")
    
    print("="*80)
    return results


if __name__ == "__main__":
    results = check_convolution_accuracy(
        T_max=2*np.pi*5, N_t=201, temperature=0.1, 
        check_type='both', save_plots=True
    )
