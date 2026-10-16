"""
State object class for Keldysh formalism.
Stores retarded Green's function, Keldysh Green's function, and all system properties.
"""

import copy
import numpy as np
from nambu_keldysh_class import NambuKeldyshTensor, get_pauli_matrix


class StateObject:
    """
    Container for complete system state in Keldysh formalism.
    Stores Green's functions and derived quantities.
    """

    def __init__(self, gr, gk, bcs_coupling_constant, temperature, grid_params=None, track_occupation=False):
        """
        Initialize state object with Green's functions and grid parameters.
        Args:
            gr: Retarded Green's function g^R (NambuKeldyshTensor)
            gk: Keldysh Green's function g^K (NambuKeldyshTensor)
            bcs_coupling_constant: renormalized BCS coupling constant λ' for gap equation 
            temperature: System temperature
            grid_params: Dictionary with grid parameters
            track_occupation: Whether to track occupation function
        """

        self.gr = gr
        self.gk = gk
        self.bcs_coupling_constant = bcs_coupling_constant
        self.temperature = temperature

        # Extract grid parameters
        if grid_params is not None:
            self.T_max = grid_params.get('time_duration', grid_params.get('T_max'))

            # Compute dt from time grid
            if 'dt' in grid_params:
                self.dt = grid_params['dt']
            elif 'time_sampling' in grid_params and 'time_duration' in grid_params:
                self.dt = grid_params['time_duration'] / (grid_params['time_sampling'] - 1)
            else:
                self.dt = None

            # Extract number of time points
            if 'time_sampling' in grid_params:
                self.ntpoints = grid_params['time_sampling']
            elif 'n_tpoints' in grid_params:
                self.ntpoints = grid_params['n_tpoints']
            else:
                # Infer from Green's function shape
                self.ntpoints = self.gr.data.shape[2]

            # Generate time grid
            if self.T_max is not None and self.ntpoints is not None:
                self.time_grid = np.linspace(-self.T_max, 0, self.ntpoints)
            else:
                self.time_grid = None
        else:
            self.T_max = None
            self.dt = None
            self.ntpoints = None
            self.time_grid = None

        # Initialize thermal distributions as None (computed on demand)
        self._thermal_dist = None
        self._thermal_integral = None

        #* initial occupation is thermal
        if track_occupation:
            self.occupation_function = 0 * self.gr
        else:
            self.occupation_function = None

    def copy(self):
        """
        Independent copy of the state: scalars and grid are shared, Green's functions get their own data.
        Used for trial evolution steps that must not modify the original state.
        """
        new_state = copy.copy(self)
        new_state.gr = self.gr.copy()
        new_state.gk = self.gk.copy()
        if self.occupation_function is not None:
            new_state.occupation_function = self.occupation_function.copy()
        return new_state

    # ========== Green's Function Relations ==========

    def _r2a(self):
        """
        Compute advanced Green's function from retarded.
        Uses involution: g^A = -(g^R)^†
        Returns:
            NambuKeldyshTensor: Advanced Green's function g^A
        """

        return -self.gr.involution() 

    # ========== State Properties ==========

    def get_gap_history(self):
        """
        Extract superconducting gap from Green's functions.

        Uses the gap equation: Δ(t) = -λ/4 Tr[τ₋ g^K(t,t)]
        where τ₋ = (τ₁ - iτ₂)/2 is the lowering operator.
        Returns:
            np.ndarray: Gap values Δ(t) at each time point
        """
        # Trace g^K over Nambu indices with τ₋
        gk_traced = self.gk.trace(pauli_index='-')

        # Gap equation: Δ = -λ/4 * Tr[τ₋ g^K(t,t)]
        gap_history = -0.25 * self.bcs_coupling_constant * np.diagonal(gk_traced) 

        return gap_history

    #** Skipped for now, not important for equilibrium 
    #~ def get_current_kernel_prefactor(self, thermal_dist, time_index=-1):

    def get_current_kernel_prefactor(self, thermal, time_index=-1): #! can be related to other terms probably
        """
        Compute current kernel prefactor: tr(K_(t,t)) for Crank-Nicolson update.

        This is the trace of the kernel matrix at equal time, needed for the
        Crank-Nicolson vector potential update (Eq. 262 in effective_circuit_model.tex).

        Kernel matrix (Eq. 211-215):
            K_(t,t) = τ₃ g'^R(t,t) τ₃ g'^K(t,t)
                    + τ₃ g'^K(t,t) τ₃ g'^A(t,t)
                    + 2τ₃ g'^R(t,t) F(t,t)
                    + 2F(t,t) τ₃ g'^A(t,t)
        Args:
            thermal: ThermalConvolutions object (numerical_support) holding f, F, thermal sums and the convolutions
            time_index: Time index to compute kernel at (default -1, supports negative indexing)

        Returns:
            complex: Trace of kernel matrix tr(K_(t,t))
        """
        # Handle negative indexing
        N_t = self.gr.data.shape[2]
        t_idx = time_index if time_index >= 0 else N_t + time_index

        # Define τ₃ Pauli matrix
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        # Get advanced Green's function
        ga = self._r2a()

        # Extract diagonal (equal-time) elements at time t_idx
        gr_diag = self.gr[t_idx, t_idx]      # g'^R(t, t) - shape (2,2)
        gk_diag = self.gk[t_idx, t_idx]      # g'^K(t, t) - shape (2,2)
        ga_diag = ga[t_idx, t_idx]           # g'^A(t, t) - shape (2,2)
        F_diag = thermal.thermal_dist[t_idx, t_idx]  # F(t, t) - shape (2,2)

        # Term 1: τ₃ g'^R(t,t) τ₃ g'^K(t,t)
        term1 = tau3 * gr_diag * tau3 * gk_diag

        # Term 2: τ₃ g'^K(t,t) τ₃ g'^A(t,t)
        term2 = tau3 * gk_diag * tau3 * ga_diag

        #? Do these term contribute in the end? g_3 around zero?
        # Term 3: 2τ₃ g'^R(t,t) F(t,t)
        term3 = 2.0 * tau3 * gr_diag * F_diag #! This needs to be modified by the shift with F_thermal_integral? but we set it to 0 arbitrarily?

        # Term 4: 2F(t,t) τ₃ g'^A(t,t)
        term4 = 2.0 * F_diag * tau3 * ga_diag

        # Sum all terms
        kernel_matrix = term1 + term2 + term3 + term4

        # Take trace over Nambu indices (pauli_index=0 gives identity component)
        kernel_trace = kernel_matrix.trace(pauli_index=0)

        return kernel_trace * (-1j * np.pi/4)/2.0 # diving by 2 because its the midpoint contribution
    
    #** Skipped for now, not important for equilibrium

    def get_current_at_time_t(self, A_history, thermal, time_index=-1, include_derivative = True):
        """
        Compute current J(t) at specific time with thermal distribution (tex Eq. 942-945).

        Formula: J(t) = -i(π /4) ∫ dt' Tr[
            τ₃ g'^R(t,t') A(t') τ₃ g'^K(t',t)
          + τ₃ g'^K(t,t') A(t') τ₃ g'^A(t',t)
          + 2τ₃ g'^R(t,t') A(t') F(t',t)
          + 2F(t,t') A(t') τ₃ g'^A(t',t)
        ]

        Terms 3 & 4 use the full thermal convolutions (f and F parts) of the ThermalConvolutions object

        Args:
            A_history: Vector potential A(t') - array of length N_t
            thermal: ThermalConvolutions object (numerical_support) holding f, F, thermal sums and the convolutions
            time_index: Time index to compute current at (default -1, supports negative indexing)

        Returns:
            complex: Current J(t) at specified time
        """

        if A_history is None:
            return 0.0
        # Handle negative indexing
        N_t = self.gr.data.shape[2]
        t_idx = time_index if time_index >= 0 else N_t + time_index

        # Define τ₃ Pauli matrix and A(t') as NambuKeldyshTensor
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        A_tensor = NambuKeldyshTensor(A_history, pauli_channel=0)

        # Get advanced Green's function
        ga = self._r2a()

        #* gap history of this state, needed for the F (gap) part of the full thermal convolutions
        gap_history = self.get_gap_history()
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        # Extract rows and columns for time t
        gr_row = self.gr[t_idx:t_idx+1, :]      # g'^R(t, :) shape (2,2,1,Nt)
        gk_row = self.gk[t_idx:t_idx+1, :]      # g'^K(t, :) shape (2,2,1,Nt)
        gk_col = self.gk[:, t_idx:t_idx+1]      # g'^K(:, t) shape (2,2,Nt,1)
        ga_col = ga[:, t_idx:t_idx+1]           # g'^A(:, t) shape (2,2,Nt,1)
   
        # Midpoint rule: subtract 1/2 weight from both endpoints
        inner_1 = A_tensor * tau3 * gk_col
        first_endpoint_1 = tau3 * (gr_row[:,0:1] * inner_1[0:1,:])[-1,-1]
        last_endpoint_1 = tau3 * (gr_row[:,-1:] * inner_1[-1:,:])[-1,-1]
        term1 = tau3 * (gr_row @ inner_1)[0,0] * self.dt - 0.5 * self.dt * first_endpoint_1 - 0.5 * self.dt * last_endpoint_1

        # Midpoint rule: subtract 1/2 weight from both endpoints
        inner_2 = A_tensor * tau3 * ga_col
        first_endpoint_2 = tau3 * (gk_row[:,0:1] * inner_2[0:1,:])[-1,-1]
        last_endpoint_2 = tau3 * (gk_row[:,-1:] * inner_2[-1:,:])[-1,-1]
        term2 = tau3 * (gk_row @ inner_2)[0,0] * self.dt - 0.5 * self.dt * first_endpoint_2 - 0.5 * self.dt * last_endpoint_2

        # Multiply gr with A*tau3, then precise_convolution_left with F (regularized)
        #~ term3 = 2.0 * (tau3 * gr_row * A_tensor).precise_convolution_left(thermal_dist, thermal_integral, self.dt, other_index=time_index, precomputed_sum=thermal_sum_right)[-1,-1]
        #~ term3 = 2.0 * thermal.convolve_thermal_dist_left(tau3 * gr_row * A_tensor, time_index)[-1,-1]
        #* full(X*tau3) = 2*dist(X) + int(X*tau3): same dist part as before plus the F (gap) part
        term3 = thermal.convolve_thermal_full_left(tau3 * gr_row * A_tensor * tau3, gap_tensor, time_index)[-1,-1]

        # Multiply ga with A*tau3, then precise_convolution_right with F (regularized)
        #~ term4 = 2.0 * (A_tensor * tau3 * ga).precise_convolution_right(F_row, F_integral_row, self.dt, self_index=time_index, precomputed_sum=thermal_sum_left_row)[-1,-1]
        #~ term4 = 2.0 * thermal.convolve_thermal_dist_right(A_tensor * tau3 * ga, time_index)[-1,-1]
        #* tau3 stands left of the kernel (2*tau3*F + T, T anticommutes with tau3), so it is applied outside the full convolution
        term4 = tau3 * thermal.convolve_thermal_full_right(A_tensor * tau3 * ga, gap_tensor, time_index)[-1,-1]

        # Sum all terms and take Nambu trace
        total = term1 + term2 + term3 + term4
        current = total.trace(pauli_index=0)

        # Apply prefactor -i(π / 4) [σ_n absorbed into normalization]
        
        current = -1j * np.pi / 4 * current 
        
        if include_derivative:
            #* second term is the anomalous term coming from combination of delta(t-t') and f(t-t') limit at zero
            #~ current += - np.gradient(A_history, self.dt)[t_idx]
            #* edge_order=2: at the newest time the one-sided difference is second order, A'(t) = (3A_n - 4A_(n-1) + A_(n-2))/(2dt); the first-order backward difference is A'(t - dt/2), which gave Im sigma = -omega dt/2 in the normal state
            current += - np.gradient(A_history, self.dt, edge_order=2)[t_idx]
        
        return current

    # ========== Utilities ==========

    def update_state_gr(self, new_gr_row, new_gr_diag):
        """
        Update g^R with new time step using sliding window.

        Updates only g^R by adding new row/column/diagonal and removing oldest.

        Args:
            new_gr_row: New row for g^R(t_new, t_j) - shape (2,2,N_t) or NambuKeldyshTensor
            new_gr_diag: New diagonal element g^R(t_new, t_new) - shape (2,2) or NambuKeldyshTensor
            new_gk_row: Not used (kept for signature compatibility)
            new_gk_diag: Not used (kept for signature compatibility)
        """
        N_t = self.gr.data.shape[2]

        # g^R column is zero due to causality (retarded function vanishes for t < t')
        gr_column_data = NambuKeldyshTensor(np.zeros((N_t), dtype=complex), pauli_channel=0)

        self.gr.update_entries(new_gr_row, gr_column_data, new_gr_diag)

    def update_state_gk(self, new_gk_row, new_gk_diag):
        """
        Update g^K with new time step using sliding window.

        Updates only g^K by adding new row/column/diagonal and removing oldest.

        Args:
            new_gr_row: Not used (kept for signature compatibility)
            new_gr_diag: Not used (kept for signature compatibility)
            new_gk_row: New row for g^K(t_new, t_j) - shape (2,2,N_t) or NambuKeldyshTensor
            new_gk_diag: New diagonal element g^K(t_new, t_new) - shape (2,2) or NambuKeldyshTensor
        """
        N_t = self.gk.data.shape[2]

        # g^K column computed from row using transformation: τ₃ @ (g^K_row)^† @ τ₃
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        gk_row_data = new_gk_row

        #Used to zero out the tau_3 component of gk, as it should be and avoide any error accumulation
        new_gk_diag = 1/2 * (new_gk_diag + new_gk_diag.involution())
        
        gk_column_data = gk_row_data.involution() 

        self.gk.update_entries(new_gk_row, gk_column_data, new_gk_diag)

    def update_state_object(self, new_gr_row, new_gr_diag, new_gk_row, new_gk_diag):
        """
        Update state object with new time step using sliding window.

        Updates both g^R and g^K by adding new row/column/diagonal and removing oldest.

        Args:
            new_gr_row: New row for g^R(t_new, t_j) - shape (2,2,N_t) or NambuKeldyshTensor
            new_gr_diag: New diagonal element g^R(t_new, t_new) - shape (2,2) or NambuKeldyshTensor
            new_gk_row: New row for g^K(t_new, t_j) - shape (2,2,N_t) or NambuKeldyshTensor
            new_gk_diag: New diagonal element g^K(t_new, t_new) - shape (2,2) or NambuKeldyshTensor
        """
        self.update_state_gr(new_gr_row, new_gr_diag)
        self.update_state_gk(new_gk_row, new_gk_diag)
        
    # ========== Consistency Checks ========== 

    def check_gr_normalization(self, t1_idx):
        """
        Verify g^R normalization at fixed t₁ for all t₂.

        Checks: ∫ dt' g'^R(t₁,t') g'^R(t',t₂) + g'^R(t₁,t₂)τ₃ + τ₃ g'^R(t₁,t₂) = 0

        Args:
            t1_idx: Index for t₁ time (supports negative indexing)

        Returns:
            errors: np.ndarray of shape (N_t,) with error norm at each t₂
            totals: np.ndarray of shape (4, N_t) with Pauli components of total violation
        """

        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)

        N_t = self.gr.data.shape[2]
        t1_pos = t1_idx if t1_idx >= 0 else N_t + t1_idx

        # Extract row at t1
        gr_row = self.gr[t1_pos:t1_pos+1, :]  # shape (2,2,1,Nt)

        # Compute convolution for all t2 using matmul
        convolution = (gr_row @ self.gr) * self.dt  # shape (2,2,1,Nt)

        # Compute commutator terms for all t2
        left_term = tau3 * gr_row  # shape (2,2,1,Nt)
        right_term = gr_row * tau3  # shape (2,2,1,Nt)

        # Total normalization violation for all t2
        # Applying the midpoint rule to the intregral itself
        total = convolution + left_term + right_term  - 1/2 * gr_row[-1:,t1_pos] * self.gr[-1:,:] * self.dt - 1/2 * gr_row[-1:,0:] * self.gr.diagonal_time() * self.dt 

        # Compute errors
        errors = np.sqrt(np.sum(np.abs(total.data)**2, axis=(0, 1)))[0, :]  # shape (Nt,)

        # Extract Pauli components
        totals = np.zeros((4, N_t), dtype=complex)
        for pauli_idx in range(4):
            pauli_component = total.trace(pauli_idx) / 2  # shape (1, Nt)
            totals[pauli_idx, :] = pauli_component[0, :]

        return errors, totals

    def check_keldysh_normalization(self, t1_idx, thermal):
        """
        Verify FDT normalization constraint at fixed t₁ for all t₂.

        Checks: ∫ g^R g^K + ∫ g^K g^A + [τ₃, g^K] + ∫ (g^R @ f) + ∫ (f @ g^A)= 0

        This relation comes from the FDT: g^K = g^R @ f - f @ g^A

        Args:
            t1_idx: Index for t₁ time (supports negative indexing)
            thermal: ThermalConvolutions object (numerical_support) holding f, F, thermal sums and the convolutions

        Returns:
            errors: np.ndarray of shape (N_t,) with error norm at each t₂
            totals: np.ndarray of shape (4, N_t) with Pauli components of total violation
            components: Dict with 'commutator', 'gr_gk_conv', 'gk_ga_conv', 'thermal_gr', 'thermal_ga' arrays (4, N_t)
        """

        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        ga = self._r2a()

        N_t = self.gr.data.shape[2]
        t1_pos = t1_idx if t1_idx >= 0 else N_t + t1_idx

        # Extract rows at t1
        gr_row = self.gr[t1_pos:t1_pos+1, :]  
        gk_row = self.gk[t1_pos:t1_pos+1, :] 

        # Compute gap_tensor and thermal_gap_term
        gap_history = self.get_gap_history()
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)

        #~ thermal_gap_term = (- thermal_integral * gap_tensor - gap_tensor * thermal_integral)
        thermal_gap_term = thermal.thermal_gap_term(gap_tensor)

        # Add thermal_gap_term commutator:
        thermal_gap_commutator = tau3 * thermal_gap_term[t1_pos:t1_pos+1, :] - thermal_gap_term[t1_pos:t1_pos+1, :] * tau3
        
        # commutator
        commutator = tau3 * gk_row - gk_row * tau3   + thermal_gap_commutator
        
        # First convolution: ∫ g^R(t1, t') g^K(t', t2) dt' for all t2

        # Apply midpoint rule to conv1
        gr_t1_0 = self.gr[t1_pos, 0:1]  
        gk_0_row = self.gk[0:1, :]  
        first_endpoint_1 = gr_t1_0 * gk_0_row  

        gr_t1_t1 = self.gr[t1_pos, t1_pos:t1_pos+1]  
        gk_t1_row = self.gk[t1_pos:t1_pos+1, :]  
        last_endpoint_1 = gr_t1_t1 * gk_t1_row  

        conv1 = (gr_row @ self.gk) * self.dt - 0.5 * self.dt * first_endpoint_1  - 0.5 * self.dt * last_endpoint_1

        thermal_gap_gr_conv = thermal.convolve_thermal_int_left(gr_row, gap_tensor, t1_idx)

        conv1 = conv1 + thermal_gap_gr_conv

        # Second convolution: ∫ g^K(t1, t') g^A(t', t2) dt' for all t2
        conv2 = (gk_row @ ga) * self.dt  # shape (2,2,1,Nt)
      
        gk_t1_0 = self.gk[t1_pos, 0]  # shape (2,2)
        ga_0_row = ga[0:1, :]  # shape (2,2,1,Nt)
        first_endpoint_2 = gk_t1_0 * ga_0_row  # gk[t1, 0] * ga[0, t2]

        #* upper end point t' = t2 of the right convolution (ga is advanced), was missing: last_endpoint_2 was undefined and conv2 had no end point weights
        last_endpoint_2 = gk_row * ga.diagonal_time()  # gk[t1, t2] * ga[t2, t2]
        conv2 = conv2 - 0.5 * self.dt * first_endpoint_2 - 0.5 * self.dt * last_endpoint_2

        # Extract diagonal elements of ga as row tensor
        thermal_gap_ga_conv = thermal.convolve_thermal_int_right(ga, gap_tensor, t1_idx)

        conv2 = conv2 + thermal_gap_ga_conv
       
        thermal_gr = thermal.convolve_thermal_dist_left(gr_row, t1_idx) * tau3 * 2
        thermal_ga = 2 * tau3 * thermal.convolve_thermal_dist_right(ga, t1_idx)

        # Save pure convolutions before adding thermal terms
        conv1_gk_only = (gr_row @ self.gk) * self.dt - 0.5 * self.dt * first_endpoint_1 - 0.5 * self.dt * last_endpoint_1
        conv2_gk_only = (gk_row @ ga) * self.dt - 0.5 * self.dt * first_endpoint_2 - 0.5 * self.dt * last_endpoint_2

        # Add thermal corrections to convolutions
        conv1 = conv1 + thermal_gr  # add gr @ (gr @ f) term
        conv2 = conv2 + thermal_ga  # Add (f @ ga) @ ga term

        # Total normalization violation for all t2
        total = commutator + conv1 + conv2  # shape (2,2,1,Nt)

        # Compute errors
        errors = np.sqrt(np.sum(np.abs(total.data)**2, axis=(0, 1)))[0, :]  # shape (Nt,)

        # Extract Pauli components
        totals = np.zeros((4, N_t), dtype=complex)
        commutators = np.zeros((4, N_t), dtype=complex)
        conv1_gk_onlys = np.zeros((4, N_t), dtype=complex)
        conv2_gk_onlys = np.zeros((4, N_t), dtype=complex)
        thermal_grs = np.zeros((4, N_t), dtype=complex)
        thermal_gas = np.zeros((4, N_t), dtype=complex)
        thermal_gap_gr_convs = np.zeros((4, N_t), dtype=complex)
        thermal_gap_ga_convs = np.zeros((4, N_t), dtype=complex)
        thermal_gap_commutators = np.zeros((4, N_t), dtype=complex)

        for pauli_idx in range(4):
            totals[pauli_idx, :] = (total.trace(pauli_idx) / 2)[0, :]
            commutators[pauli_idx, :] = ((tau3 * gk_row - gk_row * tau3).trace(pauli_idx) / 2)[0, :]
            conv1_gk_onlys[pauli_idx, :] = (conv1_gk_only.trace(pauli_idx) / 2)[0, :]
            conv2_gk_onlys[pauli_idx, :] = (conv2_gk_only.trace(pauli_idx) / 2)[0, :]
            thermal_grs[pauli_idx, :] = (thermal_gr.trace(pauli_idx) / 2)[0, :]
            thermal_gas[pauli_idx, :] = (thermal_ga.trace(pauli_idx) / 2)[0, :]
            thermal_gap_gr_convs[pauli_idx, :] = (thermal_gap_gr_conv.trace(pauli_idx) / 2)[0, :]
            thermal_gap_ga_convs[pauli_idx, :] = (thermal_gap_ga_conv.trace(pauli_idx) / 2)[0, :]
            thermal_gap_commutators[pauli_idx, :] = (thermal_gap_commutator.trace(pauli_idx) / 2)[0, :]

        return errors, totals, {
            'commutator': commutators,
            'gr_gk_conv_pure': conv1_gk_onlys,
            'gk_ga_conv_pure': conv2_gk_onlys,
            'thermal_gr': thermal_grs,
            'thermal_ga': thermal_gas,
            'thermal_gap_gr_conv': thermal_gap_gr_convs,
            'thermal_gap_ga_conv': thermal_gap_ga_convs,
            'thermal_gap_commutator': thermal_gap_commutators
        }
    
    def check_fdt(self, thermal, time_index):
        """
        Check FDT relation: g^K = g^R @ f - f @ g^A - thermal_gap_term using precise convolution.

        The thermal_gap_term represents the direct shift to g^K:
        thermal_gap_term = -(F * Δ + Δ * F)

        Computes the regularized FDT convolution for a specific time index by:
        1. Computing regularized gr @ f using precise_convolution_left
        2. Computing regularized f @ ga using precise_convolution_right
        3. Subtracting thermal_gap_term directly (not convolved)
        4. Combining as: gk = (gr @ f) - (f @ ga) - thermal_gap_term

        Args:
            thermal: ThermalConvolutions object (numerical_support) holding f, F, thermal sums and the convolutions
            time_index: int - time index to check (supports negative indexing)

        Returns:
            gk_fdt_row: NambuKeldyshTensor - FDT prediction for g^K[time_index, :]
            gk_actual_row: NambuKeldyshTensor - actual g^K[time_index, :]
            error_row: NambuKeldyshTensor - difference between actual and FDT prediction
            max_error: float - maximum absolute error for this row
        """
        if self.dt is None:
            raise ValueError("Time step dt must be set in grid_params to use check_fdt")

        # Compute advanced Green's function
        ga = self._r2a()

        # Handle negative indexing
        N_t = self.gr.data.shape[2]
        t_idx = time_index if time_index >= 0 else N_t + time_index

        # Compute gap_tensor and thermal_gap_term (the direct shift)
        gap_history = self.get_gap_history()
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)
        #~ thermal_gap_term = (- f_thermal_integral * gap_tensor - gap_tensor * f_thermal_integral)
        thermal_gap_term = thermal.thermal_gap_term(gap_tensor)

        # Extract rows for time t_idx
        gr_row = self.gr[t_idx:t_idx+1, :]
        
        term1 = thermal.convolve_thermal_dist_left(gr_row, time_index)

        term2 = thermal.convolve_thermal_dist_right(ga, time_index)

        gk_fdt_row = term1 - term2 - thermal_gap_term[t_idx:t_idx+1, :]

        gk_actual_row = self.gk[t_idx:t_idx+1, :]

        error_row = gk_actual_row - gk_fdt_row
        max_error = np.max(np.abs(error_row.data))

        return gk_fdt_row, gk_actual_row, error_row, max_error

    # ========== Fourier Transform Methods ==========
    def energy_time_representation(self, green_function_type, eta=None):
        """
        Transform Green's function to energy representation.

        Extracts anti-diagonal elements in time-time plane and Fourier transforms:
        ĝ(ε) = ∫ dτ e^{i2ετ} ĝ(t=N_t-1-i, t'=i)


        Args:
            green_function_type: 'gr', 'gk', or 'f' (occupation function)
            eta: Inelastic scattering rate for exponential damping (default: None = no damping)
                 Physically represents quasiparticle lifetime τ = 1/η
                 Suppresses endpoint discontinuities and broadens peaks by η in frequency

        Returns:
            dict: {
                'energy_grid': Energy points (1D array),
                'g_energy': NambuKeldyshTensor with shape (2, 2, N_ε)
            }
        """
        
        if self.dt is None:
            raise ValueError("Time step dt must be set in grid_params")

        # Select the appropriate Green's function
        if green_function_type == 'gr':
            g_two_time = self.gr
        elif green_function_type == 'gk':
            g_two_time = self.gk
        elif green_function_type == 'f':
            g_two_time = self.occupation_function
        else:
            raise ValueError(f"Invalid green_function_type: {green_function_type}")

        # Get dimensions
        N_t = g_two_time.data.shape[2]

        # Extract anti-diagonal in time: g[:, :, N_t-1-i, i] for all i
        # Define F(τ) = g(t+τ, t-τ) where τ = (i-j)·dt/2
        # For anti-diagonal: τ_k = (N_t-1-2k)·dt/2 with spacing Δτ = dt
        g_offdiag = g_two_time.off_diagonal()

        # Reverse along time axis so τ increases with index (required for FFT convention)
        # After reversal: τ from -(N_t-1)·dt/2 to +(N_t-1)·dt/2 with spacing dt
        g_offdiag_reversed = np.flip(g_offdiag, axis=2)
        # Subtract asymptotic boundary values to remove wrap-around discontinuity
        #g_asymptotic = (g_offdiag_reversed[:, :, :1] + g_offdiag_reversed[:, :, -1:]) / 2.0
        g_offdiag_regularized = g_offdiag_reversed #- g_asymptotic

        # Standard Fourier transform: G(ω) = ∫ dτ e^{iω·τ} F(τ) with τ spacing = dt
        # Then g(ε,t) = G(2ε), so ε = ω/2
        # ifftshift centers τ=0 at index 0 (FFT convention) before transforming
        g_fft = np.fft.ifft(np.fft.ifftshift(g_offdiag_regularized, axes=2), axis=2) * N_t
        g_fft *= self.dt  # Multiply by dτ = dt for integral approximation

        # Shift to center zero frequency
        g_energy_data = np.fft.fftshift(g_fft, axes=2)

        # Create NambuKeldyshTensor (keep all frequency points)
        g_energy = NambuKeldyshTensor(g_energy_data) * 2 #[::2] 

        # Construct energy grid: standard FFT gives ω, then ε = ω/2
        # Use d=dt (spacing in τ), then divide by 2 for ε
        freq = np.fft.fftfreq(N_t, d=self.dt)
        omega_grid = 2 * np.pi * freq  # ω = 2πf
        energy_grid = omega_grid/2  # ε/2 (from g(ε,t) = G(2ε))
        energy_grid = np.fft.fftshift(energy_grid)#[::2]

        return { 'energy_grid': energy_grid, 'g_energy': g_energy}

    #** Skipped for now, will check later, apparently this has also been updated
    def update_state_occupation(self, thermal):
        """
        Compute and update occupation distribution n(t,t') at specified time.

        Solves: (τ₃ + Δt·g^R(t,t)) * n + n * (τ₃ - Δt·g^A(t',t')) = source
        where source = g^K - Δt·g^R⊗(n+f) + Δt·(n+f)⊗g^A

        Args:
            thermal: ThermalConvolutions object (numerical_support) holding f, F, thermal sums and the convolutions;
                     the full (f and F) convolutions are used for the thermal terms

        Updates:
            self.occupation_function[time_index, :] with computed n(t,t') row
        """

        #* takes advantage of the fact that gr(t,t) and ga(t',t') have to be non tau_3 or tau_0 which is the jump condition
        time_index = -1
        N_t = self.gr.data.shape[2]
        t_idx = time_index % N_t

        ga = self._r2a()

        gr_row = self.gr[t_idx:t_idx+1, :]
        gk_row = self.gk[t_idx:t_idx+1, :]

        ##* full(X*tau3) = 2*dist(X) + int(X*tau3): 1/2 of it keeps the dist part exactly and adds half the F (gap) part
        tau3 = NambuKeldyshTensor(1.0, pauli_channel=3)
        gap_history = self.get_gap_history()
        gap_tensor = NambuKeldyshTensor(np.real(gap_history), pauli_channel=2) + NambuKeldyshTensor(np.imag(gap_history), pauli_channel=1)
        
        #* g'' = [g^R*f - f*g^A - T] + [g^R*n - n*g^A + {tau3, n}]: the first bracket (f and the thermal gap term T) is the FDT prediction of check_fdt,
        #* T enters as a local term and not as a convolution
        rhs_vector = (gk_row - thermal.convolve_thermal_dist_left(gr_row, time_index) + thermal.convolve_thermal_dist_right(ga, time_index)
                      + thermal.thermal_gap_term(gap_tensor)[t_idx:t_idx+1, :])
       
        #* gr, gk are already on the new time grid but the stored occupation is still on the old one: shift it by one step in both times,
        #* new[m, j] = old[m+1, j+1], the new last row and column (time t_N, still to be solved) are zero
        occupation_new_grid = self.occupation_function.shift(-1, axis=0).shift(-1, axis=1)
        rhs_vector += - gr_row[:,:-1] @ occupation_new_grid[:-1,:] * self.dt
        solution_tensor = self.gr[-1,-1:] * 0
        for time in range(N_t):
            source_term = rhs_vector[-1,time] 
            if time != 0:
                source_term += solution_tensor[1:] @ ga[:time,time] * self.dt
            
            solution_tensor.append_right([source_term.trace(3) / 4.0, 0.0, 0.0, source_term.trace(0) / 4.0])
        
        #* assuming also same symmetry as gk, which has been "prooved" in the note 
        #* since there is only tau_3 and tau_0 components this would imply --> it means the diagonal is purely real
        #* typically f is purely imaginary, due to oddness so this would make the diagonal zero, but this is unclear in most generality
        #* assuming gk is purely imaginary and gr purely real this would be true
        solution_tensor.append_right([0.0,0.0,0.0,0.0])

        #~ self.occupation_function.update_entries(solution_tensor[1:-1], solution_tensor[1:-1].involution(), solution_tensor[-1])
        #* the occupation is on the new grid already (occupation_new_grid), write the solved row n(t_N, t_j), j < N-1, into its last row and the
        #* involution into its last column; the diagonal n(t_N, t_N) stays zero from the shift (the solved diagonal value is not used)
        occupation_new_row = solution_tensor[1:N_t]
        occupation_new_grid.data[:, :, -1, :-1] = occupation_new_row.data
        occupation_new_grid.data[:, :, :-1, -1] = occupation_new_row.involution().data
        self.occupation_function = occupation_new_grid

    # ========== String Representation ==========

    def __str__(self):
        """String representation showing state properties."""
        gap_history = self.get_gap_history()
        gap_str = f"Gap(t_final) = {gap_history[-1]:.6f}"
        current_history = self.get_current_history()
        current_str = f"Current(t_final) = {current_history[-1]:.6f}"

        return f"StateObject:\n  {gap_str}\n  {current_str}\n  Shape: {self.gr.data.shape if self.gr is not None else 'N/A'}"

    # ========== Cleanup ==========

    def __del__(self):
        """Clean up resources."""
        pass
