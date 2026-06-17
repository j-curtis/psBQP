"""
State object class for Keldysh formalism.
Stores retarded Green's function, Keldysh Green's function, and all system properties.
"""

import numpy as np
from nambu_keldysh_class import NambuKeldyshTensor, get_pauli_matrix


class StateObject:
    """
    Container for complete system state in Keldysh formalism.
    Stores Green's functions and derived quantities.
    """

    def __init__(self, gr, gk, bcs_coupling_constant, grid_params=None, track_occupation=False):
        """
        Initialize state object with Green's functions and grid parameters.

        Args:
            gr: Retarded Green's function g^R (NambuKeldyshTensor)
            gk: Keldysh Green's function g^K (NambuKeldyshTensor)
            grid_params: Dictionary with grid parameters
            bcs_coupling_constant: BCS coupling constant λ for gap equation
        """
        self.gr = gr
        self.gk = gk
        self.bcs_coupling_constant = bcs_coupling_constant

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
        else:
            self.T_max = None
            self.dt = None

        #* initial occupation is thermal 
        if track_occupation:
            self.occupation_function = 0 * self.gr
        else:
            self.occupation_function = None
    # ========== Green's Function Relations ==========

    def _r2a(self):
        """
        Compute advanced Green's function from retarded.

        Uses involution: g^A = -(g^R)^†

        Returns:
            NambuKeldyshTensor: Advanced Green's function g^A
        """
        ga = -self.gr.involution()

        return ga #NambuKeldyshTensor(ga.data)

    # ========== State Properties ==========

    def get_gap_history(self):
        """
        Extract superconducting gap from Green's functions.

        Uses the gap equation: Δ(t) = λ Tr[τ₋ g^K(t,t)]
        where τ₋ = (τ₁ - iτ₂)/2 is the lowering operator.

        Returns:
            np.ndarray: Gap values Δ(t) at each time point
        """
        # Trace g^K over Nambu indices with lowering operator τ₋
        # This reduces (2, 2, N_t, N_t) -> (N_t, N_t)
        gk_traced = self.gk.trace(pauli_index='-')

        # Extract equal-time values g^K(t,t) using diagonal
        gk_diag = np.diagonal(gk_traced)

        # Gap equation: Δ = -λ/4 * Tr[τ₋ g^K(t,t)]
        gap_history = -0.25 * self.bcs_coupling_constant * gk_diag

        return gap_history

    def get_current_at_time_t(self, A_history, thermal_dist, thermal_integral, time_index = -1):
        """
        Compute current J(t) at specific time with thermal distribution (tex Eq. 942-945).

        Formula: J(t) = -i(π /4) ∫ dt' Tr[
            τ₃ g'^R(t,t') A(t') τ₃ g'^K(t',t)
          + τ₃ g'^K(t,t') A(t') τ₃ g'^A(t',t)
          + 2τ₃ g'^R(t,t') A(t') F(t',t)
          + 2F(t,t') A(t') τ₃ g'^A(t',t)
        ]

        Uses precise_convolution for thermal terms (3 & 4) to suppress errors

        Args:
            A_history: Vector potential A(t') - array of length N_t
            thermal_dist: Thermal distribution F(t,t') - NambuKeldyshTensor
            thermal_integral: Integral of F - NambuKeldyshTensor
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

        # Extract rows and columns for time t
        gr_row = self.gr[t_idx:t_idx+1, :]      # g'^R(t, :) shape (2,2,1,Nt)
        gk_row = self.gk[t_idx:t_idx+1, :]      # g'^K(t, :) shape (2,2,1,Nt)
        gk_col = self.gk[:, t_idx:t_idx+1]      # g'^K(:, t) shape (2,2,Nt,1)
        ga_col = ga[:, t_idx:t_idx+1]           # g'^A(:, t) shape (2,2,Nt,1)
        F_col = thermal_dist[:, t_idx:t_idx+1]  # F(:, t) shpe (2,2,Nt,1)
        F_row = thermal_dist[t_idx:t_idx+1, :]  # F(t, :) shape (2,2,1,Nt)

        # Thermal integrals for precise_convolution
        F_integral_row = thermal_integral[t_idx:t_idx+1, :]  # ∫F(t,:)

        # Term 1: ∫ dt' τ₃ g'^R(t,t') A(t') τ₃ g'^K(t',t)
        # = τ₃ [g'^R(t,:) @ (A(:) τ₃ g'^K(:,t))]=
        # Midpoint rule: subtract 1/2 weight from both endpoints
        inner_1 = A_tensor * tau3 * gk_col
        first_endpoint_1 = tau3 * (gr_row[:,0:1] * inner_1[0:1,:])[-1,-1]
        last_endpoint_1 = tau3 * (gr_row[:,-1:] * inner_1[-1:,:])[-1,-1]
        term1 = tau3 * (gr_row @ inner_1)[0,0] * self.dt - 0.5 * self.dt * first_endpoint_1 - 0.5 * self.dt * last_endpoint_1

        # Term 2: ∫ dt' τ₃ g'^K(t,t') A(t') τ₃ g'^A(t',t)
        # = τ₃ [g'^K(t,:) @ (A(:) τ₃ g'^A(:,t))]
        # Midpoint rule: subtract 1/2 weight from both endpoints
        inner_2 = A_tensor * tau3 * ga_col
        first_endpoint_2 = tau3 * (gk_row[:,0:1] * inner_2[0:1,:])[-1,-1]
        last_endpoint_2 = tau3 * (gk_row[:,-1:] * inner_2[-1:,:])[-1,-1]
        term2 = tau3 * (gk_row @ inner_2)[0,0] * self.dt - 0.5 * self.dt * first_endpoint_2 - 0.5 * self.dt * last_endpoint_2

        # Term 3: ∫ dt' 2τ₃ g'^R(t,t') A(t') F(t',t)
        # Multiply gr with A*tau3, then precise_convolution_left with F (regularized)
        #print((A_tensor * tau3 * gr_row).shape)
        term3 = 2.0 * (tau3 * gr_row * A_tensor).precise_convolution_left(thermal_dist, thermal_integral, self.dt, other_index=t_idx)[-1,-1]

        # Term 4: ∫ dt' 2F(t,t') A(t') τ₃ g'^A(t',t)
        # Multiply ga with A*tau3, then precise_convolution_right with F (regularized)
        term4 = 2.0 * (A_tensor * tau3 * ga).precise_convolution_right(F_row, F_integral_row, self.dt, self_index=t_idx)[-1,-1]

        # Sum all terms and take Nambu trace
        total = term1 + term2 + term3 + term4
        current = total.trace(pauli_index=0)

        # Apply prefactor -i(π / 4) [σ_n absorbed into normalization]
        #* second term is the anomalous term coming from combination of delta(t-t') and f(t-t') limit at zero
        current = -1j * np.pi / 4 * current - np.gradient(A_history, self.dt)[t_idx]

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
        #* seems weird that the equilibrium breaks the convolutionn even at 0  -- diagonal is tau_2 only, meaning left+right = 0
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
        #* key thing here is that the first element needs to be seriously taken into account when doing the midpoint rule
        total = convolution + left_term + right_term  - 1/2 * gr_row[-1:,t1_pos] * self.gr[-1:,:] * self.dt - 1/2 * gr_row[-1:,0:] * self.gr.diagonal_time() * self.dt # shape (2,2,1,Nt) 

        # Compute errors
        errors = np.sqrt(np.sum(np.abs(total.data)**2, axis=(0, 1)))[0, :]  # shape (Nt,)

        # Extract Pauli components
        totals = np.zeros((4, N_t), dtype=complex)
        for pauli_idx in range(4):
            pauli_component = total.trace(pauli_idx) / 2  # shape (1, Nt)
            totals[pauli_idx, :] = pauli_component[0, :]

        return errors, totals

    def check_keldysh_normalization(self, t1_idx, thermal_dist, thermal_integral):
        """
        Verify FDT normalization constraint at fixed t₁ for all t₂.

        Checks: ∫ g^R g^K + ∫ g^K g^A + [τ₃, g^K] + ∫ (g^R @ f) + ∫ (f @ g^A)= 0

        This relation comes from the FDT: g^K = g^R @ f - f @ g^A

        Args:
            t1_idx: Index for t₁ time (supports negative indexing)
            thermal_dist: Thermal distribution f(t,t') - NambuKeldyshTensor
            thermal_integral: Integral of thermal distribution F(t,t') - NambuKeldyshTensor

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
        gr_row = self.gr[t1_pos:t1_pos+1, :]  # shape (2,2,1,Nt)
        gk_row = self.gk[t1_pos:t1_pos+1, :]  # shape (2,2,1,Nt)

        # Commutator term: [τ₃, g^K(t1, t2)] for all t2
        commutator = tau3 * gk_row - gk_row * tau3  # shape (2,2,1,Nt)
        
        # First convolution: ∫ g^R(t1, t') g^K(t', t2) dt' for all t2
        conv1 = (gr_row @ self.gk) * self.dt  # shape (2,2,1,Nt) #* goes up to t for t' so we sum over all of them from -infty to t, i.e. full matrix

        # Apply midpoint rule to conv1
        # Integration from t'=0 to t'=t1 for all t2
        # First endpoint: gr[t1, 0] * gk[0, t2]
        # Last endpoint: gr[t1, t1] * gk[t1, t2]
        gr_t1_0 = self.gr[t1_pos, 0:1]  # shape (2,2)
        gk_0_row = self.gk[0:1, :]  # shape (2,2,1,Nt)
        first_endpoint_1 = gr_t1_0 * gk_0_row  # gr[t1, 0] * gk[0, t2]

        gr_t1_t1 = self.gr[t1_pos, t1_pos:t1_pos+1]  # shape (2,2,1)
        gk_t1_row = self.gk[t1_pos:t1_pos+1, :]  # shape (2,2,1,Nt)
        last_endpoint_1 = gr_t1_t1 * gk_t1_row  # gr[t1, t1] * gk[t1, t2]

        conv1 = conv1 - 0.5 * self.dt * first_endpoint_1  - 0.5 * self.dt * last_endpoint_1

        # Second convolution: ∫ g^K(t1, t') g^A(t', t2) dt' for all t2
        conv2 = (gk_row @ ga) * self.dt  # shape (2,2,1,Nt) 
        #* goes up to t', however what happens is that for elements with first element bigger than t', the integral is cut-off
        #* this means that the 1/2 midpoint rule is different!!! 
        #* carefully compute the ga diagonal and which elements come in!!
     
        gk_t1_0 = self.gk[t1_pos, 0]  # shape (2,2)
        ga_0_row = ga[0:1, :]  # shape (2,2,1,Nt)
        first_endpoint_2 = gk_t1_0 * ga_0_row  # gk[t1, 0] * ga[0, t2]

        # Extract diagonal elements of ga as row tensor
        last_endpoint_2 = gk_row * ga.diagonal_time()  # gk[t1, t2] * ga[t2, t2]

        conv2 = conv2 - 0.5 * self.dt * first_endpoint_2 - 0.5 * self.dt * last_endpoint_2
        
        # Thermal term 1: gr_row @ (gr @ f) for all t2 at once
        thermal_gr = gr_row.precise_convolution_left(thermal_dist, thermal_integral, self.dt, other_index=t1_pos) * tau3 * 2

        # Thermal term 2: (f_row @ ga) @ ga for all t2 at once
        f_row = thermal_dist[t1_pos:t1_pos+1, :]
        F_row = thermal_integral[t1_pos:t1_pos+1, :]
        thermal_ga = 2 * tau3 * ga.precise_convolution_right(f_row, F_row, self.dt, self_index=t1_pos)

        # Save pure convolutions before adding thermal terms
        conv1_pure = conv1  # Pure ∫ g^R g^K (without thermal)
        conv2_pure = conv2  # Pure ∫ g^K g^A (without thermal)

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
        conv1_pures = np.zeros((4, N_t), dtype=complex)
        conv2_pures = np.zeros((4, N_t), dtype=complex)
        thermal_grs = np.zeros((4, N_t), dtype=complex)
        thermal_gas = np.zeros((4, N_t), dtype=complex)

        for pauli_idx in range(4):
            totals[pauli_idx, :] = (total.trace(pauli_idx) / 2)[0, :]
            commutators[pauli_idx, :] = (commutator.trace(pauli_idx) / 2)[0, :]
            conv1_pures[pauli_idx, :] = (conv1_pure.trace(pauli_idx) / 2)[0, :]
            conv2_pures[pauli_idx, :] = (conv2_pure.trace(pauli_idx) / 2)[0, :]
            thermal_grs[pauli_idx, :] = (thermal_gr.trace(pauli_idx) / 2)[0, :]
            thermal_gas[pauli_idx, :] = (thermal_ga.trace(pauli_idx) / 2)[0, :]

        return errors, totals, {
            'commutator': commutators,
            'gr_gk_conv_pure': conv1_pures,
            'gk_ga_conv_pure': conv2_pures,
            'thermal_gr': thermal_grs,
            'thermal_ga': thermal_gas
        }

    def check_fdt(self, f_thermal, f_thermal_integral, time_index):
        """
        Check FDT relation: g^K = g^R @ f - f @ g^A using precise convolution.

        Computes the regularized FDT convolution for a specific time index by:
        1. Computing regularized gr @ f using precise_convolution_left
        2. Computing regularized f @ ga using precise_convolution_right
        3. Combining as: term1 - term2

        Args:
            f_thermal: NambuKeldyshTensor - thermal distribution f(t, t')
            f_thermal_integral: NambuKeldyshTensor - integral of thermal distribution F(t, t')
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

        # Extract rows for time t_idx
        gr_row = self.gr[t_idx:t_idx+1, :]
        f_row = f_thermal[t_idx:t_idx+1, :]
        F_row = f_thermal_integral[t_idx:t_idx+1, :]

        # First term: regularized gr @ f (f is regularized, on the right)
        # Pass full f_thermal_integral tensor so method can extract correct row t_idx
        term1 = gr_row.precise_convolution_left(f_thermal, f_thermal_integral, self.dt, other_index=t_idx)

        # Second term: regularized f @ ga (f is regularized, on the left)
        # Pass t_idx (positive index) for correct row extraction
        term2 = ga.precise_convolution_right(f_row, F_row, self.dt, self_index=t_idx)

        # FDT relation: g^K = term1 - term2
        gk_fdt_row = term1 - term2

        # Extract actual g^K row
        gk_actual_row = self.gk[t_idx:t_idx+1, :]

        # Compute error
        error_row = gk_actual_row - gk_fdt_row
        max_error = np.max(np.abs(error_row.data))

        return gk_fdt_row, gk_actual_row, error_row, max_error

    # ========== Fourier Transform Methods ==========

    def energy_time_representation(self, green_function_type):
        """
        Transform Green's function to energy representation.

        Extracts anti-diagonal elements in time-time plane and Fourier transforms:
        ĝ(ε) = ∫ dτ e^{i2ετ} ĝ(t=N_t-1-i, t'=i)

        Preserves full Nambu matrix structure (2x2).

        Args:
            green_function_type: 'gr', 'gk', or 'f' (occupation function)

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
        # Shape: (2, 2, N_t)
        # Note: Anti-diagonal has τ spacing of 2*dt and is in reversed order (τ decreasing)
        g_offdiag = g_two_time.off_diagonal()

        # Reverse along time axis so τ increases with index (required for FFT convention)
        g_offdiag_reversed = np.flip(g_offdiag, axis=2)

        # Fourier transform along time axis (axis=2)
        # ∫ dτ e^{iετ} g(τ) with τ spacing = 2*dt
        g_fft = np.fft.ifft(g_offdiag_reversed, axis=2) * N_t  # Remove 1/N normalization
        g_fft *= (2 * self.dt)  # Multiply by dτ = 2*dt for integral approximation

        # Shift to center zero frequency
        g_energy_data = np.fft.fftshift(g_fft, axes=2)

        # Create NambuKeldyshTensor
        g_energy = NambuKeldyshTensor(g_energy_data)

        # Construct energy grid accounting for 2*dt spacing
        freq = np.fft.fftfreq(N_t, d=self.dt)
        energy_grid = 2 * np.pi * freq  # ω = 2πf (factor of 2 from 2*dt spacing already in freq scale)
        energy_grid = np.fft.fftshift(energy_grid)

        return { 'energy_grid': energy_grid, 'g_energy': g_energy}

    def update_state_occupation(self, f_thermal, f_thermal_integral):
        """
        Compute and update occupation distribution n(t,t') at specified time.

        Solves: (τ₃ + Δt·g^R(t,t)) * n + n * (τ₃ - Δt·g^A(t',t')) = source
        where source = g^K - Δt·g^R⊗(n+f) + Δt·(n+f)⊗g^A

        Args:
            f_thermal: Thermal distribution F(t,t') - NambuKeldyshTensor (2,2,N_t,N_t)
            f_thermal_integral: Integral of F - NambuKeldyshTensor (2,2,N_t,N_t)
            time_index: Time index to compute (default -1)

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

        n_plus_f_full = self.occupation_function + f_thermal

        rhs_vector = gk_row - gr_row.precise_convolution_left(f_thermal, f_thermal_integral[t_idx:t_idx+1, :], self.dt) + ga.precise_convolution_right(f_thermal[t_idx:t_idx+1, :], f_thermal_integral[t_idx:t_idx+1, :], self.dt)
        rhs_vector += - gr_row[:,:-1] @ self.occupation_function[:-1,:]
        solution_tensor = self.gr[-1,-1:] * 0
        for time in range(N_t):
            source_term = rhs_vector[-1,time] 
            if time != 0:
                source_term += solution_tensor[1:] @ ga[:time,time]
            
            solution_tensor.append_right([source_term.trace(3) / 4.0, 0.0, 0.0, source_term.trace(0) / 4.0])
        
        #* assuming also same symmetry as gk, which has been "prooved" in the note 
        #* since there is only tau_3 and tau_0 components this would imply --> it means the diagonal is purely real
        #* typically f is purely imaginary, due to oddness so this would make the diagonal zero, but this is unclear in most generality
        #* assuming gk is purely imaginary and gr purely real this would be true
        solution_tensor.append_right([0.0,0.0,0.0,0.0])

        self.occupation_function.update_entries(solution_tensor[1:-1], solution_tensor[1:-1].involution(), solution_tensor[-1])

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
