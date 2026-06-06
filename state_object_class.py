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

    def __init__(self, gr, gk, bcs_coupling_constant, grid_params=None):
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

    # ========== Green's Function Relations ==========

    def _r2a(self):
        """
        Compute advanced Green's function from retarded.

        Uses involution: g^A = -(g^R)^†

        Returns:
            NambuKeldyshTensor: Advanced Green's function g^A
        """
        ga = -self.gr.involution()

        #* removed zeroing of the diagonal since it doesnt seem to be correct, however one has to be careful which g^a we are calling!
        # Zero out diagonal of g^A to break F(0) cancellation
        #diag_indices = np.diag_indices(ga.data.shape[2])
        #ga.data[:, :, diag_indices[0], diag_indices[1]] = 0.0

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
        F_col = thermal_dist[:, t_idx:t_idx+1]  # F(:, t) shape (2,2,Nt,1)
        F_row = thermal_dist[t_idx:t_idx+1, :]  # F(t, :) shape (2,2,1,Nt)

        # Thermal integrals for precise_convolution
        F_integral_col = thermal_integral[:, t_idx:t_idx+1]  # ∫F(:,t)
        F_integral_row = thermal_integral[t_idx:t_idx+1, :]  # ∫F(t,:)

        # Term 1: ∫ dt' τ₃ g'^R(t,t') A(t') τ₃ g'^K(t',t)
        # = τ₃ [g'^R(t,:) @ (A(:) τ₃ g'^K(:,t))]
        # Midpoint rule: subtract 1/2 weight from both endpoints
        inner_1 = A_tensor * tau3 * gk_col
        first_endpoint_1 = tau3 * (gr_row[:,0] * inner_1[0,:])[0,0]
        last_endpoint_1 = tau3 * (gr_row[:,-1] * inner_1[-1,:])[0,0]
        term1 = tau3 * (gr_row @ inner_1)[0,0] * self.dt - 0.5 * self.dt * first_endpoint_1 - 0.5 * self.dt * last_endpoint_1

        # Term 2: ∫ dt' τ₃ g'^K(t,t') A(t') τ₃ g'^A(t',t)
        # = τ₃ [g'^K(t,:) @ (A(:) τ₃ g'^A(:,t))]
        # Midpoint rule: subtract 1/2 weight from both endpoints
        inner_2 = A_tensor * tau3 * ga_col
        first_endpoint_2 = tau3 * (gk_row[:,0] * inner_2[0,:])[0,0]
        last_endpoint_2 = tau3 * (gk_row[:,-1] * inner_2[-1,:])[0,0]
        term2 = tau3 * (gk_row @ inner_2)[0,0] * self.dt - 0.5 * self.dt * first_endpoint_2 - 0.5 * self.dt * last_endpoint_2

        # Term 3: ∫ dt' 2τ₃ g'^R(t,t') A(t') F(t',t)
        # Multiply gr with A*tau3, then precise_convolution_left with F (regularized)
        term3 = 2.0 * (A_tensor * tau3 * gr_row).precise_convolution_left(F_col, F_integral_col, self.dt, other_index=t_idx)[0,0]

        # Term 4: ∫ dt' 2F(t,t') A(t') τ₃ g'^A(t',t)
        # Multiply ga with A*tau3, then precise_convolution_right with F (regularized)
        term4 = 2.0 * (A_tensor * tau3 * ga_col).precise_convolution_right(F_row, F_integral_row, self.dt, self_index=t_idx)[0,0]

        # Sum all terms and take Nambu trace
        total = term1 + term2 + term3 + term4
        current = total.trace(pauli_index=0) / 2.0

        # Apply prefactor -i(π / 4) [σ_n absorbed into normalization]
        current = -1j * np.pi / 4 * current

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

        Checks: ∫ g^R g^K + ∫ g^K g^A + [τ₃, g^K] - ∫ (g^R @ f) + ∫ (f @ g^A)= 0

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

        #! the midpoint rule could be breaking associativity of the convolution, check analytically
        conv1 = conv1 - 0.5 * self.dt * first_endpoint_1 - 0.5 * self.dt * last_endpoint_1

        # Second convolution: ∫ g^K(t1, t') g^A(t', t2) dt' for all t2
        conv2 = (gk_row @ ga) * self.dt  # shape (2,2,1,Nt) 
        #* goes up to t', however what happens is that for elements with first element bigger than t', the integral is cut-off
        #* this means that the 1/2 midpoint rule is different!!! 
        #* not -1 --> this is also wrong in precise convolutions! --> one has to be careful with ga jump conditions
        #* carefully compute the ga diagonal and which elements come in!!
        # Apply midpoint rule to conv2
        # Integration from t'=0 to t'=t2 for all t2
        # First endpoint: gk[t1, 0] * ga[0, t2]
        # Last endpoint: gk[t1, t2] * ga[t2, t2]
        gk_t1_0 = self.gk[t1_pos, 0]  # shape (2,2)
        ga_0_row = ga[0:1, :]  # shape (2,2,1,Nt)
        first_endpoint_2 = gk_t1_0 * ga_0_row  # gk[t1, 0] * ga[0, t2]

        # Extract diagonal elements of ga as row tensor
        last_endpoint_2 = gk_row * ga.diagonal_time()  # gk[t1, t2] * ga[t2, t2] #* nothing changed since this is set to zero by default right now!

        conv2 = conv2 - 0.5 * self.dt * first_endpoint_2 - 0.5 * self.dt * last_endpoint_2
        
        # Thermal terms from FDT: g^K = g^R @ f - f @ g^A
        # Thermal term 1: gr_row @ (gr @ f) for all t2 at once
        thermal_gr = gr_row.precise_convolution_left(thermal_dist, thermal_integral, self.dt, other_index=t1_pos) * tau3 * 2

        # Thermal term 2: (f_row @ ga) @ ga for all t2 at once
        f_row = thermal_dist[t1_pos:t1_pos+1, :]
        F_row = thermal_integral[t1_pos:t1_pos+1, :]
        thermal_ga = 2 * tau3 * ga.precise_convolution_right(f_row, F_row, self.dt, self_index=t1_pos) 

        # Add thermal corrections to convolutions
        conv1 = conv1 + thermal_gr  # Subtract gr @ (gr @ f) term
        conv2 = conv2 + thermal_ga  # Add (f @ ga) @ ga term

        # Total normalization violation for all t2
        total = commutator + conv1 + conv2  # shape (2,2,1,Nt)

        # Compute errors
        errors = np.sqrt(np.sum(np.abs(total.data)**2, axis=(0, 1)))[0, :]  # shape (Nt,)

        # Extract Pauli components
        totals = np.zeros((4, N_t), dtype=complex)
        commutators = np.zeros((4, N_t), dtype=complex)
        conv1s = np.zeros((4, N_t), dtype=complex)
        conv2s = np.zeros((4, N_t), dtype=complex)

        for pauli_idx in range(4):
            totals[pauli_idx, :] = (total.trace(pauli_idx) / 2)[0, :]
            commutators[pauli_idx, :] = (commutator.trace(pauli_idx) / 2)[0, :]
            conv1s[pauli_idx, :] = (conv1.trace(pauli_idx) / 2)[0, :]
            conv2s[pauli_idx, :] = (conv2.trace(pauli_idx) / 2)[0, :]

        return errors, totals, {
            'commutator': commutators,
            'gr_gk_conv': conv1s,
            'gk_ga_conv': conv2s
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
