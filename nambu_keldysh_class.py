"""
Nambu-Keldysh tensor class for 2x2 Nambu matrices in Keldysh formalism.
Handles matrix operations, Pauli decomposition, and integration over energy/angle grids.
"""

import numpy as np
import string

class NambuKeldyshTensor:
    """Container for 2x2 Nambu matrices with additional dimensions (frequency, angle, etc.)."""

    def __init__(self, data_in: np.array, pauli_channel=None):
        """Initialize Nambu tensor from data array or Pauli channel projection."""
        # Handle case where data_in is already a NambuKeldyshTensor (unwrap it)
        if isinstance(data_in, NambuKeldyshTensor):
            data_in = data_in.data

        # Convert to numpy array if it's not already
        if not isinstance(data_in, np.ndarray):
            data_in = np.asarray(data_in, dtype=complex)

        # if not a nambu tensor make one
        #NOTE: The data_in is not allowed to be of the shape (2,2)
        if pauli_channel is None:
            if (len(data_in.shape) < 2):
                data = np.tensordot(np.ones((2,2),dtype=complex),data_in,axes=0)

            elif (np.shape(data_in)[:2]!= (2,2)):
                data = np.tensordot(np.ones((2,2),dtype=complex),data_in,axes=0)

            else:
                # if a nambu tensor then just take the data
                data = np.array(data_in,dtype=complex)

        # assuming that if you pass in a pauli channel that you want to tensordot the data!
        # this means its possible to get data with (2,2) structure if a pauli channel is specified
        else:
            pauli_matrix = get_pauli_matrix(index = pauli_channel)
            data = np.tensordot(pauli_matrix,data_in,axes=0)

        self.data = data
        self.shape = data.shape

    # ========== Matrix Operations ==========

    def __mul__(self, other):
        """
        Nambu matrix multiplication with element-wise multiplication on other indices.
        Performs 2x2 Nambu matrix product on first two indices, element-wise on remaining.
        Supports: NambuKeldyshTensor, complex scalars, arrays.

        Cases:
        - (2,2,a,b) * (2,2,a,b) -> einsum('ijab,jmab->imab')
        - (2,2,a) * (2,2,a,b) -> einsum('ija,jmab->imab')  [broadcasts over b]
        - (2,2) * (2,2,a,b) -> einsum('ij,jmab->imab')  [broadcasts over a,b]
        - scalar * (2,2,...) -> element-wise multiplication

        Usage: A * B
        """
        # Handle scalar multiplication
        if isinstance(other, (int, float, complex)):
            return NambuKeldyshTensor(other * self.data)

        # Extract data from other
        if isinstance(other, NambuKeldyshTensor):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            return NotImplemented

        # Check if other has Nambu structure (2,2) at the start
        if other_data.ndim < 2 or other_data.shape[:2] != (2, 2):
            # Not a Nambu matrix - do element-wise multiplication
            return NotImplemented

        # Get number of extra dimensions beyond Nambu (2,2)
        self_ndim = self.data.ndim - 2
        other_ndim = other_data.ndim - 2
        max_ndim = max(self_ndim, other_ndim)

        # Build einsum index strings
        # Use letters starting from 'n' for extra dimensions
        extra_letters = string.ascii_lowercase[13:13+max_ndim]  # n, o, p, q, ...

        # Left operand (self) aligns from left, right operand (other) aligns from right
        # This ensures v_{ijk} * A_{jlko} -> C_{ilko} and A_{ijko} * v_{jlo} -> C_{ilko}
        self_indices = 'ij' + extra_letters[:self_ndim]
        other_indices = 'jm' + extra_letters[max_ndim - other_ndim:max_ndim]
        result_indices = 'im' + extra_letters[:max_ndim]

        einsum_str = f'{self_indices},{other_indices}->{result_indices}'
        result_data = np.einsum(einsum_str, self.data, other_data)

        return NambuKeldyshTensor(result_data)

    def __rmul__(self, other):
        """
        Right multiplication (for scalar * NambuKeldyshTensor).

        Handles cases where other doesn't know about NambuKeldyshTensor.
        Falls back to this when other.__mul__(self) returns NotImplemented.

        Usage: scalar * A
        """

        if isinstance(other, (int, float, complex)):
            # Scalar multiplication is commutative
            return NambuKeldyshTensor(other * self.data)
            
        else:
            return NotImplemented

    def __matmul__(self, other):
        """
        Matrix product with contraction over Nambu index and shared dimension.

        Performs pure matrix multiplication: contracts over Nambu j index and
        the shared inner dimension (last axis of left, first extra axis of right).
        Includes all points (edge and interior) with uniform weight 1.0.

        Supports various shapes:
        - (2,2,a,b) @ (2,2,b,c) -> (2,2,a,c)  [contracts over all b]
        - (2,2,a) @ (2,2,a,b) -> (2,2,b)      [contracts over all a]
        - (2,2,a,b) @ (2,2,b) -> (2,2,a)      [contracts over all b]
        - (2,2,a) @ (2,2,a) -> (2,2)          [contracts over all a]

        Usage: A @ B
        """
        # Extract data from other
        if isinstance(other, NambuKeldyshTensor):
            other_data = other.data
        else:
            return NotImplemented

        # Check first two dimensions are (2,2) for both
        if self.data.shape[:2] != (2, 2) or other_data.shape[:2] != (2, 2):
            raise ValueError(f"First two dimensions must be (2,2), got {self.data.shape} @ {other_data.shape}")

        # Get number of extra dimensions beyond Nambu (2,2)
        self_ndim = self.data.ndim - 2  # e.g., 2 for (2,2,a,b)
        other_ndim = other_data.ndim - 2  # e.g., 2 for (2,2,b,c)

        # Check at least one extra dimension exists
        if self_ndim == 0 or other_ndim == 0:
            raise ValueError(f"matmul requires at least one extra dimension beyond (2,2), got {self.data.shape} @ {other_data.shape}")

        # Check contraction dimension matches (last of left = first extra of right)
        if self.data.shape[-1] != other_data.shape[2]:
            raise ValueError(
                f"Contraction dimension mismatch: {self.data.shape[-1]} != {other_data.shape[2]}"
            )

        # Handle too-small contraction
        contraction_size = self.data.shape[-1]
        if contraction_size == 0:
            # Empty - return zeros
            if self_ndim > 1 and other_ndim > 1:
                result_shape = (2, 2) + self.data.shape[2:-1] + other_data.shape[3:]
            elif self_ndim > 1:
                result_shape = (2, 2) + self.data.shape[2:-1]
            else:
                result_shape = (2, 2) + other_data.shape[3:]
            result_data = np.zeros(result_shape, dtype=complex)
            return NambuKeldyshTensor(result_data)

        # Use full data for both operands (all points with weight 1.0)
        left_data = self.data
        right_data = other_data

        # Build einsum string dynamically
        letters = string.ascii_lowercase

        # Left operand indices: 'ij' + extra dimensions
        left_extra = ''.join(letters[:self_ndim])

        # Right operand indices: 'jk' + (shared index from left) + remaining
        if other_ndim == 1:
            right_extra = left_extra[-1]
        else:
            right_extra = left_extra[-1] + ''.join(letters[self_ndim:self_ndim+other_ndim-1])

        # Result indices: all non-contracted dimensions
        if self_ndim > 1 and other_ndim > 1:
            result_extra = left_extra[:-1] + right_extra[1:]
        elif self_ndim > 1:
            result_extra = left_extra[:-1]
        elif other_ndim > 1:
            result_extra = right_extra[1:]
        else:
            result_extra = ''

        # Construct and execute einsum (with trapezoidal weights included)
        einsum_str = f'ij{left_extra},jk{right_extra}->ik{result_extra}'
        result_data = np.einsum(einsum_str, left_data, right_data)

        return NambuKeldyshTensor(result_data)

    def precise_convolution_left(self, other, other_integral, dt, other_index=-1):
        #TODO: check this is implemented appropriately, are we subtracting the last term in the sum?
        """ 
        Compute regularized left convolution: self @ other (regularized).

        The regularized matrix (other) is on the right side.
        Regularization suppresses Gibbs oscillations from other.

        the same boundary treatment, so the cancellation in regularization works correctly.

        Formula:
            result = dt * (self @ other)
                   - dt * (self * (ones @ other))
                   + (self * other_integral)

        When self is a row (shape 2,2,1,N_t), other_integral_row = other_integral[other_index:other_index+1,:]
        is used for the analytic term, while full other is used for convolution.

        Args:
            other: NambuKeldyshTensor - function to convolve with (regularized)
            other_integral: NambuKeldyshTensor - integral of other
            dt: float - time step for discretization
            other_index: int - row index to extract from other_integral when self is a row (default 0)

        Returns:
            NambuKeldyshTensor - regularized convolution result

        Example:
            # gr @ f with regularization on f
            result = gr_row.precise_convolution_left(f_thermal, f_integral, dt, other_index=-1)
        """
        # ========== Input shape validation for midpoint rule ==========
        # Check that self is a row: (2,2,1,N_t)
        if self.data.ndim != 4 or self.data.shape[:3] != (2, 2, 1):
            raise ValueError(
                f"precise_convolution_left requires self to have shape (2,2,1,N_t) for proper midpoint rule.\n"
                f"Got shape {self.data.shape}. The third dimension must be 1 (row tensor)."
            )

        # Check that other is a full matrix: (2,2,N_t,N_t')
        if other.data.ndim != 4 or other.data.shape[:2] != (2, 2):
            raise ValueError(
                f"precise_convolution_left requires other to have shape (2,2,N_t,N_t') for proper midpoint rule.\n"
                f"Got shape {other.data.shape}."
            )

        # Check contraction dimension compatibility: self.shape[-1] == other.shape[2]
        N_t_self = self.data.shape[3]
        N_t_other = other.data.shape[2]
        if N_t_self != N_t_other:
            raise ValueError(
                f"Contraction dimension mismatch: self has N_t={N_t_self} but other has N_t={N_t_other}.\n"
                f"Expected shapes: self=(2,2,1,{N_t_other}), other=(2,2,{N_t_other},N_t').\n"
                f"Got: self={self.data.shape}, other={other.data.shape}"
            )

        # Create ones tensor as a row vector (identity in Nambu space)
        # Shape: (2, 2, 1, N_t) where N_t is the last dimension of other
        # pauli_channel=0 automatically creates the (2,2) identity structure
        ones_data = np.ones((1, other.data.shape[-2]), dtype=complex)
        ones_tensor = NambuKeldyshTensor(ones_data, pauli_channel=0)

        # Create filter function to suppress early-time artifacts
        # Zero out first 1/5 of time points
        N_t = self.data.shape[-1]
        filter_data = np.append(np.zeros(N_t//5), np.ones(N_t - N_t//5))
        filter_function = NambuKeldyshTensor(filter_data, pauli_channel=0)

        # Check if self is a row (shape: 2, 2, 1, N_t) - already validated above
        is_self_row = (self.data.shape[2] == 1)

        # Standard convolution (always uses full other)
        #* midpoint rule: subtract 1/2 weight from BOTH endpoints
        first_endpoint_std = self[:,0:1] * other[0:1,:]
        last_endpoint_std = self[:,-1:] * other[-1:,:]
        result_std = (self @ other) * dt - 0.5 * dt * first_endpoint_std - 0.5 * dt * last_endpoint_std

        # Factored term (non-analytic contribution) - APPLY FILTER to ones_tensor @ other
        # Filter suppresses early-time artifacts in the factorized convolution
        ones_at_other_filtered = (ones_tensor @ other) * filter_function
        #* midpoint rule: subtract 1/2 weight from BOTH endpoints
        first_endpoint_fact = self * (ones_tensor[:,0:1] * other[0,:])
        last_endpoint_fact = self * (ones_tensor[:,-1:] * other[-1,:])
        result_fact = (self * ones_at_other_filtered) * dt - 0.5 * dt * first_endpoint_fact - 0.5 * dt * last_endpoint_fact * 0 #* zero from filter function

        # For analytic term: use row of other_integral if self is a row
        if is_self_row:
            # Extract corresponding row from other_integral
            # Handle negative indices properly to avoid empty slices
            N_t = other_integral.data.shape[2]
            if other_index < 0:
                # Convert negative index to positiveo
                positive_index = N_t + other_index
            else:
                positive_index = other_index
            other_integral_for_reg = other_integral[positive_index:positive_index+1, :]
        else:
            other_integral_for_reg = other_integral

        # Analytic term (using integral)
        result_anal = self * other_integral_for_reg

        # Combine: standard - factored + analytic
        return (result_std + (- result_fact + result_anal))


    def precise_convolution_right(self, other, other_integral, dt, self_index = -1):
        """
        Compute regularized right convolution: other @ self (regularized).

        The regularized matrix (other) is on the left side.
        Regularization suppresses Gibbs oscillations from other.

        NOTE: Trapezoidal boundary weighting (0.5 at endpoints) is automatically
        handled by the @ operator. Both (other @ self) and (other @ ones) receive
        the same boundary treatment, so the cancellation in regularization works correctly.

        Formula:
            result = dt * (other @ self)
                   - dt * ((other @ ones) * self_row)
                   + (other_integral * self_row)

        When other is a row (shape 2,2,1,N_t), self_row = self[0:1,:] is used
        for factored and analytic terms, while full self is used for convolution.

        Args:
            other: NambuKeldyshTensor - function to convolve with (regularized)
            other_integral: NambuKeldyshTensor - integral of other
            dt: float - time step for discretization

        Returns:
            NambuKeldyshTensor - regularized convolution result

        Example:
            # f @ ga with regularization on f
            result = ga.precise_convolution_right(f_thermal, f_integral, dt)
        """
        # ========== Input shape validation for midpoint rule ==========
        # Check that other is a row: (2,2,1,N_t)
        if other.data.ndim != 4 or other.data.shape[:3] != (2, 2, 1):
            raise ValueError(
                f"precise_convolution_right requires other to have shape (2,2,1,N_t) for proper midpoint rule.\n"
                f"Got shape {other.data.shape}. The third dimension must be 1 (row tensor)."
            )

        # Check that self is a full matrix: (2,2,N_t,N_t')
        if self.data.ndim != 4 or self.data.shape[:2] != (2, 2):
            raise ValueError(
                f"precise_convolution_right requires self to have shape (2,2,N_t,N_t') for proper midpoint rule.\n"
                f"Got shape {self.data.shape}."
            )

        # Check contraction dimension compatibility: other.shape[-1] == self.shape[2]
        N_t_other = other.data.shape[3]
        N_t_self = self.data.shape[2]
        if N_t_other != N_t_self:
            raise ValueError(
                f"Contraction dimension mismatch: other has N_t={N_t_other} but self has N_t={N_t_self}.\n"
                f"Expected shapes: other=(2,2,1,{N_t_self}), self=(2,2,{N_t_self},N_t').\n"
                f"Got: other={other.data.shape}, self={self.data.shape}"
            )

        # Create ones tensor as a 2D matrix (identity in Nambu space)
        # Shape: (2, 2, N_t, N_t') where ones_data[t, t'] = 1 if t < t', else 0
        # This enforces causality: integration only up to each t'
        # pauli_channel=0 automatically creates the (2,2) identity structure
        N_t = other.data.shape[-1]
        N_tprime = self.data.shape[-1]

        filter_data = np.append(np.zeros(N_t//5), np.ones(N_t - N_t//5))
        filter_function = NambuKeldyshTensor(filter_data, pauli_channel=0)

        # Create mask matrix: ones_data[i, j] = 1 if i <= j (strict inequality for t <= t')
        row_indices = np.arange(N_t)[:, np.newaxis]  # Shape (N_t, 1)
        col_indices = np.arange(N_tprime)[np.newaxis, :]  # Shape (1, N_t')
        ones_data = (row_indices <= col_indices).astype(complex)
        ones_tensor = NambuKeldyshTensor(ones_data, pauli_channel=0)
        # Check if other is a row (shape: 2, 2, 1, N_t) - already validated above
        is_other_row = (other.data.shape[2] == 1)
        # Standard convolution (always uses full self)
        first_endpoint_std = other[:,0:1] * self[0:1,:]
        last_endpoint_std = other[:,:] * self.diagonal_time() #*last element is self(t't') actually!
        result_std = (other @ self) * dt - 0.5 * dt * first_endpoint_std - 0.5 * dt * last_endpoint_std
        # For factored and analytic terms: use row of self if other is a row
        if is_other_row:
            # Extract corresponding column from self
            # Handle negative indices properly to avoid empty slices
            N_t = self.data.shape[2]
            if self_index < 0:
                # Convert negative index to positive
                positive_index = N_t + self_index
            else:
                positive_index = self_index
            self_for_reg = self.diagonal_time()
        else:
            self_for_reg = self
        # Factored term (non-analytic contribution)
        #* midpoint rule: subtract 1/2 weight from BOTH endpoints
        first_endpoint_fact = (other[:,0:1] * ones_tensor[0:1,:]) * self_for_reg
        last_endpoint_fact = (other[:,:]) * self_for_reg
        result_fact = ((other @ ones_tensor) * self_for_reg) * filter_function * dt - 0.5 * dt * first_endpoint_fact - 0.5 * dt * last_endpoint_fact *0 

        # Analytic term (using integral)
        result_anal = (- other_integral) * self_for_reg

        # Combine: standard - factored + analytic
        return (result_std + ( - result_fact + result_anal))

    def _check_binary_shape_compatibility(self, other):
        """
        Check if two NambuKeldyshTensor objects have compatible shapes for binary operations.

        Args:
            other: Another NambuKeldyshTensor object5

        Raises:
            ValueError: If shapes are incompatible for element-wise operations
        """
        if not isinstance(other, NambuKeldyshTensor):
            return  # Allow broadcasting with scalars/arrays

        if self.data.shape != other.data.shape:
            raise ValueError(
                f"Shape mismatch for binary operation: "
                f"left operand has shape {self.data.shape}, "
                f"right operand has shape {other.data.shape}. "
                f"Binary operations require identical shapes."
            )

    def _binary_ewise(self, other, op):
        """Helper for element-wise binary operations."""
        if isinstance(other, NambuKeldyshTensor):
            self._check_binary_shape_compatibility(other)
            return NambuKeldyshTensor(op(self.data, other.data))
        else:
            return NambuKeldyshTensor(op(self.data, other))

    def __add__(self, other):
        return self._binary_ewise(other, np.add)

    def __radd__(self, other):
        return self._binary_ewise(other, np.add)

    def __sub__(self, other):
        return self._binary_ewise(other, np.subtract)

    def __truediv__(self, other):
        return self._binary_ewise(other, np.true_divide)

    def __rtruediv__(self, other):
        return self._binary_ewise(other, lambda a, b: np.true_divide(b, a))

    def __neg__(self):
        return NambuKeldyshTensor(np.negative(self.data))

    def __getitem__(self, index):
        """
        Index into Nambu tensor along time dimensions.

        Always preserves the first two Nambu dimensions by prepending [:, :].

        Examples:
            obj[i] - returns self.data[:, :, i]
            obj[i, j] - returns self.data[:, :, i, j]
            obj[:, i] - returns self.data[:, :, :, i]
            obj[i, :] - returns self.data[:, :, i, :]
        """
        # Ensure index is a tuple
        if not isinstance(index, tuple):
            index = (index,)

        # Always prepend [:, :] to preserve Nambu dimensions
        full_index = (slice(None), slice(None)) + index
        result = self.data[full_index]

        # Wrap in NambuKeldyshTensor if result has (2,2) structure
        if result.ndim >= 2 and result.shape[:2] == (2, 2):
            return NambuKeldyshTensor(result)
        else:
            # Result doesn't have Nambu structure, return raw array
            return result


    def __str__(self):
        """String representation showing Pauli decomposition."""
        id_trace = self.trace(pauli_index=0)
        x_trace = self.trace(pauli_index=1)
        y_trace = self.trace(pauli_index=2)
        z_trace = self.trace(pauli_index=3)

        return f"Id Trace: {id_trace} \n X Trace: {x_trace} \n Y Trace: {y_trace} \n Z Trace: {z_trace} \n"

    # ========== Matrix Structure Operations ==========

    def conj(self):
        """Complex conjugation."""
        return NambuKeldyshTensor(np.conjugate(self.data))

    def transpose(self):
        """Transpose Nambu matrix indices (0,1) only."""
        axes = (1, 0) + tuple(range(2, self.data.ndim))
        return NambuKeldyshTensor(np.transpose(self.data, axes=axes))

    def dagger(self):
        """Conjugate transpose."""
        return self.conj().transpose()

    def complete_transpose(self):
        """
        Transpose both Nambu (0,1) and time (2,3) indices.
        For 4D tensors: (0,1,2,3) -> (1,0,3,2)
        Converts (2,2,a,b) to (2,2,b,a) with Nambu transpose.
        """
        if self.data.ndim == 4:
            return NambuKeldyshTensor(np.transpose(self.data, axes=(1, 0, 3, 2)))
        else:
            # For non-4D tensors, just do regular Nambu transpose
            return self.transpose()

    def involution(self):
        """
        Compute Nambu-Keldysh involution: τ3 * conj(complete_transpose(A)) * τ3.

        Uses complete transpose for proper Keldysh structure.
        Efficiently uses only the (2,2) Pauli matrix with broadcasting.
        """
        # Get tau3 Pauli matrix (2,2) - no need to create full (2,2,n,m) array
        tau3 = get_pauli_matrix(3)

        # Middle term: conj(A^{T,T'})
        middle = self.conj().complete_transpose()

        # tau3[i,j] * middle[j,k,...] -> temp[i,k,...]
        temp = np.einsum('ij,jk...->ik...', tau3, middle.data)

        # temp[i,j,...] * tau3[j,k] -> result[i,k,...]
        result = np.einsum('ij...,jk->ik...', temp, tau3)

        return NambuKeldyshTensor(result)

    def determinant(self):
        """
        Compute determinant of 2x2 Nambu matrix.

        Returns array with shape (1,1,...) for compatibility with operations.
        """
        det = self.data[0,0,...]*self.data[1,1,...] - self.data[0,1,...]*self.data[1,0,...]
        return det[None, None, ...]

    def diagonal_time(self):
        """
        Extract time diagonal from a two-time tensor.

        For a tensor with shape (2, 2, Nt, Nt), extracts the diagonal elements
        where both time indices are equal, returning a tensor with shape (2, 2, Nt).

        This is useful for operators like R(t'', t') where we need R(t', t')
        (diagonal elements as a function of time).

        Returns:
            NambuKeldyshTensor: Tensor with shape (2, 2, Nt) containing diagonal elements

        Raises:
            ValueError: If tensor doesn't have exactly 4 dimensions (not a two-time object)

        Example:
            For R with shape (2, 2, 100, 100), R.diagonal_time() returns shape (2, 2, 100)
            where result[:, :, i] = R[:, :, i, i]
        """
        if self.data.ndim != 4:
            raise ValueError(
                f"diagonal_time() requires a two-time tensor with shape (2, 2, Nt, Nt). "
                f"Got shape {self.data.shape} with {self.data.ndim} dimensions."
            )

        # Extract diagonal: data[:, :, i, i] for all i
        Nt = self.data.shape[2]
        diagonal_data = np.zeros((2, 2, Nt), dtype=self.data.dtype)

        for i in range(Nt):
            diagonal_data[:, :, i] = self.data[:, :, i, i]

        return NambuKeldyshTensor(diagonal_data)

    # ========== Pauli Basis Operations ==========

    def trace(self, pauli_index=None):
        """Trace with respect to Pauli basis (0=I, 1=X, 2=Y, 3=Z)."""
        pauli_matrix = get_pauli_matrix(index=pauli_index)
        return (pauli_matrix[0,0]*self.data[0,0,...] +
                pauli_matrix[0,1]*self.data[1,0,...] +
                pauli_matrix[1,0]*self.data[0,1,...] +
                pauli_matrix[1,1]*self.data[1,1,...])

    def matrix_to_vector(self):
        """
        Convert Nambu matrix to vector of Pauli components.

        Decomposes ĝ = g₀·𝟙 + g₁·τ₁ + g₂·τ₂ + g₃·τ₃ and extracts [g₀, g₁, g₂, g₃].
        Uses Tr[τᵢ·ĝ] = 2gᵢ, so gᵢ = Tr[τᵢ·ĝ]/2.

        Returns:
            np.ndarray: Array of shape (4, ...) containing [g₀, g₁, g₂, g₃]
                       where ... represents any extra dimensions beyond Nambu (2,2)

        Example:
            For shape (2, 2, Nt, Nt'), returns (4, Nt, Nt')
        """
        g0 = self.trace(pauli_index=0) / 2.0
        g1 = self.trace(pauli_index=1) / 2.0
        g2 = self.trace(pauli_index=2) / 2.0
        g3 = self.trace(pauli_index=3) / 2.0

        return np.array([g0, g1, g2, g3])

    @staticmethod
    def vector_to_matrix(components):
        """
        Construct Nambu matrix from vector of Pauli components.

        Builds ĝ = g₀·𝟙 + g₁·τ₁ + g₂·τ₂ + g₃·τ₃ from [g₀, g₁, g₂, g₃].

        Args:
            components: Array of shape (4, ...) or list of 4 arrays [g₀, g₁, g₂, g₃]
                       Each component can have arbitrary shape (Nt, Nt'), etc.
                       For 1D input (4 scalars), output will be (2, 2, 1)

        Returns:
            NambuKeldyshTensor: Tensor of shape (2, 2, ...) or (2, 2, 1) for scalar input

        Example:
            components = np.array([g0, g1, g2, g3])  # shape (4, Nt, Nt')
            g_matrix = NambuKeldyshTensor.vector_to_matrix(components)  # shape (2, 2, Nt, Nt')

            components = np.array([1, 0, 0, 0])  # shape (4,)
            g_matrix = NambuKeldyshTensor.vector_to_matrix(components)  # shape (2, 2, 1)
        """
        # Extract individual components
        if isinstance(components, np.ndarray):
            g0, g1, g2, g3 = components[0], components[1], components[2], components[3]
        else:
            g0, g1, g2, g3 = components

        # Check if components are scalars (0D arrays or Python numbers)
        # If so, add an extra dimension to ensure output is (2, 2, 1)
        g0 = np.asarray(g0)
        if g0.ndim == 0:
            g0 = g0[np.newaxis]  # Add dimension: scalar -> shape (1,)
            g1 = np.asarray(g1)[np.newaxis]
            g2 = np.asarray(g2)[np.newaxis]
            g3 = np.asarray(g3)[np.newaxis]

        # Construct Nambu matrix as sum of Pauli components
        result = (NambuKeldyshTensor(g0, pauli_channel=0) +
                  NambuKeldyshTensor(g1, pauli_channel=1) +
                  NambuKeldyshTensor(g2, pauli_channel=2) +
                  NambuKeldyshTensor(g3, pauli_channel=3))

        return result

    # ========== Shift Operations ==========

    def shift(self, shift=1, axis=0):
        """
        Shift data along specified axis with zero-padding at the boundary.

        Args:
            shift: Number of positions to shift (default 1)
                   Positive values shift forward, negative shift backward
            axis: Which axis to shift along (0 or 1), referring to axes after Nambu (2,2)
                  axis=0 -> shift along 3rd dimension (array index 2)
                  axis=1 -> shift along 4th dimension (array index 3)

        Returns:
            NambuKeldyshTensor: Shifted tensor with same shape as input

        Example:
            For shape (2, 2, Nt, Nt'):
            - shift(shift=1, axis=0) shifts data forward by 1 along time dimension
            - shift(shift=-1, axis=1) shifts data backward by 1 along time' dimension
        """
        actual_axis = axis + 2
        shifted_data = np.roll(self.data, shift=shift, axis=actual_axis)

        if shift > 0:
            slices = [slice(None)] * self.data.ndim
            slices[actual_axis] = slice(0, shift)
            shifted_data[tuple(slices)] = 0.0
        elif shift < 0:
            slices = [slice(None)] * self.data.ndim
            slices[actual_axis] = slice(shift, None)
            shifted_data[tuple(slices)] = 0.0

        return NambuKeldyshTensor(shifted_data)

    # ========== Gradient Operations ==========

    def gradient(self, axis=0):
        """
        Compute discrete gradient along specified axis (1st-order backward difference).

        Computes finite difference: data[..., i, ...] - data[..., i-1, ...]
        The first entry along the differentiated axis is set to zero (no previous value).

        Args:
            axis: Which axis to differentiate along (0 or 1), referring to axes after Nambu (2,2)
                  axis=0 -> differentiate along 3rd dimension (array index 2)
                  axis=1 -> differentiate along 4th dimension (array index 3)

        Returns:
            NambuKeldyshTensor: Gradient with same shape as input

        Example:
            For shape (2, 2, Nt, Nt'):
            - gradient(axis=0) computes ∂g/∂t
            - gradient(axis=1) computes ∂g/∂t'
        """
        # Map user axis (0 or 1) to actual array axis (2 or 3)
        actual_axis = axis + 2

        # Compute difference: data[i] - data[i-1]
        # Roll the array by 1 along the axis to get shifted version
        shifted = np.roll(self.data, shift=1, axis=actual_axis)

        # Compute gradient
        gradient_data = self.data - shifted

        # The first entry along this axis is invalid (wraps around from roll)
        # Set it to zero since there's no previous value
        slices = [slice(None)] * self.data.ndim
        slices[actual_axis] = 0
        gradient_data[tuple(slices)] = 0

        return NambuKeldyshTensor(gradient_data)

    def gradient_second_order(self, axis=0):
        """
        Compute discrete gradient along specified axis using 2nd-order backward difference.

        Uses formula: dg/dx ≈ (3*g[i] - 4*g[i-1] + g[i-2]) / 2  (with dx=1)
        First two entries use lower-order schemes (no sufficient history).

        Args:
            axis: Which axis to differentiate along (0 or 1), referring to axes after Nambu (2,2)
                  axis=0 -> differentiate along 3rd dimension (array index 2)
                  axis=1 -> differentiate along 4th dimension (array index 3)

        Returns:
            NambuKeldyshTensor: Gradient with same shape as input

        Example:
            For shape (2, 2, Nt, Nt'):
            - gradient_second_order(axis=0) computes ∂g/∂t
            - gradient_second_order(axis=1) computes ∂g/∂t'
        """
        # Map user axis (0 or 1) to actual array axis (2 or 3)
        actual_axis = axis + 2

        # Get shifted versions for backward difference
        shifted_1 = np.roll(self.data, shift=1, axis=actual_axis)  # g[i-1]
        shifted_2 = np.roll(self.data, shift=2, axis=actual_axis)  # g[i-2]

        # 2nd-order backward difference: (3*g[i] - 4*g[i-1] + g[i-2]) / 2
        gradient_data = (3.0 * self.data - 4.0 * shifted_1 + shifted_2) / 2.0

        # Handle first two entries (insufficient past history)
        slices_0 = [slice(None)] * self.data.ndim
        slices_1 = [slice(None)] * self.data.ndim

        slices_0[actual_axis] = 0
        slices_1[actual_axis] = 1

        # First entry: no past, set to zero
        gradient_data[tuple(slices_0)] = 0

        # Second entry: use 1st-order backward (g[1] - g[0])
        gradient_data[tuple(slices_1)] = self.data[tuple(slices_1)] - shifted_1[tuple(slices_1)]

        return NambuKeldyshTensor(gradient_data)

    # ========== Sliding Window Update ==========

    def append(self, vector):
        """
        Append vector to Nambu tensor from the left (prepend).

        Takes a vector of Pauli components [g₀, g₁, g₂, g₃] and prepends it
        as a new entry at the beginning of the extra dimensions.

        Args:
            vector: Array of shape (4,) or (4, ...) containing [g₀, g₁, g₂, g₃]

        Updates:
            self.data: Modified in-place from (2, 2, N, ...) to (2, 2, N+1, ...)
                      New entry added at index 0 of dimension 2

        Example:
            obj has shape (2, 2, N)
            vector has shape (4,)
            After obj.append(vector), obj has shape (2, 2, N+1)
            New data is at obj.data[:, :, 0]
        """
        # Convert vector to Nambu matrix
        # This will have shape (2, 2, 1) for 1D vector or (2, 2, 1, ...) for higher-dim vector
        new_entry = NambuKeldyshTensor.vector_to_matrix(vector)

        # Prepend along axis 2 (first dimension after Nambu (2,2))
        self.data = np.concatenate([new_entry.data, self.data], axis=2)

        # Update shape
        self.shape = self.data.shape

    def append_right(self, vector):
        """
        Append vector to Nambu tensor from the right.

        Takes a vector of Pauli components [g₀, g₁, g₂, g₃] and appends it
        as a new entry at the end of the extra dimensions.

        Args:
            vector: Array of shape (4,) or (4, ...) containing [g₀, g₁, g₂, g₃]

        Updates:
            self.data: Modified in-place from (2, 2, N, ...) to (2, 2, N+1, ...)
                      New entry added at index N (last position) of dimension 2

        Example:
            obj has shape (2, 2, N)
            vector has shape (4,)
            After obj.append_right(vector), obj has shape (2, 2, N+1)
            New data is at obj.data[:, :, N] (now the last entry)
        """
        # Convert vector to Nambu matrix
        # This will have shape (2, 2, 1) for 1D vector or (2, 2, 1, ...) for higher-dim vector
        new_entry = NambuKeldyshTensor.vector_to_matrix(vector)

        # Append along axis 2 (first dimension after Nambu (2,2))
        self.data = np.concatenate([self.data, new_entry.data], axis=2)

        # Update shape
        self.shape = self.data.shape

    def update_entries(self, new_upper_row, new_lower_column, new_diagonal):
        """
        Update matrix using sliding window: add new timestep, remove oldest.

        Efficiently shifts data using np.roll and overwrites last entries.
        Maintains (2, 2, Nt, Nt) shape by removing oldest and adding newest time.

        Args:
            new_upper_row: New row entries g[Nt-1, j] for j < Nt
                          Shape: (2, 2, Nt) or NambuKeldyshTensor
            new_lower_column: New column entries g[i, Nt-1] for i < Nt
                             Shape: (2, 2, Nt) or NambuKeldyshTensor
            new_diagonal: New diagonal element g[Nt-1, Nt-1]
                         Shape: (2, 2) or NambuKeldyshTensor

        Updates:
            self.data: Modified in-place, oldest time removed, newest added
        """
        Nt = self.data.shape[2]

        # Extract raw arrays from inputs
        if isinstance(new_upper_row, NambuKeldyshTensor):
            upper_row_data = new_upper_row.data[:, :, 0, :Nt] if new_upper_row.data.ndim == 4 else new_upper_row.data
        else:
            upper_row_data = new_upper_row

        if isinstance(new_lower_column, NambuKeldyshTensor):
            lower_col_data = new_lower_column.data[:, :, :Nt, 0] if new_lower_column.data.ndim == 4 else new_lower_column.data
        else:
            lower_col_data = new_lower_column

        if isinstance(new_diagonal, NambuKeldyshTensor):
            diag_data = new_diagonal.data[:, :, 0, 0] if new_diagonal.data.ndim == 4 else new_diagonal.data
        else:
            diag_data = new_diagonal

        # Roll data to remove oldest time (index 0) and make space at end
        # Roll by -1 along both time axes to shift: [0,1,2,...,Nt-1] -> [1,2,...,Nt-1,?]
        self.data = np.roll(self.data, shift=-1, axis=2)  # Shift along first time axis
        self.data = np.roll(self.data, shift=-1, axis=3)  # Shift along second time axis

        # Now overwrite last row (index Nt-1) with new data
        self.data[:, :, Nt-1, :Nt-1] = upper_row_data[:, :, 1:]

        # Overwrite last column (index Nt-1) with new data
        self.data[:, :, :Nt-1, Nt-1] = lower_col_data[:, :, 1:]

        # Overwrite diagonal (Nt-1, Nt-1) with new diagonal
        self.data[:, :, Nt-1, Nt-1] = diag_data

        # Update shape (should be unchanged, but good practice)
        self.shape = self.data.shape

    def update_diagonal_only(self, new_diagonal, time_index=-1):
        """
        Update only the diagonal element at specified time index without shifting.

        Used when diagonal is computed separately and matrix has already been shifted.
        Critical for Keldysh evolution where g^K(t,t) is computed after g^K(t,t') row
        is inserted and matrix is shifted.

        Args:
            new_diagonal: New diagonal element g[t, t] - shape (2, 2) or NambuKeldyshTensor
            time_index: Time index to update (default -1 for last element)
        """
        # Extract data from input
        if isinstance(new_diagonal, NambuKeldyshTensor):
            diag_data = new_diagonal.data[:, :, 0, 0] if new_diagonal.data.ndim == 4 else new_diagonal.data
        else:
            diag_data = new_diagonal

        # Update only the specified diagonal position
        self.data[:, :, time_index, time_index] = diag_data


def get_pauli_matrix(index = None) -> np.array:
    """A function returning the Pauli matrices or the whole list (if index is none)

    Args:
        index (int, optional): The index of the Pauli matrix from 0 to 3 or - or +. Defaults to None.

    Raises:
        ValueError: If the index is not from 0,1,2,3,+,-,x,y,z,I

    Returns:
        np.array: The Pauli matrix or a identity if index is None
    """
    pauli = [np.eye(2,dtype=complex), np.array([[0.j,1.],[1.,0.j]]), np.array([[0.j,-1.j],[1.j,0.j]]), np.array([[1.0,0.j],[0.j,-1.]]) ]
    if index is None:
        # If not index is given return the whole list
        return pauli[0]
    elif type(index) is str:
        # If index is specified allow for different input types as strings
        if index == 'x':
            return pauli[1]
        elif index == 'y':
            return pauli[2]
        elif index == 'z':
            return pauli[3]
        elif index == 'I':
            return pauli[0]
        elif index == '-':
            return 0.5*(pauli[1] -1.j*pauli[2])
        elif index == '+':
            return 0.5*(pauli[1] + 1.j*pauli[2])
        else:
            raise ValueError("Invalid Pauli matrix index")
    else:
        # if index is an integer return the corresponding Pauli matrix
        return pauli[index]
