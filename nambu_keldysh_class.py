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
        # if the input data has a lot of dimensions then we assume its already a nambu tensor

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
        self.data_shape = data.shape

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

        self_indices = 'ij' + extra_letters[:max_ndim]
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
        Convolution for time-domain Green's functions.

        Only defined for (2,2,Nt,Nt') @ (2,2,Nt',Nt'') -> (2,2,Nt,Nt'')
        Contracts over Nambu index j and shared time index.

        Usage: A @ B
        """
        # Extract data from other
        if isinstance(other, NambuKeldyshTensor):
            other_data = other.data
        else:
            return NotImplemented

        # Check both are (2,2,a,b) shaped
        if self.data.ndim != 4 or other_data.ndim != 4:
            raise ValueError(f"matmul only defined for (2,2,a,b) shapes, got {self.data.shape} @ {other_data.shape}")

        if self.data.shape[:2] != (2, 2) or other_data.shape[:2] != (2, 2):
            raise ValueError(f"First two dimensions must be (2,2)")

        # Check contraction dimension matches
        if self.data.shape[-1] != other_data.shape[2]:
            raise ValueError(
                f"Contraction dimension mismatch: {self.data.shape[-1]} != {other_data.shape[2]}"
            )

        # Perform convolution: (2,2,a,b) @ (2,2,b,c) -> (2,2,a,c)
        result_data = np.einsum('ijab,jkbc->ikac', self.data, other_data)

        return NambuKeldyshTensor(result_data)

    def _binary_ewise(self, other, op):
        """Helper for element-wise binary operations."""
        if isinstance(other, NambuKeldyshTensor):
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
        """Index into Nambu tensor along non-Nambu dimensions."""
        return NambuKeldyshTensor(self.data[:, :, index])

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

    # ========== Pauli Basis Operations ==========

    def trace(self, pauli_index=None):
        """Trace with respect to Pauli basis (0=I, 1=X, 2=Y, 3=Z)."""
        pauli_matrix = get_pauli_matrix(index=pauli_index)
        return (pauli_matrix[0,0]*self.data[0,0,...] +
                pauli_matrix[0,1]*self.data[1,0,...] +
                pauli_matrix[1,0]*self.data[0,1,...] +
                pauli_matrix[1,1]*self.data[1,1,...])


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
