### Jonathan Curtis 
### 08/14/2025
### Class for handling specificities of Nambu space tensors 

import numpy as np

### Class for handling and manipulating Nambu space matrices as tensors on frequency and angular space 
### Optionally we can also append the correct Pauli matrix when constructing the tensor 
class NambuTensor:
	def __init__(self, data,Pauli_channel=None):
		data = np.asarray(data)
		if len(data.shape) == 2: 
			### We are constructing from a tensor which is a scalar function over the w,theta and we want to promote this to a Nambu tensor 
			if Pauli_channel is None:
				data = np.tensordot(np.ones((2,2),dtype=complex),data,axes=0) 
			else:
				data = np.tensordot(Pauli_channel,data,axes=0) 
        	
		assert data.shape[:2] == (2,2), "First two dims must be 2x2"
        
		self.data = data
		self.shape = self.data.shape

	### Overload @ matrix multiplication operator 
	def __matmul__(self, other):
		"""Custom @ operator."""
		if isinstance(other, NambuTensor):
			result = np.einsum('ijnm,jknm->iknm', self.data, other.data)
			return NambuTensor(result)
		else:
			return NotImplemented
			
	### Overload * to give usual pointwise multiplication 
	def __mul__(self,other):
		"""Custom * operator."""
		if isinstance(other,NambuTensor):
			result = self.data * other.data 
			return NambuTensor(result)
		else:
			return NotImplemented
			
	def _binary_ewise(self, other, op):
		if isinstance(other, NambuTensor):
			return NambuTensor(op(self.data, other.data))
		else:
    			return NambuTensor(op(self.data, other))

	def __add__(self, other): return self._binary_ewise(other, np.add)
	def __radd__(self, other): return self._binary_ewise(other, np.add)
	def __sub__(self, other): return self._binary_ewise(other, np.subtract)
	def __truediv__(self, other): return self._binary_ewise(other, np.true_divide)
	def __rtruediv__(self, other): return self._binary_ewise(other, lambda a,b: np.true_divide(b,a))
	def __neg__(self): return NambuTensor(np.negative(self.data))
	def conj(self): return NambuTensor(np.conjugate(self.data))
	def trans(self): return NambuTensor(np.transpose(self.data,axes=(1,0,2,3))) ### Transposes the matrix indices 
		

