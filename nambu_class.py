from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util
import jax 
import custom_optimizer
#jax.config.update("jax_disable_jit", True)

# contain the description of the Nambu class and its methods 

# has to be a pytree class 

# should be tested by jitting nambu addition and checking whether the jitter complains
@tree_util.register_pytree_node_class
class NambuTensor:

    def __init__(self, data_in: jnp.array, pauli_channel = None):
        
        # if the input data has a lot of dimensions then we assume its already a nambu tensor

        # if not a nambu tensor make one
        #NOTE: The data_in is not allowed to be of the shape (2,2)
        if pauli_channel is None:
            if (len(data_in.shape) < 2):

                data = jnp.tensordot(jnp.ones((2,2),dtype=complex),data_in,axes=0)

            elif (jnp.shape(data_in)[:2]!= (2,2)):
                data = jnp.tensordot(jnp.ones((2,2),dtype=complex),data_in,axes=0)
        
            else:
                # if a nambu tensor then just take the data
                data = jnp.array(data_in,dtype=complex)

        # assuming that if you pass in a pauli channel that you want to tensordot the data!
        # this means its possible to get data with (2,2) structure if a pauli channel is specified
        else:
            pauli_matrix = get_pauli_matrix(index = pauli_channel)
            data = jnp.tensordot(pauli_matrix,data_in,axes=0)

        self.data = data
        self.data_shape = data.shape

    def __matmul__(self, other, expansion_indices_left = None, expansion_indices_right = None):
        if not (expansion_indices_left is None and expansion_indices_right is None):
            self_object,other_object = self._make_same_dimension(other,expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
            if self_object.data_shape != other_object.data_shape:
                raise ValueError("Matrices must be same shape, got", self_object.data_shape, 'and', other_object.data_shape)
            result = jnp.einsum('ij...,jk...->ik...', self_object.data, other_object.data)
        else:
            result = jnp.einsum('ij...,jk...->ik...', self.data, other.data)
        return NambuTensor(result,None)

    def __mul__(self,other,expansion_indices_left = None, expansion_indices_right = None):   
        if type(other) is NambuTensor and not (expansion_indices_left is None and expansion_indices_right is None):
            self_object, other_object = self._make_same_dimension(other = other,expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
            result = self_object.data * other_object.data
        else:
            result = self.data * other

        return NambuTensor(result,None)

    def __rmul__(self,other,expansion_indices_left = None, expansion_indices_right = None):   
        if type(other) is NambuTensor:
            self_object, other_object = self._make_same_dimension(other = other,expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
            result = self_object.data * other_object.data
        else:
            result = self.data * other

        return NambuTensor(result,None)

    def __del__(self):
        del self.data

    # when operating with nambu matrices, automatically expand if the shape of the two matrrices is not the same! 
    def _make_same_dimension(self, other, expansion_indices_left = None, expansion_indices_right = None):
        
        # if the other object is not nambu tensor then make it one by expanding its data
        if expansion_indices_left is None and expansion_indices_right is None:
            return self, other

        #* It assumes here that the other object knows about how many dimensions there are but just has 1 size and not full size 
        elif expansion_indices_left is None and expansion_indices_right is not None:
            new_data = other.data
            #new_data = jnp.expand_dims(other.data, axis = tuple(expansion_indices_right)) 
            new_data *= jnp.ones(self.data.shape)
            other_obj = NambuTensor(new_data,None)
            return self, other_obj

        elif expansion_indices_left is not None and expansion_indices_right is None:
            new_data = self.data
            #new_data = jnp.expand_dims(self.data, axis = tuple(expansion_indices_left)) 
            new_data *= jnp.ones(other.data.shape)

            self_obj = NambuTensor(new_data,None)
            return self_obj, other

        else:
            #new_data_left = jnp.expand_dims(self.data, axis = tuple(expansion_indices_left + 2)) 
            new_data_left *= jnp.ones(other.data.shape)

            #new_data_right = jnp.expand_dims(other.data, axis = tuple(expansion_indices_right + 2)) 
            new_data_right *= jnp.ones(self.data.shape)

            self_obj = NambuTensor(new_data_left,None)
            other_obj = NambuTensor(new_data_right,None)
            return self_obj, other_obj

    def _binary_ewise(self, other, op, expansion_indices_left = None, expansion_indices_right = None):
        if type(other) is NambuTensor:
            self_object, other_object = self._make_same_dimension(other,expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)	
            return NambuTensor(op(self_object.data, other_object.data),None)
        else:
            return NambuTensor(op(self.data, other),None)

    def __add__(self, other, expansion_indices_left = None, expansion_indices_right = None): return self._binary_ewise(other, jnp.add, expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
    def __radd__(self, other, expansion_indices_left = None, expansion_indices_right = None): return self._binary_ewise(other, jnp.add, expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
    def __sub__(self, other, expansion_indices_left = None, expansion_indices_right = None): return self._binary_ewise(other, jnp.subtract, expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
    def __truediv__(self, other, expansion_indices_left = None, expansion_indices_right = None): return self._binary_ewise(other, jnp.true_divide, expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
    def __rtruediv__(self, other, expansion_indices_left = None, expansion_indices_right = None): return self._binary_ewise(other, lambda a,b: jnp.true_divide(b,a), expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
    def __neg__(self): return NambuTensor(jnp.negative(self.data))

    def __str__(self):  
        id_trace = self._trace(pauli_index = 0)
        x_trace = self._trace(pauli_index = 1)
        y_trace = self._trace(pauli_index = 2)
        z_trace = self._trace(pauli_index = 3)

        return f"Id Trace: {id_trace} \n X Trace: {x_trace} \n Y Trace: {y_trace} \n Z Trace: {z_trace} \n"

    def __getitem__(self,index): 
        return NambuTensor(self.data[:,:,index],None)

    def _conj(self): return NambuTensor(jnp.conjugate(self.data), None)
    def _transpose(self): 
        axes = (1, 0) + tuple(range(2, self.data.ndim)) 
        return NambuTensor(jnp.transpose(self.data,axes = axes),None) ### Transposes the matrix indices 
    
    def _commutator(self,other,expansion_indices_left = None, expansion_indices_right = None): 
        self_object, other_object = self._make_same_dimension(other,expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
        return self_object @ other_object - self_object @ self 
    def _anti_commutator(self,other, expansion_indices_left = None, expansion_indices_right = None): 
        self_object, other_object = self._make_same_dimension(other,expansion_indices_left = expansion_indices_left, expansion_indices_right = expansion_indices_right)
        return self_object @ other_object + other_object @ self_object 

    def _trace(self,pauli_index = None): 
        pauli_matrix = get_pauli_matrix(index = pauli_index)
        return  pauli_matrix[0,0]*self.data[0,0,...]+ pauli_matrix[0,1]*self.data[1,0]+ pauli_matrix[1,0]*self.data[0,1]+ pauli_matrix[1,1]*self.data[1,1]

    def _determinant(self): 
        det = self.data[0,0,...]*self.data[1,1,...] - self.data[0,1,...]*self.data[1,0,...]
        # still returns a nambu tensor object so it can be used in divide and multiply
        return det[None,None,...]

    def _involution(self):
        return  NambuTensor(jnp.ones(self.data_shape[2:]), 3) @ (self._conj())._transpose() @  NambuTensor(jnp.ones(self.data_shape[2:]), 3) 

    def _integrate(self, integration_weights = None, integration_axis = None):
        ### We will simply sum this over all indices to return a single number 
        if integration_axis is None: integration_axis = 2
        else: integration_axis = integration_axis + 2 

        #new_data = jnp.expand_dims(self.data, axis = tuple(expansion_indices_left)) 

        result = custom_optimizer.custom_jax_trapz(self.data,integration_weights, integration_axis)
 
        result = jnp.expand_dims(result, axis = (integration_axis,))
        return NambuTensor(result,None)

    def _flatten_nambu_object(self, included_indices = (0,1,2,3)): 
        id_trace = self._trace(pauli_index = 0)/2
        x_trace = self._trace(pauli_index = 1)/2
        y_trace = self._trace(pauli_index = 2)/2
        z_trace = self._trace(pauli_index = 3)/2

        out_data_unfilt = jnp.stack((id_trace.real,id_trace.imag,x_trace.real,x_trace.imag,y_trace.real,y_trace.imag,z_trace.real,z_trace.imag))
        included_indicies_modified = jnp.sort(jnp.append(jnp.array(included_indices) * 2, jnp.array(included_indices) * 2 + 1))
        out_data = out_data_unfilt[included_indicies_modified]
        
        out_data = jnp.array(out_data)
        axes = tuple(range(1, out_data.ndim)) + (0,)
        return (jnp.transpose(out_data,axes = axes)).flatten() # output is xyz, xyz, xyz .... 

    def _flatten_nambu_object_to_complex(self, included_indices = (0,1,2,3)): 
        id_trace = self._trace(pauli_index = 0)/2
        x_trace = self._trace(pauli_index = 1)/2
        y_trace = self._trace(pauli_index = 2)/2
        z_trace = self._trace(pauli_index = 3)/2

        out_data_unfilt = jnp.stack((id_trace,id_trace,x_trace,y_trace,z_trace))
        included_indicies_modified = (jnp.array(included_indices))
        out_data = out_data_unfilt[included_indicies_modified]
        
        out_data = jnp.array(out_data)
        axes = tuple(range(1, out_data.ndim)) + (0,)
        return (jnp.transpose(out_data,axes = axes)).flatten() # output is xyz, xyz, xyz .... 


    @staticmethod
    def _unflatten_nambu_object(data, data_shape, included_indices = jnp.array([0,1,2,3])): 
        data_reshape_size = tuple(data_shape) + (jnp.size(included_indices) *2,)
        #print(data_reshape_size)
        new_data = (jnp.reshape(data,data_reshape_size))
        axes = (new_data.ndim-1,) + tuple(range(0, new_data.ndim-1))
        data = jnp.transpose(new_data,axes = axes) 
        out_data = 0 

        for i in range(len(included_indices)): 
            out_data += NambuTensor(data[i*2] + 1j*data[i*2 + 1],included_indices[i])

        return out_data 

    #* Custom functions telling python how to handle the leaves of this object, needed for jitting
    #* It describes how this object can be made into jnp.array

    def tree_flatten(self):
        children = self.data, self.data_shape
        aux_data = None
        return children, aux_data

    def tree_unflatten(aux_data, children):
        data, data_shape = children
        return NambuTensor(data, None)

def get_pauli_matrix(index = None) -> jnp.array:
    """A function returning the Pauli matrices or the whole list (if index is none)

    Args:
        index (int, optional): The index of the Pauli matrix from 0 to 3 or - or +. Defaults to None.

    Raises:
        ValueError: If the index is not from 0,1,2,3,+,-,x,y,z,I

    Returns:
        jnp.array: The Pauli matrix or a identity if index is None 
    """
    pauli = [jnp.eye(2,dtype=complex), jnp.array([[0.j,1.],[1.,0.j]]), jnp.array([[0.j,-1.j],[1.j,0.j]]), jnp.array([[1.0,0.j],[0.j,-1.]]) ]
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
