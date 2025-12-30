

# describes the general placeholder and template for the self-energies of the system

# should have flatten and reflatten ffunctions which describe how it can get vectorized

# sigma should indicate which variables it does not depend on? 

# should know about self-consistency equation and its derivative i.e. jacobian automatically! 
# should indicate its diagonal or block-diagonal or somethingW

# should have a function which takes gr and f and computes the self-energy of the system 
# The function takes THE OBJECT and computes sigma_r self-consistently 
# This is since the self-energy_r is self-consistent even in dynamics and sigma_K will follow from f and others
# in principle sigma should also have sigma_r sigma_K returns which depend on f and gr

from abc import ABC, abstractmethod

from nambu_class import NambuTensor 

import jax
import jax.numpy as jnp
from jax import tree_util
from custom_optimizer import custom_jax_trapz as custom_integrate

# a set of rules describing how sigma is computed from g thats it! 
@tree_util.register_pytree_node_class
class SelfEnergy(ABC):
     
    def __init__(self, scattering_rate: float, omega_arr):
        self.scattering_rate = scattering_rate
        self.omega_arr = omega_arr
        self.system_shape = omega_arr.shape[-1]
        self.sigma_shape = self._sigma_shape()
        self.sigma_indicies = self._get_sigma_indicies()

    @abstractmethod
    def _sigma_r(self,gr,f):
        pass

    @abstractmethod
    def _sigma_shape(self):
        pass

    def _self_consistency(self,system_state,sigma):
        return sigma - self._sigma_r(system_state)

    """
    @abstractmethod
    def _sigma_jacobian(self,system_state,z):
        pass
    """ 

    @abstractmethod
    def _get_sigma_indicies(self):
        pass

    def _unflatten_sigma(self,data):
        #jitted_flat = jax.jit(self._unflatten_nambu_object, static_argnums = (1,2))
        return self._unflatten_nambu_object(data, self.sigma_shape, included_indices = self.sigma_indicies)

    def _unflatten_nambu_object(self,data, data_shape, included_indices = (0,1,2,3)): 
        data_reshape_size = data_shape + (len(included_indices) *2, )  
        new_data = (jnp.reshape(data,data_reshape_size))
        axes = (new_data.ndim-1,) + tuple(range(0, new_data.ndim-1))
        new_data = jnp.transpose(new_data,axes = axes) 
        out_data = NambuTensor(new_data[0] + 1j*new_data[1],included_indices[0])
        for i in range(1,len(included_indices)): 
            out_data += NambuTensor(new_data[i*2] + 1j*new_data[i*2 + 1],included_indices[i])

        return out_data 

    def tree_flatten(self):
        children = self.scattering_rate, self.omega_arr
        aux_data = None
        return children, aux_data

    def tree_unflatten(aux_data, children):
        scattering_rate, omega_arr = children
        return SelfEnergy(scattering_rate, omega_arr)


class ElasticScattering(SelfEnergy):

    def __init__(self, scattering_rate: float, theta_arr, omega_arr):
        super().__init__(scattering_rate, theta_arr, omega_arr)
        self.name = 'Elastic'
    
    def _sigma_r(self,system_state):
        return -0.5j*self.scattering_rate * (system_state.gr)._integrate(integration_weights = self.theta_arr, integration_axis = 1)

    def _sigma_shape(self):
        return (self.omega_arr.shape[0],1)

    def _get_sigma_indicies(self):
        return (1,2,3)

    def _sigma_jacobian(self,system_state,z):
        grid_shape = (self.omega_arr.shape[0],self.theta_arr.shape[0])
        index_size = len(self._get_sigma_indicies())* 2

        crt_g_box = (system_state.gr._flatten_nambu_object(included_indices=(1,2,3))).reshape((6,) + grid_shape)

        g_tensor = jnp.einsum('ijk,mjk -> mijk',crt_g_box, crt_g_box)
        g_tensor = g_tensor - jnp.tensordot(jnp.identity(index_size), jnp.ones((self.omega_arr.shape[0],self.theta_arr.shape[0])),axes=0) 
        g_tensor = jnp.einsum('ijkl, kl -> ijkl', g_tensor, 1/z)
        
        #TODO: should somehow indicate that there is a delta function in energy! for now just adding a dummy index in energy and putting the values on diagonal
        #TODO: This is technically wrong result for now, but lets keep it and see what happens 
        g_tensor_out = jnp.einsum('ijkl, m -> imjkl', g_tensor, jnp.ones(self.omega_arr.shape[0]))
        
        return  (-0.5j*self.scattering_rate * g_tensor_out).real #TODO: Figure out how 1j factor mixes things? This is very wrong probably


class DynesScattering(SelfEnergy):

    def __init__(self, scattering_rate: float, theta_arr, omega_arr):
        super().__init__(scattering_rate, theta_arr, omega_arr)
        self.name = 'Dynes'

    def _sigma_r(self,system_state):
        return NambuTensor(jnp.ones((1,1)) * -0.5j * self.scattering_rate,3)
    
    def _sigma_shape(self):
        return (1,1)

    def _get_sigma_indicies(self):
        return (3,)

    def _sigma_jacobian(self,system_state,z):
        index_size = self._get_sigma_indicies()
        return jnp.zeros((2 * len(index_size), 6, self.omega_arr.shape[0],self.theta_arr.shape[0]),dtype = jnp.float32)

@tree_util.register_pytree_node_class
class PhononScattering(SelfEnergy):

    def __init__(self, scattering_rate: float, omega_arr, temperature):
        super().__init__(scattering_rate, omega_arr)
        self.name = 'Phonon' 
        self.temperature = temperature
    
    def _sigma_r(self,gr,f):
        gk = gr @ f + f @ gr._involution()
        #epsilon_diff = jnp.outer(self.omega_arr ,jnp.ones_like(self.omega_arr)) - jnp.outer(jnp.ones_like(self.omega_arr),self.omega_arr)
        
        epsilon_diff = self.omega_arr[None,:] - self.omega_arr[:,None]

        kernel1 = epsilon_diff**2/(jnp.abs(jnp.tanh(epsilon_diff/2/self.temperature)) + 1e-4)
        kernel2 = (epsilon_diff**2 * jnp.sign(epsilon_diff))

        deps = jnp.diff(self.omega_arr,prepend= self.omega_arr[0])

        #* Somehow this avoids large memory allocation
        #? Should use trapezoid rule and not sum like this but ok 
        integrated1 = jnp.einsum('...i,ij -> ...j', (gr*deps).data, kernel1)
        integrated2 = jnp.einsum('...i,ij -> ...j', (gk*deps).data, kernel2)
        return NambuTensor( -0.25j * self.scattering_rate/jnp.max(self.omega_arr)**3 * (integrated1 - 0.5 * integrated2),None)


    def _sigma_shape(self):
        return (self.omega_arr.shape[0],)

    def _get_sigma_indicies(self):
        return (2,3)


    def tree_flatten(self):
        children = self.scattering_rate, self.omega_arr, self.temperature
        aux_data = None
        return children, aux_data

    def tree_unflatten(aux_data, children):
        scattering_rate, omega_arr, temperature = children
        return SelfEnergy(scattering_rate, omega_arr, temperature)
