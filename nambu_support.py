import numpy as np
from scipy import integrate as intg
from scipy import optimize as opt

import jax 
import jax.numpy as jnp


def get_bcs_gap_constant():
    return 2.*jnp.exp(np.euler_gamma)/jnp.pi ### 2e^gamma/pi constant often appearing in BCS integrals 

def get_bcs_ratio():
    return 2./get_bcs_gap_constant() ### Ratio of Delta(0)/Tc in BCS limit 


def get_pauli_matrix(index = None):
    
    pauli = [jnp.eye(2,dtype=complex), jnp.array([[0.j,1.],[1.,0.j]]), jnp.array([[0.j,-1.j],[1.j,0.j]]), jnp.array([[1.0,0.j],[0.j,-1.]]) ]
    if index is None:
        return pauli
    elif type(index) is str:
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
        return pauli[index]

def nambu_mul(mat_x,mat_b): # was _NambuMul before
    return jnp.einsum('ijnm,jknm->iknm', mat_x,mat_b)

def nambu_scalar2nambu(scalar_x):
    ### Promotes a scalar tensor function to a Nambu compatible tensor 
    return jnp.tensordot(jnp.ones((2,2),dtype=complex),scalar_x,axes=0)

def nambu_det(mat_x):
    if jnp.shape(mat_x)[:2] != (2,2):
        raise ValueError("Matrix must be 2x2 in first two indicies, got {}".format(jnp.shape(mat_x)))
    det = mat_x[0,0,...] * mat_x[1,1,...] - mat_x[0,1,...]*mat_x[1,0,...] 

    return det[None,None,...]
