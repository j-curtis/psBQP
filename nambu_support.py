import numpy as np
from scipy import integrate as intg
from scipy import optimize as opt

import jax 
import jax.numpy as jnp

"""A support module handling all the pauli matrix manipulations in nambu space
"""

def get_bcs_gap_constant() -> float:
    """A function returning the value of the bcs gap constant

    Returns:
        float: BCS gap constant 
    """
    return 2.*jnp.exp(np.euler_gamma)/jnp.pi

def get_bcs_ratio() -> float:
    """ A function returning the value of the ratio of Delta(0)/Tc in the BCS limit

    Returns:
        float: BCS ratio
    """
    return 2./get_bcs_gap_constant()  


def get_pauli_matrix(index = None) -> jnp.array:
    """A function returning the Pauli matrices or the whole list (if index is none)

    Args:
        index (int, optional): The index of the Pauli matrix from 0 to 3 or - or +. Defaults to None.

    Raises:
        ValueError: If the index is not from 0,1,2,3,+,-,x,y,z,I

    Returns:
        jnp.array: The Pauli matrix or the whole list if index is None 
    """
    pauli = [jnp.eye(2,dtype=complex), jnp.array([[0.j,1.],[1.,0.j]]), jnp.array([[0.j,-1.j],[1.j,0.j]]), jnp.array([[1.0,0.j],[0.j,-1.]]) ]
    if index is None:
        # If not index is given return the whole list
        return pauli
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

def nambu_mul(mat_x:jnp.array,mat_b: jnp.array) -> jnp.array: # was _NambuMul before
    """A function returning the product of two pauli matrices with possibly other dimensions as well

    Args:
        mat_x (jnp.array): matrix on the left
        mat_b (jnp.array): matrix on the right

    Returns:
        jnp.array: product of the two Pauli matrices mat_x @ mat_b
    """
    # returns the usual matrix product of two pauli matrices while other indicies are left the same
    return jnp.einsum('ijnm,jknm->iknm', mat_x,mat_b)

def nambu_scalar2nambu(scalar_x: jnp.array) -> jnp.array:
    """A function returning a Nambu compatible tensor from a scalar array

    Args:
        scalar_x (jnp.array): a scalar array

    Returns:
        jnp.array: a Nambu compatible tensor outer product with nambu identity
    """
    ### Promotes a scalar tensor function to a Nambu compatible tensor 
    return jnp.tensordot(jnp.ones((2,2),dtype=complex),scalar_x,axes=0)

def nambu_det(mat_x:jnp.array) -> jnp.array:
    """A function returning the determinant of a Nambu tensor. The first two indicies have to be (2,2).

    Args:
        mat_x (jnp.array): a Nambu tensor

    Raises:
            ValueError: if the matrix shape is not (2,2,...)

    Returns:
        jnp.array: the determinant of the Nambu tensor
    """
    # check the tensor has the correct shape
    if jnp.shape(mat_x)[:2] != (2,2):
        raise ValueError("Matrix must be 2x2 in first two indicies, got {}".format(jnp.shape(mat_x)))
    # compute the determinant
    det = mat_x[0,0,...] * mat_x[1,1,...] - mat_x[0,1,...]*mat_x[1,0,...] 
    # returns the determinant of the matrix w.r.t. nambu indicies 
    return det[None,None,...]
