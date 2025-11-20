
import jax
import jax.numpy as jnp

from jax.scipy.sparse.linalg import gmres

#jax.config.update("jax_disable_jit", True)


def custom_jax_trapz(y: jnp.ndarray, x = None, axis = -1) -> jnp.ndarray:

    if x is None:
        dx = 1/jnp.size(y, axis = axis) * jnp.ones(y.shape[axis])
    else:
        dx = jnp.diff(x, axis = 0)
        dx = jnp.append(dx[0], dx)

    avg_y = 0.5 * (jnp.roll(y, -1, axis = axis) + y)

    if x is None:
        return jnp.sum(avg_y * dx, axis = axis)
    else:
        shape = [1] *y.ndim
        shape[axis] = dx.shape[0]
        dx_reshaped = dx.reshape(shape)
        return jnp.sum(dx_reshaped * avg_y, axis = axis)



#TODO make this work! 
def custom_matrix_inverter(matrix: jnp.ndarray, block_indicies = None) -> jnp.ndarray: 
    pass 


def _newton_nd(f, x0: jnp.ndarray, tol=1e-3, maxiter=20) -> jnp.ndarray:
    """A function performing newton step given a function f and x0. Jacobian is computed directly

    Args:
        f (function): function whose root is to be found
        x0 (jnp.ndarray): initial guess
        tol (float, optional): tolerance in error describing when the function should terminate. Defaults to 1e-3.
        maxiter (int, optional): maximum number of iterations to perform. Defaults to 20.

    Returns:
        jnp.ndarray: new value of x after newton steps were performed
    """
    # condition to break the loop 
    def cond_fun(state):
        x, i = state
        return jnp.logical_and(jnp.linalg.norm(f(x)) > tol, i < maxiter)

    # newton step 
    def body_fun(state):
        x, i = state

        J = jax.jacobian(f)(x)          # Jacobian (n×n)
        dx = jnp.linalg.solve(J, f(x))  # Newton step
        return (x - dx, i + 1)

    # jax while loop repeating the steps until the condition is met
    x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))
    return x_final


def _iterative_jax_solver(f1, f2, x0: jnp.ndarray, y0: jnp.ndarray, optimization_parameters = None ) -> jnp.ndarray:
    """A function which is finding root of f1 and f2 by iteratively performing newton steps

    Args:
        f1 (function of x,y): _description_
        f2 (function of x,y): _description_
        x0 (jnp.ndarray): initial guess for x 
        y0 (jnp.ndarray): initial guess for y
        optimization_parameters (dict, optional): Dictionary containing the optimization parameters like tolerance and maxiter of the total loop. Defaults to None.

    Returns:
        jnp.ndarray: final result of 
    """
    # If no optimization parameters are provided set the to some default
    if optimization_parameters is None:
        optimization_parameters = {"tol": 1e-3, "maxiter": 200}

    tol = optimization_parameters["tol"]
    maxiter = optimization_parameters["maxiter"]

    # condition to break the loop
    def cond_fun(state):
        x, y, i = state
        err = jnp.maximum(jnp.linalg.norm(f1(x, y)), jnp.linalg.norm(f2(x, y)))
        return jnp.logical_and(err > tol, i < maxiter)


    # newton step consisting of two individual newton steps
    def body_fun(state):
        x, y, i = state

        x_new = _newton_nd(lambda x_: f1(x_, y), x)
        y_new = _newton_nd(lambda y_: f2(x_new, y_), y)

        return (x_new, y_new, i + 1)

    # jax while loop repeating the steps until the condition is met
    x_final, y_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, y0, 0))
    return x_final, y_final


def _newton_nd_with_jacobian(func, jacobian_list: jnp.ndarray, x0: jnp.ndarray, tol=1e-3, maxiter=20) -> jnp.ndarray:
    """A function performing newton step given a function f and x0. Jacobian should be provided as a list of function which output 6x6 matrices 

    Args:
        func (function): Function whose root is to be found
        jacobian_list (jnp.ndarray): List of individual jacobian functions split into 6x6 matrices 
        x0 (jnp.ndarray): initial guess
        tol (_type_, optional): tolerance in error describing when the function should terminate. Defaults to 1e-3.
        maxiter (int, optional): maximum number of iterations to perform. Defaults to 20.

    Returns:
        jnp.ndarray: new value of x after newton steps were performed
    """
   
   # condition to break the loop
    def cond_fun(state):
        x, i = state
        return jnp.logical_and(jnp.linalg.norm(func(x)) > tol, i < maxiter)

    # newton step where it is assumed jacobian is properly formatted 
    def body_fun(state):
        x, i = state
        # values of f
        f = func(x)
        # evaluate jacobian list on the value of x 
        jacs_list = jacobian_list(x)
        # split into 6x6 blocks which can individually be solved
        # this is becausee sigma_epsilon does not have dependence on other sigma_epsilon' 
        x_ns = x.reshape(jnp.size(x)//6,6)
        f_ns = f.reshape(jnp.size(f)//6,6)
        
        # solve for each 6x6 block
        jac_n  = lambda dJ, df :jnp.linalg.solve(dJ, df)

        # vectorize the solving function
        vectorized_jacobian = jax.vmap(jac_n, in_axes=(0,0))
        # apply the vectorized function of all the individual sigma_epsilons
        dx_ns = vectorized_jacobian(jacs_list, f_ns)

        # update the solution by flattening individual solutions
        dx = dx_ns.flatten()
        
        return (x - dx, i + 1)

    x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))
    return x_final

# here f1 jacobian gives a list!
def _iterative_jax_solver_with_jacobian(f1, f2,x0: jnp.ndarray, y0: jnp.ndarray, f1_jacobian: jnp.ndarray, f2_jacobian = None, optimization_parameters = None ) -> jnp.ndarray:
    """A function which iteratively finds roots of two function f1 and f2 given an initial guess and a jacobian for f1 

    Args:
        f1 (function): first function whose root we are trying to find
        f2 (function): second function whose root we are trying to find
        x0 (jnp.ndarray): initial guess for x
        y0 (jnp.ndarray): initial guess for y
        f1_jacobian (jnp.ndarray): jacobian list of function f1 consisting of 6x6 blocks
        f2_jacobian (jnp.ndarray, optional): jacobian list of function f2. Defaults to None. ##This does not work for now
        optimization_parameters (dict, optional): A dictionary containing the optimization parameters like tolerance and maxiter. Defaults to None.

    Returns:
        jnp.ndarray: final result
    """

    if optimization_parameters is None:
        optimization_parameters = {"tol": 1e-3, "maxiter": 200}

    tol = optimization_parameters["tol"]
    maxiter = optimization_parameters["maxiter"]

    # condition to break the loop
    def cond_fun(state):
        x, y, i = state
        err = jnp.maximum(jnp.linalg.norm(f1(x, y)), jnp.linalg.norm(f2(x, y)))
        return jnp.logical_and(err > tol, i < maxiter)

    # newton step consisting of two individual newton steps
    def body_fun(state):
        x, y, i = state

        # solve x_new with the full jacobian
        x_function = lambda x_: f1(x_, y)
        # define the jacobian function for fixed y
        x_jacobian = lambda x_ : f1_jacobian(x_,y)
        x_new = _newton_nd_with_jacobian(x_function, x_jacobian, x)

        y_function = lambda y_: f2(x_new, y_)
        # Solve f2(x, y) = 0 for y given x_new
        y_new = _newton_nd(y_function, y)

        return (x_new, y_new, i + 1)

    # call the while loop
    x_final, y_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, y0, 0))
    return x_final, y_final


#TODO diagonal jac has to be passed in as well it is state ddependent but its diagonal so inverse is trivial!
def _iterative_solver(f, x0, jac = None,  optimization_parameters = None):
    if optimization_parameters is None:
        optimization_parameters = {"tol": 1e-3, "maxiter": 200}

    tol = optimization_parameters["tol"]
    maxiter = optimization_parameters["maxiter"]

    # condition to break the loop
    def cond_fun(state):
        x, i = state
        return jnp.logical_and(jnp.linalg.norm(f(x)) > tol, i < maxiter)

    # newton step 
    def body_fun(state):
        x, i = state
        #TODO: Change default to not computing the Jacobian at all!
        if jac is None:
            #J = jnp.ones((jnp.size(x), jnp.size(x)))
            #f_remat = jax.remat(f)
            print(x.shape)
            J = (jax.jacfwd(f)(x)).real
            dx = jnp.linalg.solve(J, f(x))
        
        else:
            #J = jac(x)
            Jf = lambda u, _x: jac(_x) @ u
            dx, info = gmres(lambda u: Jf(u,x), f(x), tol = 1e-5, maxiter= 200)  
        
        return (x - dx, i + 1)

    # call the while loop
    x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))

    return x_final

#! Not implemented yet!!
def _fast_iterative_solver(f, x0, jac = None, diagonal_jac = None,  optimization_parameters = None):
    if optimization_parameters is None:
        optimization_parameters = {"tol": 1e-3, "maxiter": 200}

    tol = optimization_parameters["tol"]
    maxiter = optimization_parameters["maxiter"]

    # condition to break the loop
    def cond_fun(state):
        x, i = state
        return jnp.logical_and(jnp.linalg.norm(f(x)) > tol, i < maxiter)

    # newton step 
    def body_fun(state):
        x, i = state
        #TODO: Change default to not computing the Jacobian at all!
        if jac is None:
            #J = jnp.ones((jnp.size(x), jnp.size(x)))
            J = (jax.jacobian(f)(x)).real
        else:
            J = jac(x)
        dx = jnp.linalg.solve(J, f(x))

        return (x - dx, i + 1)

    # call the while loop
    x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))

    return x_final


