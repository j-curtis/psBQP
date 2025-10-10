
import jax
import jax.numpy as jnp

import nambu_support

#TODO: Generalize to be able to take axis argument
def custom_jax_trapz(y, x):
    dx = jnp.diff(x)
    avg_y = 0.5 * (y[:-1] + y[1:])
    return jnp.sum(dx * avg_y)


def _newton_nd(f, x0, tol=1e-3, maxiter=20):
    """
    Multidimensional Newton's method for vector input x ∈ R^n.
    Uses JAX Jacobian automatically.
    """
    def cond_fun(state):
        x, i = state
        return jnp.logical_and(jnp.linalg.norm(f(x)) > tol, i < maxiter)

    def body_fun(state):
        x, i = state

        J = jax.jacobian(f)(x)          # Jacobian (n×n)
        dx = jnp.linalg.solve(J, f(x))  # Newton step
        return (x - dx, i + 1)

    x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))
    return x_final


def _iterative_jax_solver(f1, f2, x0, y0, optimization_parameters = None ):
    """
    Alternating self-consistent root finder for vector x, y.

    f1: function f1(x, y) → R^n
    f2: function f2(x, y) → R^m
    """
    if optimization_parameters is None:
        optimization_parameters = {"tol": 1e-3, "maxiter": 200}

    tol = optimization_parameters["tol"]
    maxiter = optimization_parameters["maxiter"]
    def cond_fun(state):
        x, y, i = state
        err = jnp.maximum(jnp.linalg.norm(f1(x, y)), jnp.linalg.norm(f2(x, y)))
        return jnp.logical_and(err > tol, i < maxiter)

    def body_fun(state):
        x, y, i = state

        # Solve f1(x, y) = 0 for x given y
        #x_new = jnp.zeros_like(x)
        x_new = _newton_nd(lambda x_: f1(x_, y), x)
        # Solve f2(x, y) = 0 for y given x_new
        y_new = _newton_nd(lambda y_: f2(x_new, y_), y)

        return (x_new, y_new, i + 1)

    x_final, y_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, y0, 0))
    return x_final, y_final


def _newton_nd_with_jacobian(func, jacobian_list, x0, tol=1e-3, maxiter=20):
    """
    Multidimensional Newton's method for vector input x ∈ R^n.
    Uses JAX Jacobian automatically.
    """
    def cond_fun(state):
        x, i = state
        return jnp.logical_and(jnp.linalg.norm(func(x)) > tol, i < maxiter)

    def body_fun(state):
        x, i = state
        f = func(x)
        jacs_list = jacobian_list(x)
        x_ns = x.reshape(jnp.size(x)//6,6)
        f_ns = f.reshape(jnp.size(f)//6,6)
        #print(jnp.shape(x))
        #print(jnp.shape(x_ns))
        #print(jnp.shape(jacs_list)) 
        jac_n  = lambda dJ, df :jnp.linalg.solve(dJ, df)
        #print(jnp.shape(jac_n(jacs_list[0],f_ns[0])))
        vectorized_jacobian = jax.vmap(jac_n, in_axes=(0,0))
        dx_ns = vectorized_jacobian(jacs_list, f_ns)
        dx = dx_ns.flatten()
        #dx = jnp.zeros_like(x)
        """
        J = jax.jacobian(f)(x)          # Jacobian (n×n)
        dx = jnp.linalg.solve(J, f(x))  # Newton step
        """
        return (x - dx, i + 1)

    x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))
    return x_final

# here f1 jacobian gives a list!
def _iterative_jax_solver_with_jacobian(f1, f2,x0, y0, f1_jacobian, f2_jacobian = None, optimization_parameters = None ):
    """
    Alternating self-consistent root finder for vector x, y.

    f1: function f1(x, y) → R^n
    f2: function f2(x, y) → R^m
    """
    if optimization_parameters is None:
        optimization_parameters = {"tol": 1e-3, "maxiter": 200}

    tol = optimization_parameters["tol"]
    maxiter = optimization_parameters["maxiter"]
    def cond_fun(state):
        x, y, i = state
        err = jnp.maximum(jnp.linalg.norm(f1(x, y)), jnp.linalg.norm(f2(x, y)))
        return jnp.logical_and(err > tol, i < maxiter)

    def body_fun(state):
        x, y, i = state

        # Solve f1(x, y) = 0 for x given y
        #x_new = jnp.zeros_like(x)
        # solve x_new with the full jacobian
        x_function = lambda x_: f1(x_, y)
        x_jacobian = lambda x_ : f1_jacobian(x_,y)
        x_new = _newton_nd_with_jacobian(x_function, x_jacobian, x)

        y_function = lambda y_: f2(x_new, y_)
        # Solve f2(x, y) = 0 for y given x_new
        y_new = _newton_nd(y_function, y)

        return (x_new, y_new, i + 1)

    x_final, y_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, y0, 0))
    return x_final, y_final

