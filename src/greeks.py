def build_delta_engine(pricing_fn):
    """
    Wraps the pricing function with jax.grad to calculate the exact Delta Jacobian vector.
    jax.grad computes derivatives with respect to the first positional argument (S).
    """
    delta_fn = jax.grad(pricing_fn, argnums=0)
    return delta_fn

def build_gamma_engine(pricing_fn):
    """
    Wraps the pricing function with jax.hessian to compute the complete 
    N x N second-derivative matrix (Gammas and Cross-Gammas).
    Differentiates twice with respect to the first positional argument (S).
    """
    hessian_fn = jax.hessian(pricing_fn, argnums=0)
    return hessian_fn
