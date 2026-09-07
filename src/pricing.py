import jax.numpy as jnp
import numpy as np

def generate_synthetic_market_data(num_assets=5, seed=42):
    np.random.seed(seed)
    
    # 1. Spot Prices (S): Uniform random prices between £50 and £150
    S = jnp.array(np.random.uniform(50.0, 150.0, size=num_assets))
    
    # 2. Volatilities (sigma): Annualized vols between 10% and 40%
    sigma = jnp.array(np.random.uniform(0.10, 0.40, size=num_assets))
    
    # 3. Asset Weights (w): Equal weighting normalized to 1.0
    w = jnp.ones(num_assets) / num_assets
    
    # 4. Correlation Matrix (Sigma): Random positive semi-definite matrix
    A = np.random.uniform(-0.2, 0.8, size=(num_assets, num_assets))
    corr = np.dot(A, A.T)
    # Scale to valid correlation matrix (1.0 on diagonals)
    inv_d = np.diag(1.0 / np.sqrt(np.diag(corr)))
    corr_matrix = jnp.array(inv_d @ corr @ inv_d)
    
    # 5. Contract Constants
    K = float(jnp.sum(S * w))  # At-the-money (ATM) strike price
    r = 0.05                   # 5% Risk-free rate
    T = 1.0                    # 1-year maturity
    
    return S, sigma, w, corr_matrix, K, r, T


# We now define the continuous, fully differentiable pricing function for the multi-asset basket option using
# Milevsky-Posner Moment-Matching.

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

# Enable 64-bit precision in JAX for financial accuracy
jax.config.update("jax_enable_x64", True)

def basket_option_price(S, sigma, w, corr, K, r, T, option_type="call"):
    """
    Computes the price of a Multi-Asset Basket Option using Moment-Matching.
    """
    # 1. Forward prices for each asset: F_i = S_i * exp(r * T)
    F = S * jnp.exp(r * T)
    
    # 2. First Moment (Basket Expected Forward Value, M1)
    M1 = jnp.sum(w * F)
    
    # 3. Covariance Matrix construction: Cov_ij = sigma_i * sigma_j * corr_ij
    cov_matrix = jnp.outer(sigma, sigma) * corr
    
    # 4. Second Moment (M2)
    wF = w * F
    exp_cov_T = jnp.exp(cov_matrix * T)
    M2 = jnp.dot(wF, jnp.dot(exp_cov_T, wF))
    
    # 5. Effective Lognormal Parameters
    v_sq = jnp.log(M2 / (M1 ** 2))
    v = jnp.sqrt(v_sq)
    
    # 6. Black-Scholes d1 and d2
    d1 = (jnp.log(M1 / K) + 0.5 * v_sq) / v
    d2 = d1 - v
    
    # 7. Discount Factor
    df = jnp.exp(-r * T)
    
    # 8. Option Payoff Calculation using JAX continuous CDF
    if option_type == "call":
        price = df * (M1 * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = df * (K * norm.cdf(-d2) - M1 * norm.cdf(-d1))
        
    return price
