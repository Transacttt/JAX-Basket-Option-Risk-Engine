import numpy as np

def numpy_cfd_delta(pricing_fn, S, sigma, w, corr, K, r, T, h=1e-4, option_type="call"):
    """
    Computes Delta vector using Central Finite Differences:
    d P / d S_i ≈ [P(S_i + h) - P(S_i - h)] / (2 * h)
    """
    N = len(S)
    delta_cfd = np.zeros(N)
    
    for i in range(N):
        S_plus = np.array(S, dtype=float)
        S_minus = np.array(S, dtype=float)
        
        S_plus[i] += h
        S_minus[i] -= h
        
        P_plus = pricing_fn(S_plus, sigma, w, corr, K, r, T, option_type)
        P_minus = pricing_fn(S_minus, sigma, w, corr, K, r, T, option_type)
        
        delta_cfd[i] = (P_plus - P_minus) / (2 * h)
        
    return delta_cfd


def numpy_cfd_hessian(pricing_fn, S, sigma, w, corr, K, r, T, h=1e-4, option_type="call"):
    """
    Computes Hessian Matrix (Gammas and Cross-Gammas) using 4-point CFD stencil:
    d²P / (dS_i dS_j) ≈ [P(S_i+h, S_j+h) - P(S_i+h, S_j-h) - P(S_i-h, S_j+h) + P(S_i-h, S_j-h)] / (4 * h²)
    """
    N = len(S)
    hessian_cfd = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                # Standard diagonal Gamma: [P(S+h) - 2P(S) + P(S-h)] / h²
                S_plus = np.array(S, dtype=float)
                S_minus = np.array(S, dtype=float)
                S_plus[i] += h
                S_minus[i] -= h
                
                P_center = pricing_fn(S, sigma, w, corr, K, r, T, option_type)
                P_plus = pricing_fn(S_plus, sigma, w, corr, K, r, T, option_type)
                P_minus = pricing_fn(S_minus, sigma, w, corr, K, r, T, option_type)
                
                hessian_cfd[i, j] = (P_plus - 2 * P_center + P_minus) / (h ** 2)
            else:
                # Off-diagonal Cross-Gamma: 4-point stencil evaluation
                S_pp, S_pm = np.array(S, dtype=float), np.array(S, dtype=float)
                S_mp, S_mm = np.array(S, dtype=float), np.array(S, dtype=float)
                
                S_pp[i] += h; S_pp[j] += h
                S_pm[i] += h; S_pm[j] -= h
                S_mp[i] -= h; S_mp[j] += h
                S_mm[i] -= h; S_mm[j] -= h
                
                P_pp = pricing_fn(S_pp, sigma, w, corr, K, r, T, option_type)
                P_pm = pricing_fn(S_pm, sigma, w, corr, K, r, T, option_type)
                P_mp = pricing_fn(S_mp, sigma, w, corr, K, r, T, option_type)
                P_mm = pricing_fn(S_mm, sigma, w, corr, K, r, T, option_type)
                
                hessian_cfd[i, j] = (P_pp - P_pm - P_mp + P_mm) / (4 * (h ** 2))
                
    return hessian_cfd
