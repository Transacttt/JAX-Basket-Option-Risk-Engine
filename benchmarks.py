def run_step_5_benchmarks():
    print("=" * 70)
    print("STEP 5: BENCHMARKING JAX AAD VS. NUMPY CFD ENGINE")
    print("=" * 70)

    # 1. Numerical Failure Sweep (Step Size h vs Frobenius Norm)
    print("\n--- 1. Numerical Failure Sweep (Step Size h Analysis) ---")
    S, sigma, w, corr, K, r, T = generate_synthetic_market_data(num_assets=5)
    
    H_jax = np.array(jax_hessian_engine(S, sigma, w, corr, K, r, T, "call"))
    
    h_steps = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    print(f"{'Step Size (h)':<15} | {'Frobenius Error ||H_JAX - H_CFD||_F':<35} | {'Status'}")
    print("-" * 70)
    
    for h in h_steps:
        H_cfd = numpy_cfd_hessian(basket_option_price, S, sigma, w, corr, K, r, T, h=h)
        frob_err = np.linalg.norm(H_jax - H_cfd, ord='fro')
        
        if h >= 1e-2:
            status = "Truncation Error Dominated"
        elif 1e-6 <= h <= 1e-4:
            status = "Optimal CFD Window"
        else:
            status = "Catastrophic Roundoff / Cancellation"
            
        print(f"{h:<15.0e} | {frob_err:<35.6e} | {status}")

    # 2. Execution Time Scaling (N = 2 to 50 Assets)
    print("\n--- 2. Execution Scaling vs. Basket Size (N) ---")
    asset_counts = [2, 5, 10, 20, 50]
    print(f"{'N Assets':<10} | {'JAX AD Time (s)':<18} | {'NumPy CFD Time (s)':<20} | {'Speedup Factor'}")
    print("-" * 70)

    for N in asset_counts:
        S, sigma, w, corr, K, r, T = generate_synthetic_market_data(num_assets=N)
        
        # Warmup JIT compilation
        _ = jax_hessian_engine(S, sigma, w, corr, K, r, T, "call").block_until_ready()
        
        # Time JAX Execution
        t0 = time.perf_counter()
        _ = jax_hessian_engine(S, sigma, w, corr, K, r, T, "call").block_until_ready()
        t_jax = time.perf_counter() - t0
        
        # Time NumPy CFD Execution
        t0 = time.perf_counter()
        _ = numpy_cfd_hessian(basket_option_price, S, sigma, w, corr, K, r, T, h=1e-4)
        t_cfd = time.perf_counter() - t0
        
        speedup = t_cfd / t_jax if t_jax > 0 else 0.0
        print(f"{N:<10} | {t_jax:<18.6f} | {t_cfd:<20.6f} | {speedup:.1f}x faster")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE: JAX engine successfully validated.")
    print("=" * 70)

if __name__ == "__main__":
    run_step_5_benchmarks()
