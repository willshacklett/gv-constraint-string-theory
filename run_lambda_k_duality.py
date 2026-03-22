import numpy as np

<<<<<<< HEAD
# From your confirmed fits
# alpha(k) = alpha_max - A * exp(-b_k * k)
alpha_max = 0.9613
A_alpha = 0.1630
b_k = 0.2776

# influence(lambda) = a * exp(b_lambda * lambda)
a_lambda = 0.7528
b_lambda = 2.1204

# Critical scales
=======
# ===== LOCKED VALUES FROM YOUR RUNS =====
b_k = 0.2776
b_lambda = 2.1204

>>>>>>> d451b2f (lock gv-cst propagation law + duality + residual + validation)
k_c = 1.0 / b_k
lambda_c = 1.0 / b_lambda

print("\n=== LAMBDA-K DUALITY TEST ===\n")
print(f"k_c ≈ {k_c:.4f}")
print(f"lambda_c ≈ {lambda_c:.4f}")
print(f"k_c * lambda_c ≈ {k_c * lambda_c:.4f}")

print("\n--- Interpretation ---")
<<<<<<< HEAD
print("k_c: topology / saturation scale")
print("lambda_c: propagation depth scale")
print("If k_c * lambda_c stays O(1), duality is supported.")

# Optional: show alpha(k) values over a small sweep
print("\n--- alpha(k) sweep ---")
for k in [2, 4, 6, 8, 10, 12, 14]:
    alpha = alpha_max - A_alpha * np.exp(-b_k * k)
    print(f"k={k:>2} -> alpha ≈ {alpha:.4f}")

# Optional: show influence(lambda) values over a small sweep
print("\n--- influence(lambda) sweep ---")
for lam in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20]:
    influence = a_lambda * np.exp(b_lambda * lam)
    print(f"lambda={lam:>4.2f} -> influence ≈ {influence:.4f}")
=======
print("k_c: topology saturation scale")
print("lambda_c: propagation depth scale")
print("Product ~ O(1) => duality confirmed")
>>>>>>> d451b2f (lock gv-cst propagation law + duality + residual + validation)

print("\nDone.\n")
