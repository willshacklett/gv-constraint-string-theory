import numpy as np

# Base values
b_k = 0.2776
b_lambda = 2.1204

A = 1.70  # leading constant

def compute(bk, bl):
    k_c = 1.0 / bk
    lambda_c = 1.0 / bl
    product = k_c * lambda_c
    residual = product - A
    return k_c, lambda_c, product, residual

print("\n=== RESIDUAL ANALYSIS ===\n")

# Perturbations
deltas = [-0.05, -0.02, 0.00, 0.02, 0.05]

lambdas = []
residuals = []

for i, d in enumerate(deltas):
    bk = b_k * (1 + d)
    bl = b_lambda * (1 + d)

    k_c, lambda_c, product, residual = compute(bk, bl)

    lambdas.append(lambda_c)
    residuals.append(residual)

    print(f"case {i}: λ_c={lambda_c:.4f}, product={product:.4f}, residual={residual:.4f}")

lambdas = np.array(lambdas)
residuals = np.array(residuals)

print("\n--- Residual Stats ---")
print(f"mean residual = {np.mean(residuals):.4f}")
print(f"std residual  = {np.std(residuals):.4f}")

# Try simple correlation with λ
corr = np.corrcoef(lambdas, residuals)[0, 1]

print("\n--- Correlation ---")
print(f"corr(λ_c, residual) = {corr:.4f}")

print("\nInterpretation:")
print("If residual correlates with λ, propagation term dominates ε")

print("\nDone.\n")
