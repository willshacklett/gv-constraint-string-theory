import numpy as np

<<<<<<< HEAD
# Locked fit parameters from your confirmed runs
b_k = 0.2776
b_lambda = 2.1204

# Baseline critical scales
k_c_base = 1.0 / b_k
lambda_c_base = 1.0 / b_lambda


def alpha_k(k, alpha_max=0.9613, A=0.1630, b=b_k):
    return alpha_max - A * np.exp(-b * k)


def influence_lambda(lam, a=0.7528, b=b_lambda):
    return a * np.exp(b * lam)


def report_case(name, bk, bl):
    kc = 1.0 / bk
    lc = 1.0 / bl
    prod = kc * lc
    print(f"{name:>12} | b_k={bk:.4f} | b_λ={bl:.4f} | k_c={kc:.4f} | λ_c={lc:.4f} | k_c*λ_c={prod:.4f}")


def main():
    print("\n=== DUALITY SWEEP ===\n")
    print(f"{'case':>12} | {'b_k':>8} | {'b_λ':>8} | {'k_c':>8} | {'λ_c':>8} | {'k_c*λ_c':>10}")
    print("-" * 78)

    # Baseline
    report_case("baseline", b_k, b_lambda)

    # Small perturbations to test invariance around the fitted values
    dk_fracs = [-0.05, -0.02, 0.00, 0.02, 0.05]
    dl_fracs = [-0.05, -0.02, 0.00, 0.02, 0.05]

    products = []

    for i, fk in enumerate(dk_fracs):
        bk = b_k * (1.0 + fk)
        bl = b_lambda * (1.0 + dl_fracs[i])
        kc = 1.0 / bk
        lc = 1.0 / bl
        prod = kc * lc
        products.append(prod)
        report_case(f"pair_{i+1}", bk, bl)

    print("-" * 78)
    products = np.array(products, dtype=float)
    print(f"{'mean':>12} | {'':8} | {'':8} | {'':8} | {'':8} | {np.mean(products):10.4f}")
    print(f"{'std':>12} | {'':8} | {'':8} | {'':8} | {'':8} | {np.std(products):10.4f}")
    print(f"{'min':>12} | {'':8} | {'':8} | {'':8} | {'':8} | {np.min(products):10.4f}")
    print(f"{'max':>12} | {'':8} | {'':8} | {'':8} | {'':8} | {np.max(products):10.4f}")

    print("\n--- Sample alpha(k) values ---")
    for k in [2, 4, 6, 8, 10, 12, 14]:
        print(f"k={k:>2} -> alpha ≈ {alpha_k(k):.4f}")

    print("\n--- Sample influence(λ) values ---")
    for lam in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20]:
        print(f"λ={lam:>4.2f} -> influence ≈ {influence_lambda(lam):.4f}")

    print("\nInterpretation:")
    print("If k_c * λ_c stays near O(1) under small perturbations, λ–k duality is structurally stable.")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
=======
# Locked values from your runs
b_k = 0.2776
b_lambda = 2.1204

def compute(bk, bl):
    k_c = 1.0 / bk
    lambda_c = 1.0 / bl
    return k_c, lambda_c, k_c * lambda_c

print("\n=== DUALITY SWEEP ===\n")

# Baseline
k_c, lambda_c, prod = compute(b_k, b_lambda)
print(f"baseline  | k_c={k_c:.4f} | lambda_c={lambda_c:.4f} | product={prod:.4f}")

# Perturbations
deltas = [-0.05, -0.02, 0.00, 0.02, 0.05]
products = []

for i, d in enumerate(deltas):
    bk = b_k * (1 + d)
    bl = b_lambda * (1 + d)
    k_c, lambda_c, prod = compute(bk, bl)
    products.append(prod)
    print(f"perturb_{i} | k_c={k_c:.4f} | lambda_c={lambda_c:.4f} | product={prod:.4f}")

products = np.array(products)

print("\n--- Stats ---")
print(f"mean = {products.mean():.4f}")
print(f"std  = {products.std():.4f}")
print(f"min  = {products.min():.4f}")
print(f"max  = {products.max():.4f}")

print("\nInterpretation:")
print("If product stays ~constant => duality is invariant")

print("\nDone.\n")
>>>>>>> d451b2f (lock gv-cst propagation law + duality + residual + validation)
