import numpy as np

<<<<<<< HEAD
# Locked baseline law pieces from your runs
A0 = 1.70
eps_a = 7.2387
eps_b = -3.4130

# Baseline fitted scales
=======
# ===== LOCKED LAW (from your fits) =====
A = 1.70
a = 7.2387
b = -3.4130

# ===== BASE VALUES =====
>>>>>>> d451b2f (lock gv-cst propagation law + duality + residual + validation)
b_k_base = 0.2776
b_lambda_base = 2.1204

def kc_from_bk(bk):
    return 1.0 / bk

def lambdac_from_bl(bl):
    return 1.0 / bl

<<<<<<< HEAD
def kc_pred_from_lambda(lam):
    # From your closed collapse relation:
    # k_c(λ) = (A0 + eps_a*λ + eps_b) / λ
    return (A0 + eps_a * lam + eps_b) / lam

def run_case(name, bk, bl):
    kc_obs = kc_from_bk(bk)
    lam = lambdac_from_bl(bl)
    kc_pred = kc_pred_from_lambda(lam)

    abs_err = abs(kc_pred - kc_obs)
    rel_err = abs_err / kc_obs if kc_obs != 0 else np.nan

    return {
        "name": name,
        "b_k": bk,
        "b_lambda": bl,
        "k_c_obs": kc_obs,
        "lambda_c": lam,
        "k_c_pred": kc_pred,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }

def main():
    print("\n=== CROSS-REGIME VALIDATION ===\n")

    # Simulated regime variations around fitted values
    # These stand in for different slices / regimes until you wire in full re-extraction
    cases = [
        ("baseline", b_k_base, b_lambda_base),
        ("regime_1", b_k_base * 0.95, b_lambda_base * 0.95),
        ("regime_2", b_k_base * 0.98, b_lambda_base * 0.98),
        ("regime_3", b_k_base * 1.02, b_lambda_base * 1.02),
        ("regime_4", b_k_base * 1.05, b_lambda_base * 1.05),
    ]

    results = [run_case(*case) for case in cases]

    print(
        f"{'case':>10} | {'b_k':>8} | {'b_λ':>8} | {'k_c(obs)':>9} | "
        f"{'λ_c':>8} | {'k_c(pred)':>10} | {'abs err':>8} | {'rel err':>8}"
    )
    print("-" * 92)

    for r in results:
        print(
            f"{r['name']:>10} | "
            f"{r['b_k']:8.4f} | "
            f"{r['b_lambda']:8.4f} | "
            f"{r['k_c_obs']:9.4f} | "
            f"{r['lambda_c']:8.4f} | "
            f"{r['k_c_pred']:10.4f} | "
            f"{r['abs_err']:8.4f} | "
            f"{r['rel_err']:8.4f}"
        )

    abs_errs = np.array([r["abs_err"] for r in results], dtype=float)
    rel_errs = np.array([r["rel_err"] for r in results], dtype=float)

    print("-" * 92)
    print(f"{'mean err':>10} | {'':8} | {'':8} | {'':9} | {'':8} | {'':10} | {np.mean(abs_errs):8.4f} | {np.mean(rel_errs):8.4f}")
    print(f"{'max err':>10}  | {'':8} | {'':8} | {'':9} | {'':8} | {'':10} | {np.max(abs_errs):8.4f} | {np.max(rel_errs):8.4f}")

    print("\nInterpretation:")
    print("If predicted k_c tracks observed k_c across regime perturbations,")
    print("the λ→k closure is structurally stable beyond the baseline fit.")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
=======
def kc_pred(lam):
    return (A + a * lam + b) / lam

print("\n=== CROSS REGIME VALIDATION ===\n")

cases = [
    ("baseline", 1.00),
    ("perturb_-5%", 0.95),
    ("perturb_-2%", 0.98),
    ("perturb_+2%", 1.02),
    ("perturb_+5%", 1.05),
]

results = []

for name, scale in cases:
    bk = b_k_base * scale
    bl = b_lambda_base * scale

    k_obs = kc_from_bk(bk)
    lam = lambdac_from_bl(bl)
    k_model = kc_pred(lam)

    err = abs(k_model - k_obs)
    rel = err / k_obs

    results.append((name, k_obs, k_model, err, rel))

    print(
        f"{name:>12} | "
        f"k_obs={k_obs:.4f} | "
        f"k_pred={k_model:.4f} | "
        f"err={err:.4f} | "
        f"rel={rel:.4f}"
    )

errs = np.array([r[3] for r in results])
rels = np.array([r[4] for r in results])

print("\n--- Stats ---")
print(f"mean err = {np.mean(errs):.4f}")
print(f"max err  = {np.max(errs):.4f}")
print(f"mean rel = {np.mean(rels):.4f}")

print("\nInterpretation:")
print("If errors stay low across perturbations → law is universal")

print("\nDone.\n")
>>>>>>> d451b2f (lock gv-cst propagation law + duality + residual + validation)
