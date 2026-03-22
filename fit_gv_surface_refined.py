import numpy as np
import csv
from collections import defaultdict


# alpha(k) from your earlier connectivity fit
def alpha_k(k):
    return 0.9623 - 0.1627 * np.exp(-0.2715 * k)


def load_surface_csv(path="gv_phase_surface.csv"):
    rows = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inj = float(row["inj_min"])
            if np.isnan(inj):
                continue

            rows.append(
                {
                    "C": float(row["C"]),
                    "S": float(row["S"]),
                    "k": float(row["k"]),
                    "inj_min": inj,
                }
            )

    return rows


def fit_slice_for_k(rows_k):
    """
    Fit per-k slice using:

        inj_min(C,S;k) ≈ A(k) * C^{alpha(k)} * exp(beta(k) * S)

    Taking logs:
        log(inj) - alpha(k)*log(C) = log(A) + beta*S

    This is linear in [1, S].
    """
    k_val = rows_k[0]["k"]
    alpha = alpha_k(k_val)

    C = np.array([r["C"] for r in rows_k], dtype=float)
    S = np.array([r["S"] for r in rows_k], dtype=float)
    inj = np.array([r["inj_min"] for r in rows_k], dtype=float)

    mask = (C > 0) & (inj > 0) & np.isfinite(C) & np.isfinite(S) & np.isfinite(inj)
    C = C[mask]
    S = S[mask]
    inj = inj[mask]

    y = np.log(inj) - alpha * np.log(C)

    # linear regression: y = b0 + b1*S
    X = np.column_stack([np.ones_like(S), S])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    b0, beta = coeffs

    A = np.exp(b0)

    pred = A * (C ** alpha) * np.exp(beta * S)

    abs_err = np.abs(pred - inj)
    rel_err = abs_err / inj
    rmse = np.sqrt(np.mean((pred - inj) ** 2))

    # log-space R^2
    y_pred = b0 + beta * S
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "k": k_val,
        "alpha": alpha,
        "A": A,
        "beta": beta,
        "rmse": rmse,
        "mean_rel": float(np.mean(rel_err)),
        "max_rel": float(np.max(rel_err)),
        "r2_log": r2,
        "n": len(inj),
        "C": C,
        "S": S,
        "inj": inj,
        "pred": pred,
    }


def main():
    rows = load_surface_csv("gv_phase_surface.csv")

    by_k = defaultdict(list)
    for row in rows:
        by_k[row["k"]].append(row)

    results = []
    all_true = []
    all_pred = []

    print("\n" + "=" * 96)
    print("REFINED GV SURFACE FIT")
    print("=" * 96)
    print(
        f"{'k':>6} | {'alpha(k)':>10} | {'A(k)':>10} | {'beta(k)':>10} | {'RMSE':>10} | {'MeanRel':>10} | {'MaxRel':>10} | {'R2(log)':>10}"
    )
    print("-" * 96)

    for k in sorted(by_k.keys()):
        fit = fit_slice_for_k(by_k[k])
        results.append(fit)

        all_true.extend(fit["inj"])
        all_pred.extend(fit["pred"])

        print(
            f"{fit['k']:6.1f} | "
            f"{fit['alpha']:10.4f} | "
            f"{fit['A']:10.4f} | "
            f"{fit['beta']:10.4f} | "
            f"{fit['rmse']:10.5f} | "
            f"{fit['mean_rel']:10.4f} | "
            f"{fit['max_rel']:10.4f} | "
            f"{fit['r2_log']:10.4f}"
        )

    all_true = np.array(all_true, dtype=float)
    all_pred = np.array(all_pred, dtype=float)

    all_abs_err = np.abs(all_pred - all_true)
    all_rel_err = all_abs_err / all_true
    global_rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))
    global_mean_rel = float(np.mean(all_rel_err))
    global_max_rel = float(np.max(all_rel_err))

    print("-" * 96)
    print(f"{'GLOBAL':>6} | {'-':>10} | {'-':>10} | {'-':>10} | {global_rmse:10.5f} | {global_mean_rel:10.4f} | {global_max_rel:10.4f} | {'-':>10}")
    print("=" * 96)

    print("\nRefined model by k:")
    print("  inj_min(C,S;k) ≈ A(k) * C^{alpha(k)} * exp(beta(k) * S)")
    print("\nwith alpha(k) fixed from prior fit:")
    print("  alpha(k) = 0.9623 - 0.1627 * exp(-0.2715 * k)")

    print("\nRecovered slice parameters:")
    for fit in results:
        print(
            f"  k={fit['k']:.1f}: "
            f"A≈{fit['A']:.4f}, "
            f"beta≈{fit['beta']:.4f}, "
            f"alpha≈{fit['alpha']:.4f}"
        )


if __name__ == "__main__":
    main()
