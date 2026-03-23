# experiments/gv_ridge_extractor.py

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

print("\n🚀 RUNNING GV RIDGE EXTRACTOR\n")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SWEEP_CSV = os.path.join(DATA_DIR, "gv_equilibrium_sweep.csv")
RIDGE_CSV = os.path.join(DATA_DIR, "gv_ridge_points.csv")
RIDGE_PNG = os.path.join(FIG_DIR, "gv_ridge_curve.png")
RIDGE_SCORE_PNG = os.path.join(FIG_DIR, "gv_ridge_score_curve.png")


def load_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "C": float(row["C"]),
                    "ratio": float(row["ratio"]),
                    "injection": float(row["injection"]),
                    "final_gv": float(row["final_gv"]),
                    "final_entropy": float(row["final_entropy"]),
                    "collapsed": str(row["collapsed"]).lower() == "true",
                    "regime": row["regime"],
                    "tail_mean_gv": float(row["tail_mean_gv"]),
                    "tail_min_gv": float(row["tail_min_gv"]),
                    "tail_max_gv": float(row["tail_max_gv"]),
                    "tail_span": float(row["tail_span"]),
                    "tail_stable": str(row["tail_stable"]).lower() == "true",
                }
            )
    return rows


def ridge_score(row, span_weight=2.5, collapse_penalty=10.0):
    """
    Higher is better.

    Rewards:
    - high tail_mean_gv
    - high final_gv

    Penalizes:
    - large oscillation band (tail_span)
    - collapse
    """
    score = row["tail_mean_gv"] + 0.35 * row["final_gv"] - span_weight * row["tail_span"]
    if row["collapsed"]:
        score -= collapse_penalty
    return score


def choose_ridge_points(rows):
    constraints = sorted(set(row["C"] for row in rows))
    ridge_rows = []

    for C in constraints:
        candidates = [row for row in rows if row["C"] == C and not row["collapsed"]]

        if not candidates:
            ridge_rows.append(
                {
                    "C": C,
                    "ratio": np.nan,
                    "injection": np.nan,
                    "final_gv": np.nan,
                    "tail_mean_gv": np.nan,
                    "tail_span": np.nan,
                    "tail_stable": False,
                    "ridge_score": np.nan,
                    "status": "no_noncollapsed_case",
                }
            )
            continue

        for row in candidates:
            row["ridge_score"] = ridge_score(row)

        best = max(candidates, key=lambda r: r["ridge_score"])

        ridge_rows.append(
            {
                "C": best["C"],
                "ratio": best["ratio"],
                "injection": best["injection"],
                "final_gv": best["final_gv"],
                "tail_mean_gv": best["tail_mean_gv"],
                "tail_span": best["tail_span"],
                "tail_stable": best["tail_stable"],
                "ridge_score": best["ridge_score"],
                "status": "ok",
            }
        )

    return ridge_rows


def save_ridge_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "C",
                "ratio",
                "injection",
                "final_gv",
                "tail_mean_gv",
                "tail_span",
                "tail_stable",
                "ridge_score",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_ridge_curve(rows, path):
    valid = [r for r in rows if r["status"] == "ok" and not np.isnan(r["ratio"])]
    if not valid:
        return

    x = [r["C"] for r in valid]
    y = [r["ratio"] for r in valid]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel("Constraint C")
    plt.ylabel("Ridge ratio (Injection / Entropy)")
    plt.title("GV Ridge Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ridge_score_curve(rows, path):
    valid = [r for r in rows if r["status"] == "ok" and not np.isnan(r["ridge_score"])]
    if not valid:
        return

    x = [r["C"] for r in valid]
    y1 = [r["tail_mean_gv"] for r in valid]
    y2 = [r["final_gv"] for r in valid]
    y3 = [r["tail_span"] for r in valid]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker="o", label="tail_mean_gv")
    plt.plot(x, y2, marker="s", label="final_gv")
    plt.plot(x, y3, marker="^", label="tail_span")
    plt.xlabel("Constraint C")
    plt.ylabel("Value")
    plt.title("GV Ridge Diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def print_summary(rows):
    valid = [r for r in rows if r["status"] == "ok" and not np.isnan(r["ratio"])]

    print("GV RIDGE EXTRACTION SUMMARY\n")
    print(f"Total constraint slices: {len(rows)}")
    print(f"Valid ridge points:      {len(valid)}")

    if not valid:
        print("\nNo valid ridge points found.")
        return

    print("\nTop ridge points by score:")
    print(
        f"{'C':>6} {'ratio*':>8} {'tail_mean':>10} {'final_gv':>10} "
        f"{'tail_span':>10} {'score':>10}"
    )
    print("-" * 64)

    for row in sorted(valid, key=lambda x: x["ridge_score"], reverse=True)[:12]:
        print(
            f"{row['C']:>6.3f} "
            f"{row['ratio']:>8.3f} "
            f"{row['tail_mean_gv']:>10.4f} "
            f"{row['final_gv']:>10.4f} "
            f"{row['tail_span']:>10.4f} "
            f"{row['ridge_score']:>10.4f}"
        )

    nearest_0962 = min(valid, key=lambda x: abs(x["tail_mean_gv"] - 0.962))
    print("\n🔥 Ridge point nearest tail_mean_gv = 0.962:")
    print(
        f"  C={nearest_0962['C']:.3f}, "
        f"ratio*={nearest_0962['ratio']:.3f}, "
        f"tail_mean_gv={nearest_0962['tail_mean_gv']:.4f}, "
        f"final_gv={nearest_0962['final_gv']:.4f}, "
        f"tail_span={nearest_0962['tail_span']:.4f}, "
        f"score={nearest_0962['ridge_score']:.4f}"
    )

    best_band_center = float(np.mean([r["tail_mean_gv"] for r in valid]))
    print(f"\nBand-center estimate from ridge points: {best_band_center:.4f}")


def main():
    if not os.path.exists(SWEEP_CSV):
        raise FileNotFoundError(
            f"Could not find sweep CSV at: {SWEEP_CSV}\n"
            "Run python experiments/gv_equilibrium_sweep.py first."
        )

    rows = load_rows(SWEEP_CSV)
    ridge_rows = choose_ridge_points(rows)

    save_ridge_csv(ridge_rows, RIDGE_CSV)
    plot_ridge_curve(ridge_rows, RIDGE_PNG)
    plot_ridge_score_curve(ridge_rows, RIDGE_SCORE_PNG)

    print("✅ DONE\n")
    print_summary(ridge_rows)
    print(f"\nSaved ridge CSV: {RIDGE_CSV}")
    print(f"Saved ridge curve: {RIDGE_PNG}")
    print(f"Saved ridge diagnostics: {RIDGE_SCORE_PNG}\n")


if __name__ == "__main__":
    main()
