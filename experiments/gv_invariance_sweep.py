cat > experiments/gv_invariance_sweep.py <<'PY'
# experiments/gv_invariance_sweep.py

import math
import os
import sys
import numpy as np

print("\n🚀 RUNNING GV INVARIANCE SWEEP\n")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_dynamics import GVDynamics


OUT_DIR = os.path.join(REPO_ROOT, "data", "logs")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "gv_invariance_sweep.csv")


def run_case(C, ratio, gamma, entropy_drag, S_input=0.03, steps=3000, dt=0.01):
    inj = ratio * S_input

    model = GVDynamics(
        beta=1.2,
        gamma=gamma,
        entropy_drag=entropy_drag,
        collapse_threshold=0.35,
        collapse_rate=2.5,
        collapse_floor=0.0,
        collapse_entropy_drag=0.20,
    )

    model.reset(gv=0.95, entropy=0.02)

    history = model.run(
        steps=steps,
        dt=dt,
        constraint_fn=lambda i: C,
        injection_fn=lambda i: inj,
        entropy_fn=lambda i: S_input,
    )

    final = history[-1]
    tail = history[-200:] if len(history) >= 200 else history
    tail_vals = np.array([row["gv"] for row in tail], dtype=float)

    return {
        "collapsed": bool(final["collapsed"]),
        "final_gv": float(final["gv"]),
        "tail_mean_gv": float(np.mean(tail_vals)),
        "tail_std_gv": float(np.std(tail_vals)),
        "tail_width_gv": float(np.max(tail_vals) - np.min(tail_vals)),
    }


def summarize_parameter_pair(gamma, entropy_drag):
    constraints = np.round(np.linspace(0.05, 1.50, 20), 3)
    ratios = np.round(np.linspace(0.10, 4.00, 30), 3)

    rows = []

    for C in constraints:
        for ratio in ratios:
            result = run_case(
                C=float(C),
                ratio=float(ratio),
                gamma=float(gamma),
                entropy_drag=float(entropy_drag),
            )
            rows.append(
                {
                    "C": float(C),
                    "ratio": float(ratio),
                    "gamma": float(gamma),
                    "entropy_drag": float(entropy_drag),
                    **result,
                }
            )

    ratio_summaries = []

    for ratio in ratios:
        group = [r for r in rows if abs(r["ratio"] - float(ratio)) < 1e-12]
        survivors = [r for r in group if not r["collapsed"]]

        if not group:
            continue

        survival_fraction = len(survivors) / len(group)

        if len(survivors) == 0:
            ratio_summaries.append(
                {
                    "ratio": float(ratio),
                    "survival_fraction": 0.0,
                    "mean_tail_mean_gv": math.nan,
                    "std_tail_mean_gv": math.nan,
                    "band_width": math.nan,
                    "survivor_count": 0,
                }
            )
            continue

        tail_means = np.array([r["tail_mean_gv"] for r in survivors], dtype=float)

        ratio_summaries.append(
            {
                "ratio": float(ratio),
                "survival_fraction": float(survival_fraction),
                "mean_tail_mean_gv": float(np.mean(tail_means)),
                "std_tail_mean_gv": float(np.std(tail_means)),
                "band_width": float(np.max(tail_means) - np.min(tail_means)),
                "survivor_count": len(survivors),
            }
        )

    valid = [r for r in ratio_summaries if r["survivor_count"] > 0]
    if not valid:
        return {
            "gamma": float(gamma),
            "entropy_drag": float(entropy_drag),
            "best_ratio": math.nan,
            "best_survival_fraction": 0.0,
            "band_center": math.nan,
            "band_std": math.nan,
            "band_width": math.nan,
            "survivor_count": 0,
            "status": "all_collapsed",
        }

    best = max(valid, key=lambda r: (r["survival_fraction"], r["mean_tail_mean_gv"]))

    return {
        "gamma": float(gamma),
        "entropy_drag": float(entropy_drag),
        "best_ratio": float(best["ratio"]),
        "best_survival_fraction": float(best["survival_fraction"]),
        "band_center": float(best["mean_tail_mean_gv"]),
        "band_std": float(best["std_tail_mean_gv"]),
        "band_width": float(best["band_width"]),
        "survivor_count": int(best["survivor_count"]),
        "status": "ok",
    }


def save_csv(rows, path):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gamma",
                "entropy_drag",
                "best_ratio",
                "best_survival_fraction",
                "band_center",
                "band_std",
                "band_width",
                "survivor_count",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    valid = [r for r in rows if r["status"] == "ok"]

    print("GV INVARIANCE SWEEP SUMMARY")
    print("---------------------------")
    print(f"Total parameter pairs: {len(rows)}")
    print(f"Valid pairs:           {len(valid)}")

    if not valid:
        print("\nNo valid parameter pairs survived.")
        return

    print("\nTop parameter pairs by survival:")
    print(
        f"{'gamma':>8} {'drag':>8} {'ratio*':>8} {'survival':>10} "
        f"{'center':>10} {'std':>10} {'width':>10} {'count':>8}"
    )
    print("-" * 80)

    for row in sorted(valid, key=lambda x: (x["best_survival_fraction"], x["band_center"]), reverse=True)[:12]:
        print(
            f"{row['gamma']:>8.3f} "
            f"{row['entropy_drag']:>8.3f} "
            f"{row['best_ratio']:>8.3f} "
            f"{row['best_survival_fraction']:>10.4f} "
            f"{row['band_center']:>10.4f} "
            f"{row['band_std']:>10.4f} "
            f"{row['band_width']:>10.4f} "
            f"{row['survivor_count']:>8d}"
        )

    ratio4_like = [r for r in valid if abs(r["best_ratio"] - 4.0) < 1e-9]
    if ratio4_like:
        centers = np.array([r["band_center"] for r in ratio4_like], dtype=float)
        stds = np.array([r["band_std"] for r in ratio4_like], dtype=float)
        widths = np.array([r["band_width"] for r in ratio4_like], dtype=float)

        print("\nAcross parameter pairs with best_ratio = 4:")
        print(f"  count            = {len(ratio4_like)}")
        print(f"  mean(center)     = {np.mean(centers):.4f}")
        print(f"  std(center)      = {np.std(centers):.4f}")
        print(f"  mean(band_std)   = {np.mean(stds):.4f}")
        print(f"  mean(band_width) = {np.mean(widths):.4f}")

    closest_center = min(valid, key=lambda r: abs(r["band_center"] - 0.9588))
    print("\nClosest parameter pair to center = 0.9588:")
    print(
        f"  gamma={closest_center['gamma']:.3f}, "
        f"drag={closest_center['entropy_drag']:.3f}, "
        f"ratio*={closest_center['best_ratio']:.3f}, "
        f"center={closest_center['band_center']:.4f}, "
        f"std={closest_center['band_std']:.4f}, "
        f"width={closest_center['band_width']:.4f}, "
        f"survival={closest_center['best_survival_fraction']:.4f}"
    )


def main():
    gamma_values = [0.50, 0.60, 0.70]
    entropy_drag_values = [0.07, 0.09, 0.11]

    rows = []

    for gamma in gamma_values:
        for entropy_drag in entropy_drag_values:
            print(f"Running gamma={gamma:.3f}, entropy_drag={entropy_drag:.3f} ...")
            rows.append(summarize_parameter_pair(gamma, entropy_drag))

    save_csv(rows, OUT_CSV)

    print("\n✅ DONE\n")
    print_summary(rows)
    print(f"\nSaved CSV: {OUT_CSV}\n")


if __name__ == "__main__":
    main()
PY
