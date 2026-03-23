# experiments/gv_band_width.py

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

print("\n🚀 RUNNING GV BAND-WIDTH ANALYSIS\n")

DATA_PATH = "data/logs/gv_equilibrium_sweep.csv"
FIG_DIR = "figures"
OUT_CSV = "data/logs/gv_band_width_summary.csv"
OUT_PNG = os.path.join(FIG_DIR, "gv_band_width_curve.png")


def load():
    rows = []
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "ratio": float(r["ratio"]),
                    "collapsed": r["collapsed"] == "True",
                    "final_gv": float(r["final_gv"]),
                    "tail_mean_gv": float(r["tail_mean_gv"]),
                }
            )
    return rows


def summarize_by_ratio(rows):
    ratios = sorted(set(r["ratio"] for r in rows))
    out = []

    for ratio in ratios:
        group = [r for r in rows if abs(r["ratio"] - ratio) < 1e-12]
        noncollapsed = [r for r in group if not r["collapsed"]]

        total = len(group)
        noncollapsed_count = len(noncollapsed)

        if noncollapsed_count == 0:
            out.append(
                {
                    "ratio": ratio,
                    "total_count": total,
                    "noncollapsed_count": 0,
                    "survival_fraction": 0.0,
                    "mean_tail_mean_gv": np.nan,
                    "std_tail_mean_gv": np.nan,
                    "min_tail_mean_gv": np.nan,
                    "max_tail_mean_gv": np.nan,
                    "band_width": np.nan,
                    "mean_final_gv": np.nan,
                }
            )
            continue

        tail_vals = np.array([r["tail_mean_gv"] for r in noncollapsed], dtype=float)
        final_vals = np.array([r["final_gv"] for r in noncollapsed], dtype=float)

        out.append(
            {
                "ratio": ratio,
                "total_count": total,
                "noncollapsed_count": noncollapsed_count,
                "survival_fraction": noncollapsed_count / total,
                "mean_tail_mean_gv": float(np.mean(tail_vals)),
                "std_tail_mean_gv": float(np.std(tail_vals)),
                "min_tail_mean_gv": float(np.min(tail_vals)),
                "max_tail_mean_gv": float(np.max(tail_vals)),
                "band_width": float(np.max(tail_vals) - np.min(tail_vals)),
                "mean_final_gv": float(np.mean(final_vals)),
            }
        )

    return out


def save_csv(rows):
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "total_count",
                "noncollapsed_count",
                "survival_fraction",
                "mean_tail_mean_gv",
                "std_tail_mean_gv",
                "min_tail_mean_gv",
                "max_tail_mean_gv",
                "band_width",
                "mean_final_gv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(rows):
    valid = [r for r in rows if not np.isnan(r["band_width"])]
    if not valid:
        return

    x = np.array([r["ratio"] for r in valid], dtype=float)
    width = np.array([r["band_width"] for r in valid], dtype=float)
    std = np.array([r["std_tail_mean_gv"] for r in valid], dtype=float)
    mean_tail = np.array([r["mean_tail_mean_gv"] for r in valid], dtype=float)
    survival = np.array([r["survival_fraction"] for r in valid], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, width, marker="o", label="band_width")
    plt.plot(x, std, marker="s", label="std_tail_mean_gv")
    plt.plot(x, survival, marker="^", label="survival_fraction")
    plt.plot(x, mean_tail, marker="x", label="mean_tail_mean_gv")
    plt.axvline(4.0, linestyle="--", linewidth=1, label="ridge ratio = 4")
    plt.xlabel("Injection / Entropy")
    plt.ylabel("Value")
    plt.title("GV Band Width / Survival / Mean vs Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    plt.close()


def print_summary(rows):
    valid = [r for r in rows if not np.isnan(r["band_width"])]

    print("GV BAND-WIDTH SUMMARY")
    print("---------------------")
    print(f"Ratios with surviving cases: {len(valid)}")

    if not valid:
        print("\nNo surviving ratios found.")
        return

    min_width = min(valid, key=lambda r: r["band_width"])
    min_std = min(valid, key=lambda r: r["std_tail_mean_gv"])
    max_survival = max(valid, key=lambda r: r["survival_fraction"])
    best_center = min(valid, key=lambda r: abs(r["mean_tail_mean_gv"] - 0.9588))

    print("\nNarrowest band:")
    print(
        f"  ratio={min_width['ratio']:.3f}, "
        f"band_width={min_width['band_width']:.4f}, "
        f"std={min_width['std_tail_mean_gv']:.4f}, "
        f"mean_tail={min_width['mean_tail_mean_gv']:.4f}, "
        f"survival_fraction={min_width['survival_fraction']:.4f}"
    )

    print("\nLowest std:")
    print(
        f"  ratio={min_std['ratio']:.3f}, "
        f"std={min_std['std_tail_mean_gv']:.4f}, "
        f"band_width={min_std['band_width']:.4f}, "
        f"mean_tail={min_std['mean_tail_mean_gv']:.4f}, "
        f"survival_fraction={min_std['survival_fraction']:.4f}"
    )

    print("\nHighest survival fraction:")
    print(
        f"  ratio={max_survival['ratio']:.3f}, "
        f"survival_fraction={max_survival['survival_fraction']:.4f}, "
        f"mean_tail={max_survival['mean_tail_mean_gv']:.4f}, "
        f"band_width={max_survival['band_width']:.4f}, "
        f"std={max_survival['std_tail_mean_gv']:.4f}"
    )

    print("\nClosest mean-tail ratio to 0.9588:")
    print(
        f"  ratio={best_center['ratio']:.3f}, "
        f"mean_tail={best_center['mean_tail_mean_gv']:.4f}, "
        f"band_width={best_center['band_width']:.4f}, "
        f"std={best_center['std_tail_mean_gv']:.4f}, "
        f"survival_fraction={best_center['survival_fraction']:.4f}"
    )

    ratio4 = next((r for r in valid if abs(r["ratio"] - 4.0) < 1e-12), None)
    if ratio4:
        print("\nAt ratio = 4:")
        print(
            f"  mean_tail={ratio4['mean_tail_mean_gv']:.4f}, "
            f"band_width={ratio4['band_width']:.4f}, "
            f"std={ratio4['std_tail_mean_gv']:.4f}, "
            f"survival_fraction={ratio4['survival_fraction']:.4f}"
        )


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = load()
    summary = summarize_by_ratio(rows)
    save_csv(summary)
    plot(summary)

    print("✅ DONE\n")
    print_summary(summary)
    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Saved plot: {OUT_PNG}\n")


if __name__ == "__main__":
    main()
