# experiments/gv_overlay_stats.py

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

print("\n🚀 RUNNING GV OVERLAY STATS\n")

DATA_PATH = "data/logs/gv_equilibrium_sweep.csv"
OUT_CSV = "data/logs/gv_overlay_stats_summary.csv"
OUT_PNG = "figures/gv_overlay_stats.png"


def load():
    rows = []
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "ratio": float(r["ratio"]),
                    "collapsed": r["collapsed"] == "True",
                    "tail_mean_gv": float(r["tail_mean_gv"]),
                    "final_gv": float(r["final_gv"]),
                }
            )
    return rows


def summarize(rows):
    ratios = sorted(set(r["ratio"] for r in rows))
    out = []

    for ratio in ratios:
        group = [r for r in rows if abs(r["ratio"] - ratio) < 1e-12]
        noncollapsed = [r for r in group if not r["collapsed"]]

        total = len(group)
        survivors = len(noncollapsed)
        survival_fraction = survivors / total if total else 0.0

        if survivors == 0:
            out.append(
                {
                    "ratio": ratio,
                    "total_count": total,
                    "survivor_count": 0,
                    "survival_fraction": survival_fraction,
                    "mean_tail_mean_gv": np.nan,
                    "std_tail_mean_gv": np.nan,
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
                "survivor_count": survivors,
                "survival_fraction": float(survival_fraction),
                "mean_tail_mean_gv": float(np.mean(tail_vals)),
                "std_tail_mean_gv": float(np.std(tail_vals)),
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
                "survivor_count",
                "survival_fraction",
                "mean_tail_mean_gv",
                "std_tail_mean_gv",
                "band_width",
                "mean_final_gv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(rows):
    valid = [r for r in rows if not np.isnan(r["mean_tail_mean_gv"])]
    if not valid:
        return

    x = np.array([r["ratio"] for r in valid], dtype=float)
    survival = np.array([r["survival_fraction"] for r in valid], dtype=float)
    mean_tail = np.array([r["mean_tail_mean_gv"] for r in valid], dtype=float)
    std_tail = np.array([r["std_tail_mean_gv"] for r in valid], dtype=float)
    band_width = np.array([r["band_width"] for r in valid], dtype=float)

    plt.figure(figsize=(11, 7))
    plt.plot(x, survival, marker="o", label="survival_fraction")
    plt.plot(x, mean_tail, marker="s", label="mean_tail_mean_gv")
    plt.plot(x, std_tail, marker="^", label="std_tail_mean_gv")
    plt.plot(x, band_width, marker="x", label="band_width")
    plt.axvline(4.0, linestyle="--", linewidth=1, label="ridge ratio = 4")
    plt.xlabel("Injection / Entropy")
    plt.ylabel("Value")
    plt.title("GV Overlay Stats vs Injection / Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    plt.close()


def print_summary(rows):
    valid = [r for r in rows if not np.isnan(r["mean_tail_mean_gv"])]

    print("GV OVERLAY SUMMARY")
    print("------------------")
    print(f"Ratios with survivors: {len(valid)}")

    if not valid:
        print("\nNo valid survivor ratios.")
        return

    best_survival = max(valid, key=lambda r: r["survival_fraction"])
    highest_mean = max(valid, key=lambda r: r["mean_tail_mean_gv"])

    qualified = [r for r in valid if r["survivor_count"] >= 3]
    if qualified:
        lowest_std = min(qualified, key=lambda r: r["std_tail_mean_gv"])
        lowest_width = min(qualified, key=lambda r: r["band_width"])
    else:
        lowest_std = min(valid, key=lambda r: r["std_tail_mean_gv"])
        lowest_width = min(valid, key=lambda r: r["band_width"])

    ratio4 = next((r for r in valid if abs(r["ratio"] - 4.0) < 1e-12), None)

    print("\nHighest survival:")
    print(best_survival)

    print("\nHighest mean-tail:")
    print(highest_mean)

    print("\nLowest std (survivor_count >= 3):")
    print(lowest_std)

    print("\nLowest width (survivor_count >= 3):")
    print(lowest_width)

    if ratio4:
        print("\nAt ratio = 4:")
        print(ratio4)


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    rows = load()
    summary = summarize(rows)
    save_csv(summary)
    plot(summary)

    print("✅ DONE\n")
    print_summary(summary)
    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Saved plot: {OUT_PNG}\n")


if __name__ == "__main__":
    main()
