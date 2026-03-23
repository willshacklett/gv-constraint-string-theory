# experiments/gv_curve_collapse.py

import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

print("\n🚀 RUNNING GV CURVE COLLAPSE TEST\n")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

SWEEP_CSV = os.path.join(DATA_DIR, "gv_equilibrium_sweep.csv")

ALL_FINAL_PNG = os.path.join(FIG_DIR, "gv_curve_collapse_final.png")
ALL_TAILMEAN_PNG = os.path.join(FIG_DIR, "gv_curve_collapse_tailmean.png")
BINNED_FINAL_PNG = os.path.join(FIG_DIR, "gv_curve_collapse_binned_final.png")
BINNED_TAILMEAN_PNG = os.path.join(FIG_DIR, "gv_curve_collapse_binned_tailmean.png")


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


def make_bins(rows, xkey, ykey, nbins=24, noncollapsed_only=False):
    if noncollapsed_only:
        rows = [r for r in rows if not r["collapsed"]]

    xs = np.array([r[xkey] for r in rows], dtype=float)
    ys = np.array([r[ykey] for r in rows], dtype=float)

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    edges = np.linspace(x_min, x_max, nbins + 1)

    centers = []
    means = []
    stds = []
    counts = []

    for i in range(nbins):
        lo = edges[i]
        hi = edges[i + 1]

        if i < nbins - 1:
            mask = (xs >= lo) & (xs < hi)
        else:
            mask = (xs >= lo) & (xs <= hi)

        bucket_y = ys[mask]
        if len(bucket_y) == 0:
            continue

        centers.append((lo + hi) / 2.0)
        means.append(float(np.mean(bucket_y)))
        stds.append(float(np.std(bucket_y)))
        counts.append(int(len(bucket_y)))

    return {
        "centers": np.array(centers),
        "means": np.array(means),
        "stds": np.array(stds),
        "counts": np.array(counts),
    }


def plot_all_by_constraint(rows, ykey, outpath, title, ylabel):
    groups = defaultdict(list)
    for row in rows:
        groups[row["C"]].append(row)

    plt.figure(figsize=(10, 6))

    for C in sorted(groups.keys()):
        group = sorted(groups[C], key=lambda r: r["ratio"])
        x = [r["ratio"] for r in group]
        y = [r[ykey] for r in group]
        plt.plot(x, y, alpha=0.8, linewidth=1)

    plt.axvline(4.0, linestyle="--", linewidth=1)
    plt.xlabel("Injection / Entropy")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_binned(rows, ykey, outpath, title, ylabel, noncollapsed_only=False):
    binned = make_bins(rows, xkey="ratio", ykey=ykey, nbins=24, noncollapsed_only=noncollapsed_only)

    plt.figure(figsize=(10, 6))
    plt.plot(binned["centers"], binned["means"], marker="o")
    if len(binned["centers"]) > 0:
        lower = binned["means"] - binned["stds"]
        upper = binned["means"] + binned["stds"]
        plt.fill_between(binned["centers"], lower, upper, alpha=0.2)

    plt.axvline(4.0, linestyle="--", linewidth=1)
    plt.xlabel("Injection / Entropy")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

    return binned


def print_summary(rows):
    noncollapsed = [r for r in rows if not r["collapsed"]]
    near_ridge = [r for r in noncollapsed if abs(r["ratio"] - 4.0) < 1e-9]

    print("GV CURVE COLLAPSE SUMMARY\n")
    print(f"Total rows:         {len(rows)}")
    print(f"Non-collapsed rows: {len(noncollapsed)}")
    print(f"Rows at ratio=4:    {len(near_ridge)}")

    if near_ridge:
        final_mean = float(np.mean([r["final_gv"] for r in near_ridge]))
        tail_mean = float(np.mean([r["tail_mean_gv"] for r in near_ridge]))
        tail_span_mean = float(np.mean([r["tail_span"] for r in near_ridge]))

        print("\nAt ridge ratio = 4:")
        print(f"  mean(final_gv)     = {final_mean:.4f}")
        print(f"  mean(tail_mean_gv) = {tail_mean:.4f}")
        print(f"  mean(tail_span)    = {tail_span_mean:.4f}")

        nearest = min(near_ridge, key=lambda r: abs(r["tail_mean_gv"] - 0.962))
        print("\nNearest row on ratio=4 ridge to tail_mean_gv = 0.962:")
        print(
            f"  C={nearest['C']:.3f}, "
            f"ratio={nearest['ratio']:.3f}, "
            f"final_gv={nearest['final_gv']:.4f}, "
            f"tail_mean_gv={nearest['tail_mean_gv']:.4f}, "
            f"tail_span={nearest['tail_span']:.4f}"
        )

    # crude collapse check: variance across C at fixed ratio = 4
    by_ratio4 = sorted(near_ridge, key=lambda r: r["C"])
    if by_ratio4:
        y = np.array([r["tail_mean_gv"] for r in by_ratio4], dtype=float)
        print(f"\nSpread along ratio=4 ridge:")
        print(f"  min(tail_mean_gv)  = {np.min(y):.4f}")
        print(f"  max(tail_mean_gv)  = {np.max(y):.4f}")
        print(f"  std(tail_mean_gv)  = {np.std(y):.4f}")


def main():
    if not os.path.exists(SWEEP_CSV):
        raise FileNotFoundError(
            f"Missing sweep CSV: {SWEEP_CSV}\n"
            "Run python experiments/gv_equilibrium_sweep.py first."
        )

    os.makedirs(FIG_DIR, exist_ok=True)

    rows = load_rows(SWEEP_CSV)

    plot_all_by_constraint(
        rows,
        ykey="final_gv",
        outpath=ALL_FINAL_PNG,
        title="GV Curve Collapse Test: final_gv vs injection/entropy",
        ylabel="Final GV",
    )

    plot_all_by_constraint(
        rows,
        ykey="tail_mean_gv",
        outpath=ALL_TAILMEAN_PNG,
        title="GV Curve Collapse Test: tail_mean_gv vs injection/entropy",
        ylabel="Tail Mean GV",
    )

    binned_final = plot_binned(
        rows,
        ykey="final_gv",
        outpath=BINNED_FINAL_PNG,
        title="GV Curve Collapse (binned): final_gv vs injection/entropy",
        ylabel="Final GV",
        noncollapsed_only=False,
    )

    binned_tail = plot_binned(
        rows,
        ykey="tail_mean_gv",
        outpath=BINNED_TAILMEAN_PNG,
        title="GV Curve Collapse (binned): tail_mean_gv vs injection/entropy",
        ylabel="Tail Mean GV",
        noncollapsed_only=False,
    )

    print("✅ DONE\n")
    print_summary(rows)

    if len(binned_tail["centers"]) > 0:
        nearest_idx = int(np.argmin(np.abs(binned_tail["centers"] - 4.0)))
        print("\nBinned curve near ratio = 4:")
        print(
            f"  ratio_bin_center = {binned_tail['centers'][nearest_idx]:.4f}, "
            f"tail_mean_mean = {binned_tail['means'][nearest_idx]:.4f}, "
            f"tail_mean_std = {binned_tail['stds'][nearest_idx]:.4f}, "
            f"count = {binned_tail['counts'][nearest_idx]}"
        )

    print(f"\nSaved all-lines final plot: {ALL_FINAL_PNG}")
    print(f"Saved all-lines tail plot: {ALL_TAILMEAN_PNG}")
    print(f"Saved binned final plot: {BINNED_FINAL_PNG}")
    print(f"Saved binned tail plot: {BINNED_TAILMEAN_PNG}\n")


if __name__ == "__main__":
    main()
