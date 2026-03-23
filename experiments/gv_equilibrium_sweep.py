# experiments/gv_equilibrium_sweep.py

import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

print("\n🚀 RUNNING UPDATED GV SWEEP (DESATURATED MODE)\n")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_dynamics import GVDynamics


OUT_DIR = os.path.join(REPO_ROOT, "data", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def detect_plateau(history, window=120, gv_tol=0.0025, slope_tol=2e-5):
    if len(history) < window:
        return False, history[-1]["gv"]

    for i in range(window, len(history)):
        chunk = history[i - window : i]
        gvs = [r["gv"] for r in chunk]

        if (max(gvs) - min(gvs)) < gv_tol:
            slope = (gvs[-1] - gvs[0]) / window
            if abs(slope) < slope_tol:
                return True, float(np.mean(gvs))

    return False, history[-1]["gv"]


def run_case(C, S_input, ratio):
    inj = ratio * S_input

    model = GVDynamics(
        beta=1.2,
        gamma=0.45,          # 🔧 FIXED
        entropy_drag=0.12,   # 🔧 FIXED
        collapse_threshold=0.35,
        collapse_rate=2.5,
        collapse_floor=0.0,
        collapse_entropy_drag=0.20,
    )

    model.reset(gv=0.95, entropy=0.02)

    history = model.run(
        steps=3000,
        dt=0.01,
        constraint_fn=lambda i: C,
        injection_fn=lambda i: inj,
        entropy_fn=lambda i: S_input,
    )

    plateau, gv_star = detect_plateau(history)

    final = history[-1]

    return {
        "C": C,
        "ratio": ratio,
        "gv_star": gv_star,
        "final_gv": final["gv"],
        "collapsed": final["collapsed"],
    }


def main():

    S_input = 0.03

    constraints = np.round(np.linspace(0.05, 1.5, 20), 3)
    ratios = np.round(np.linspace(0.05, 2.5, 30), 3)  # 🔥 FIXED RANGE

    print(f"Constraint range: {constraints[0]} → {constraints[-1]}")
    print(f"Ratio range: {ratios[0]} → {ratios[-1]}\n")

    rows = []

    for C in constraints:
        for r in ratios:
            rows.append(run_case(C, S_input, r))

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "gv_equilibrium_sweep.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["C", "ratio", "gv_star", "final_gv", "collapsed"])
        writer.writeheader()
        writer.writerows(rows)

    # Build heatmap
    Z = np.zeros((len(ratios), len(constraints)))

    for i, r in enumerate(ratios):
        for j, C in enumerate(constraints):
            match = next(x for x in rows if x["C"] == C and x["ratio"] == r)
            Z[i, j] = match["gv_star"]

    plt.figure(figsize=(10, 7))
    plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[constraints[0], constraints[-1], ratios[0], ratios[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="GV*")
    plt.xlabel("Constraint C")
    plt.ylabel("Injection / Entropy")
    plt.title("GV Equilibrium (DESATURATED)")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "gv_equilibrium_heatmap.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print("\n✅ DONE")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved heatmap: {fig_path}\n")

    # Quick sanity check print
    sample = sorted(rows, key=lambda x: x["gv_star"])[:10]

    print("Lowest GV* samples:")
    for s in sample:
        print(s)


if __name__ == "__main__":
    main()
