# experiments/gv_equilibrium_sweep.py

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

print("\n🚀 RUNNING UPDATED GV SWEEP (BALANCED REGIME V2)\n")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_dynamics import GVDynamics


OUT_DIR = os.path.join(REPO_ROOT, "data", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def detect_plateau(history, window=120, tol=0.003):
    if len(history) < window:
        return False, history[-1]["gv"]

    for i in range(window, len(history)):
        chunk = history[i - window:i]
        gvs = [r["gv"] for r in chunk]

        if (max(gvs) - min(gvs)) < tol:
            return True, float(np.mean(gvs))

    return False, history[-1]["gv"]


def run_case(C, S_input, ratio):
    inj = ratio * S_input

    model = GVDynamics(
        beta=1.2,
        gamma=0.60,
        entropy_drag=0.09,
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
        "C": float(C),
        "ratio": float(ratio),
        "gv_star": float(gv_star),
        "final_gv": float(final["gv"]),
        "collapsed": bool(final["collapsed"]),
    }


def main():

    S_input = 0.03

    constraints = np.round(np.linspace(0.05, 1.50, 20), 3)

    # 🔥 THIS IS THE CRITICAL LINE
    ratios = np.round(np.linspace(0.10, 4.00, 30), 3)

    print(f"Constraint range: {constraints[0]} → {constraints[-1]}")
    print(f"Ratio range: {ratios[0]} → {ratios[-1]}")
    print("gamma = 0.60")
    print("entropy_drag = 0.09\n")

    rows = []

    for C in constraints:
        for r in ratios:
            rows.append(run_case(C, S_input, r))

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "gv_equilibrium_sweep.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["C", "ratio", "gv_star", "final_gv", "collapsed"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Build heatmap
    Z = np.zeros((len(ratios), len(constraints)))

    for i, r in enumerate(ratios):
        for j, C in enumerate(constraints):
            match = next(x for x in rows if x["C"] == float(C) and x["ratio"] == float(r))
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
    plt.title("GV Equilibrium Heatmap (BALANCED REGIME)")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "gv_equilibrium_heatmap.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print("\n✅ DONE\n")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved heatmap: {fig_path}\n")

    # 🔥 KEY DEBUG OUTPUT
    non_collapsed = [r for r in rows if not r["collapsed"]]

    print(f"Non-collapsed cases: {len(non_collapsed)}")

    if non_collapsed:
        print("\nTop stable cases:")
        for row in sorted(non_collapsed, key=lambda x: x["gv_star"], reverse=True)[:10]:
            print(row)

        nearest = min(non_collapsed, key=lambda x: abs(x["gv_star"] - 0.962))
        print("\n🔥 Closest to 0.962:")
        print(nearest)

    else:
        print("\n⚠️ All cases collapsed — still underpowered")


if __name__ == "__main__":
    main()
