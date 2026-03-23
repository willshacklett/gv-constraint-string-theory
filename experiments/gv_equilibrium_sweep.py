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


def detect_plateau(history, window=120, gv_tol=0.0025, slope_tol=2e-5):
    if len(history) < window:
        return False, history[-1]["gv"]

    for i in range(window, len(history)):
        chunk = history[i - window:i]
        gvs = [row["gv"] for row in chunk]

        gv_span = max(gvs) - min(gvs)
        slope = (gvs[-1] - gvs[0]) / max(1, window)

        if gv_span < gv_tol and abs(slope) < slope_tol:
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

    plateau_found, gv_star = detect_plateau(history)
    final = history[-1]

    return {
        "C": float(C),
        "ratio": float(ratio),
        "injection": float(inj),
        "gv_star": float(gv_star),
        "plateau_found": bool(plateau_found),
        "final_gv": float(final["gv"]),
        "final_entropy": float(final["entropy"]),
        "collapsed": bool(final["collapsed"]),
        "regime": final.get("regime", "unknown"),
    }


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "C",
                "ratio",
                "injection",
                "gv_star",
                "plateau_found",
                "final_gv",
                "final_entropy",
                "collapsed",
                "regime",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(rows, constraints, ratios, key="gv_star"):
    Z = np.full((len(ratios), len(constraints)), np.nan)

    for i, ratio in enumerate(ratios):
        for j, C in enumerate(constraints):
            match = next(
                (
                    row for row in rows
                    if abs(row["C"] - float(C)) < 1e-12
                    and abs(row["ratio"] - float(ratio)) < 1e-12
                ),
                None,
            )
            if match is not None:
                Z[i, j] = match[key]

    return Z


def save_heatmap(Z, constraints, ratios, path, title, colorbar_label):
    plt.figure(figsize=(10, 7))
    plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[constraints[0], constraints[-1], ratios[0], ratios[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label=colorbar_label)
    plt.xlabel("Constraint C")
    plt.ylabel("Injection / Entropy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_collapse_map(rows, constraints, ratios, path):
    Z = np.zeros((len(ratios), len(constraints)))

    for i, ratio in enumerate(ratios):
        for j, C in enumerate(constraints):
            match = next(
                (
                    row for row in rows
                    if abs(row["C"] - float(C)) < 1e-12
                    and abs(row["ratio"] - float(ratio)) < 1e-12
                ),
                None,
            )
            if match is not None:
                Z[i, j] = 1.0 if match["collapsed"] else 0.0

    plt.figure(figsize=(10, 7))
    plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[constraints[0], constraints[-1], ratios[0], ratios[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="Collapsed (1=yes, 0=no)")
    plt.xlabel("Constraint C")
    plt.ylabel("Injection / Entropy")
    plt.title("GV Collapse Map (Balanced Regime V2)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def print_summary(rows):
    non_collapsed = [r for r in rows if not r["collapsed"]]
    plateau_rows = [r for r in rows if r["plateau_found"] and not r["collapsed"]]

    print("GV EQUILIBRIUM / PLATEAU SWEEP\n")
    print(f"Total cases:        {len(rows)}")
    print(f"Non-collapsed:      {len(non_collapsed)}")
    print(f"Plateaus detected:  {len(plateau_rows)}")

    if non_collapsed:
        print("\nTop non-collapsed cases by gv_star:")
        print(f"{'C':>6} {'inj/S':>8} {'GV*':>8} {'final_gv':>10} {'regime':>12}")
        print("-" * 52)
        for row in sorted(non_collapsed, key=lambda x: x["gv_star"], reverse=True)[:12]:
            print(
                f"{row['C']:>6.3f} "
                f"{row['ratio']:>8.3f} "
                f"{row['gv_star']:>8.4f} "
                f"{row['final_gv']:>10.4f} "
                f"{row['regime']:>12}"
            )

        nearest = min(non_collapsed, key=lambda x: abs(x["gv_star"] - 0.962))
        print("\nNearest non-collapsed case to GV = 0.962:")
        print(
            f"  C={nearest['C']:.3f}, "
            f"inj/S={nearest['ratio']:.3f}, "
            f"gv_star={nearest['gv_star']:.4f}, "
            f"final_gv={nearest['final_gv']:.4f}, "
            f"regime={nearest['regime']}, "
            f"collapsed={nearest['collapsed']}"
        )

    collapsed = [r for r in rows if r["collapsed"]]
    if collapsed:
        print("\nLowest GV* samples:")
        for row in sorted(collapsed, key=lambda x: x["gv_star"])[:10]:
            print(row)


def main():
    S_input = 0.03

    constraints = np.round(np.linspace(0.05, 1.50, 20), 3)
    ratios = np.round(np.linspace(0.10, 4.00, 30), 3)

    print(f"Constraint range: {constraints[0]} -> {constraints[-1]}")
    print(f"Ratio range: {ratios[0]} -> {ratios[-1]}")
    print("gamma = 0.60")
    print("entropy_drag = 0.09\n")

    rows = []

    for C in constraints:
        for ratio in ratios:
            rows.append(run_case(C, S_input, ratio))

    csv_path = os.path.join(OUT_DIR, "gv_equilibrium_sweep.csv")
    heatmap_path = os.path.join(FIG_DIR, "gv_equilibrium_heatmap.png")
    collapse_path = os.path.join(FIG_DIR, "gv_collapse_heatmap.png")

    save_csv(rows, csv_path)

    Z_gv = build_matrix(rows, constraints, ratios, key="gv_star")
    save_heatmap(
        Z_gv,
        constraints,
        ratios,
        heatmap_path,
        title="GV Equilibrium Heatmap (Balanced Regime V2)",
        colorbar_label="GV*",
    )

    save_collapse_map(rows, constraints, ratios, collapse_path)

    print("✅ DONE\n")
    print_summary(rows)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved collapse map: {collapse_path}\n")


if __name__ == "__main__":
    main()
