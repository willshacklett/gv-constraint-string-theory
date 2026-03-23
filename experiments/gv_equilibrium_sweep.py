# experiments/gv_equilibrium_sweep.py

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

print("\n🚀 RUNNING UPDATED GV SWEEP (TAIL-AVERAGE V3)\n")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_dynamics import GVDynamics


OUT_DIR = os.path.join(REPO_ROOT, "data", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def analyze_tail(history, tail_window=200, stability_tol=0.01):
    if len(history) < tail_window:
        tail = history
    else:
        tail = history[-tail_window:]

    gvs = [row["gv"] for row in tail]

    tail_mean = float(np.mean(gvs))
    tail_min = float(np.min(gvs))
    tail_max = float(np.max(gvs))
    tail_span = float(tail_max - tail_min)
    tail_stable = tail_span <= stability_tol

    return {
        "tail_mean_gv": tail_mean,
        "tail_min_gv": tail_min,
        "tail_max_gv": tail_max,
        "tail_span": tail_span,
        "tail_stable": tail_stable,
    }


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

    final = history[-1]
    tail = analyze_tail(history, tail_window=200, stability_tol=0.01)

    return {
        "C": float(C),
        "ratio": float(ratio),
        "injection": float(inj),
        "final_gv": float(final["gv"]),
        "final_entropy": float(final["entropy"]),
        "collapsed": bool(final["collapsed"]),
        "regime": final.get("regime", "unknown"),
        "tail_mean_gv": tail["tail_mean_gv"],
        "tail_min_gv": tail["tail_min_gv"],
        "tail_max_gv": tail["tail_max_gv"],
        "tail_span": tail["tail_span"],
        "tail_stable": bool(tail["tail_stable"]),
    }


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "C",
                "ratio",
                "injection",
                "final_gv",
                "final_entropy",
                "collapsed",
                "regime",
                "tail_mean_gv",
                "tail_min_gv",
                "tail_max_gv",
                "tail_span",
                "tail_stable",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(rows, constraints, ratios, key):
    Z = np.full((len(ratios), len(constraints)), np.nan)

    for i, ratio in enumerate(ratios):
        for j, C in enumerate(constraints):
            match = next(
                (
                    row
                    for row in rows
                    if abs(row["C"] - float(C)) < 1e-12
                    and abs(row["ratio"] - float(ratio)) < 1e-12
                ),
                None,
            )
            if match is not None:
                Z[i, j] = match[key]

    return Z


def save_heatmap(Z, constraints, ratios, path, title, colorbar_label, vmin=0.0, vmax=1.0):
    plt.figure(figsize=(10, 7))
    plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[constraints[0], constraints[-1], ratios[0], ratios[-1]],
        vmin=vmin,
        vmax=vmax,
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
                    row
                    for row in rows
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
    plt.title("GV Collapse Map (TAIL-AVERAGE V3)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def print_summary(rows):
    non_collapsed = [r for r in rows if not r["collapsed"]]
    stable_tail = [r for r in non_collapsed if r["tail_stable"]]

    print("GV EQUILIBRIUM / TAIL-AVERAGE SWEEP\n")
    print(f"Total cases:             {len(rows)}")
    print(f"Non-collapsed:           {len(non_collapsed)}")
    print(f"Tail-stable non-collapsed: {len(stable_tail)}")

    if stable_tail:
        print("\nTop tail-stable cases by tail_mean_gv:")
        print(
            f"{'C':>6} {'inj/S':>8} {'tail_mean':>10} {'final_gv':>10} "
            f"{'tail_span':>10}"
        )
        print("-" * 56)

        for row in sorted(stable_tail, key=lambda x: x["tail_mean_gv"], reverse=True)[:12]:
            print(
                f"{row['C']:>6.3f} "
                f"{row['ratio']:>8.3f} "
                f"{row['tail_mean_gv']:>10.4f} "
                f"{row['final_gv']:>10.4f} "
                f"{row['tail_span']:>10.4f}"
            )

        nearest = min(stable_tail, key=lambda x: abs(x["tail_mean_gv"] - 0.962))
        print("\n🔥 Closest tail-stable case to 0.962:")
        print(
            f"  C={nearest['C']:.3f}, "
            f"inj/S={nearest['ratio']:.3f}, "
            f"tail_mean_gv={nearest['tail_mean_gv']:.4f}, "
            f"final_gv={nearest['final_gv']:.4f}, "
            f"tail_span={nearest['tail_span']:.4f}, "
            f"collapsed={nearest['collapsed']}"
        )

    elif non_collapsed:
        print("\nNo tail-stable cases yet. Showing top non-collapsed by final_gv:")
        for row in sorted(non_collapsed, key=lambda x: x["final_gv"], reverse=True)[:12]:
            print(row)
    else:
        print("\n⚠️ All cases collapsed.")

    print("\nLowest final_gv samples:")
    for row in sorted(rows, key=lambda x: x["final_gv"])[:10]:
        print(row)


def main():
    S_input = 0.03

    constraints = np.round(np.linspace(0.05, 1.50, 20), 3)
    ratios = np.round(np.linspace(0.10, 4.00, 30), 3)

    print(f"Constraint range: {constraints[0]} → {constraints[-1]}")
    print(f"Ratio range: {ratios[0]} → {ratios[-1]}")
    print("gamma = 0.60")
    print("entropy_drag = 0.09")
    print("tail_window = 200")
    print("stability_tol = 0.01\n")

    rows = []

    for C in constraints:
        for ratio in ratios:
            rows.append(run_case(C, S_input, ratio))

    csv_path = os.path.join(OUT_DIR, "gv_equilibrium_sweep.csv")
    heatmap_tail_mean_path = os.path.join(FIG_DIR, "gv_equilibrium_heatmap.png")
    heatmap_final_path = os.path.join(FIG_DIR, "gv_final_heatmap.png")
    collapse_map_path = os.path.join(FIG_DIR, "gv_collapse_heatmap.png")

    save_csv(rows, csv_path)

    Z_tail = build_matrix(rows, constraints, ratios, key="tail_mean_gv")
    Z_final = build_matrix(rows, constraints, ratios, key="final_gv")

    save_heatmap(
        Z_tail,
        constraints,
        ratios,
        heatmap_tail_mean_path,
        title="GV Tail-Mean Heatmap (TAIL-AVERAGE V3)",
        colorbar_label="Tail Mean GV",
    )

    save_heatmap(
        Z_final,
        constraints,
        ratios,
        heatmap_final_path,
        title="GV Final-State Heatmap (TAIL-AVERAGE V3)",
        colorbar_label="Final GV",
    )

    save_collapse_map(rows, constraints, ratios, collapse_map_path)

    print("✅ DONE\n")
    print_summary(rows)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved tail-mean heatmap: {heatmap_tail_mean_path}")
    print(f"Saved final-state heatmap: {heatmap_final_path}")
    print(f"Saved collapse map: {collapse_map_path}\n")


if __name__ == "__main__":
    main()
