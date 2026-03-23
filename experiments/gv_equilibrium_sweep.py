# experiments/gv_equilibrium_sweep.py

import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_dynamics import GVDynamics  # noqa: E402


OUT_DIR = os.path.join(REPO_ROOT, "data", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def detect_plateau(
    history: List[Dict],
    window: int = 120,
    gv_range_tol: float = 0.0025,
    slope_tol: float = 2.0e-5,
) -> Tuple[bool, float, int]:

    if len(history) < window + 2:
        return False, history[-1]["gv"], history[-1]["step"]

    for i in range(window, len(history)):
        chunk = history[i - window : i]
        gvs = [row["gv"] for row in chunk]

        gv_min = min(gvs)
        gv_max = max(gvs)
        gv_span = gv_max - gv_min

        slope = (gvs[-1] - gvs[0]) / max(1, len(gvs) - 1)

        if gv_span <= gv_range_tol and abs(slope) <= slope_tol:
            return True, float(np.mean(gvs)), history[i]["step"]

    return False, history[-1]["gv"], history[-1]["step"]


def run_case(
    constraint: float,
    entropy_input: float,
    inj_over_entropy: float,
    *,
    steps: int = 3000,
    dt: float = 0.01,
    init_gv: float = 0.95,
    init_entropy: float = 0.02,
    beta: float = 1.2,
    gamma: float = 0.45,            # 🔧 lowered
    entropy_drag: float = 0.12,     # 🔧 increased
    collapse_threshold: float = 0.35,
    collapse_rate: float = 2.5,
    collapse_floor: float = 0.0,
    collapse_entropy_drag: float = 0.20,
) -> Dict:

    injection = inj_over_entropy * entropy_input

    model = GVDynamics(
        beta=beta,
        gamma=gamma,
        entropy_drag=entropy_drag,
        collapse_threshold=collapse_threshold,
        collapse_rate=collapse_rate,
        collapse_floor=collapse_floor,
        collapse_entropy_drag=collapse_entropy_drag,
    )
    model.reset(gv=init_gv, entropy=init_entropy)

    history = model.run(
        steps=steps,
        dt=dt,
        constraint_fn=lambda i: constraint,
        injection_fn=lambda i: injection,
        entropy_fn=lambda i: entropy_input,
    )

    plateau_found, plateau_gv, plateau_step = detect_plateau(history)

    final = history[-1]
    min_gv = min(row["gv"] for row in history)
    max_entropy = max(row["entropy"] for row in history)

    return {
        "constraint": constraint,
        "entropy_input": entropy_input,
        "injection": injection,
        "inj_over_entropy": inj_over_entropy,
        "plateau_found": plateau_found,
        "plateau_gv": plateau_gv,
        "plateau_step": plateau_step,
        "final_gv": final["gv"],
        "final_entropy": final["entropy"],
        "final_regime": final["regime"],
        "collapsed": final["collapsed"],
        "min_gv": min_gv,
        "max_entropy": max_entropy,
        "history": history,
    }


def save_summary_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "constraint",
                "entropy_input",
                "injection",
                "inj_over_entropy",
                "plateau_found",
                "plateau_gv",
                "plateau_step",
                "final_gv",
                "final_entropy",
                "final_regime",
                "collapsed",
                "min_gv",
                "max_entropy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "constraint": f"{row['constraint']:.6f}",
                    "entropy_input": f"{row['entropy_input']:.6f}",
                    "injection": f"{row['injection']:.6f}",
                    "inj_over_entropy": f"{row['inj_over_entropy']:.6f}",
                    "plateau_found": row["plateau_found"],
                    "plateau_gv": f"{row['plateau_gv']:.6f}",
                    "plateau_step": row["plateau_step"],
                    "final_gv": f"{row['final_gv']:.6f}",
                    "final_entropy": f"{row['final_entropy']:.6f}",
                    "final_regime": row["final_regime"],
                    "collapsed": row["collapsed"],
                    "min_gv": f"{row['min_gv']:.6f}",
                    "max_entropy": f"{row['max_entropy']:.6f}",
                }
            )


def build_heatmap_matrix(
    rows: List[Dict],
    constraints: List[float],
    ratios: List[float],
    value_key: str = "plateau_gv",
) -> np.ndarray:

    z = np.full((len(ratios), len(constraints)), np.nan, dtype=float)

    for r_idx, ratio in enumerate(ratios):
        for c_idx, constraint in enumerate(constraints):
            match = next(
                (
                    row
                    for row in rows
                    if abs(row["constraint"] - constraint) < 1e-12
                    and abs(row["inj_over_entropy"] - ratio) < 1e-12
                ),
                None,
            )
            if match is not None:
                z[r_idx, c_idx] = match[value_key]

    return z


def save_heatmap(matrix, constraints, ratios, path, title):
    plt.figure(figsize=(10, 7))
    plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[min(constraints), max(constraints), min(ratios), max(ratios)],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="GV plateau")
    plt.xlabel("Constraint C")
    plt.ylabel("Injection / Entropy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def print_summary(rows: List[Dict]) -> None:
    plateau_rows = [r for r in rows if r["plateau_found"] and not r["collapsed"]]

    print("\nGV EQUILIBRIUM / PLATEAU SWEEP\n")
    print(f"Total cases:        {len(rows)}")
    print(f"Plateaus detected:  {len(plateau_rows)}")

    if plateau_rows:
        top = sorted(plateau_rows, key=lambda r: r["plateau_gv"], reverse=True)[:10]
        print("\nTop plateau cases:")
        print(f"{'C':>6} {'inj/S':>8} {'GV*':>8}")
        print("-" * 30)
        for row in top:
            print(
                f"{row['constraint']:>6.3f} "
                f"{row['inj_over_entropy']:>8.3f} "
                f"{row['plateau_gv']:>8.4f}"
            )


def main() -> None:

    entropy_input = 0.03

    constraints = np.round(np.linspace(0.05, 1.50, 20), 3).tolist()

    # 🔥 FIXED RANGE
    ratios = np.round(np.linspace(0.05, 2.5, 30), 3).tolist()

    rows: List[Dict] = []

    for constraint in constraints:
        for ratio in ratios:
            row = run_case(
                constraint=constraint,
                entropy_input=entropy_input,
                inj_over_entropy=ratio,
            )
            rows.append(row)

    summary_csv = os.path.join(OUT_DIR, "gv_equilibrium_sweep.csv")
    heatmap_png = os.path.join(FIG_DIR, "gv_equilibrium_heatmap.png")

    save_summary_csv(rows, summary_csv)

    matrix = build_heatmap_matrix(rows, constraints, ratios)

    save_heatmap(
        matrix,
        constraints,
        ratios,
        heatmap_png,
        "GV equilibrium (desaturated regime)",
    )

    print_summary(rows)

    print(f"\nSaved CSV: {summary_csv}")
    print(f"Saved heatmap: {heatmap_png}")


if __name__ == "__main__":
    main()
