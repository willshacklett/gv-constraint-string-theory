# experiments/gv_equilibrium_sweep.py

import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root:
#   python experiments/gv_equilibrium_sweep.py
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
    """
    Detect a practical plateau (quasi-equilibrium) in GV.

    Returns:
        plateau_found (bool)
        plateau_gv (float)
        plateau_step (int)
    """
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
    gamma: float = 0.65,
    entropy_drag: float = 0.08,
    collapse_threshold: float = 0.35,
    collapse_rate: float = 2.5,
    collapse_floor: float = 0.0,
    collapse_entropy_drag: float = 0.20,
) -> Dict:
    """
    Run one constant-parameter case using the existing GVDynamics engine.

    Since entropy accumulates in the current model, we interpret "equilibrium"
    as a plateau / quasi-equilibrium if the trajectory settles into a narrow band.
    """
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


def save_heatmap(
    matrix: np.ndarray,
    constraints: List[float],
    ratios: List[float],
    path: str,
    *,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    plt.figure(figsize=(10, 7))
    plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[min(constraints), max(constraints), min(ratios), max(ratios)],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="GV plateau")
    plt.xlabel("Constraint C")
    plt.ylabel("Injection / Entropy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_collapse_heatmap(
    rows: List[Dict],
    constraints: List[float],
    ratios: List[float],
    path: str,
) -> None:
    z = np.full((len(ratios), len(constraints)), 0.0, dtype=float)

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
                z[r_idx, c_idx] = 1.0 if match["collapsed"] else 0.0

    plt.figure(figsize=(10, 7))
    plt.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=[min(constraints), max(constraints), min(ratios), max(ratios)],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="Collapsed (1=yes, 0=no)")
    plt.xlabel("Constraint C")
    plt.ylabel("Injection / Entropy")
    plt.title("GV collapse map")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def print_summary(rows: List[Dict]) -> None:
    stable_rows = [r for r in rows if not r["collapsed"]]
    plateau_rows = [r for r in rows if r["plateau_found"] and not r["collapsed"]]

    print("\nGV EQUILIBRIUM / PLATEAU SWEEP\n")
    print(f"Total cases:        {len(rows)}")
    print(f"Non-collapsed:      {len(stable_rows)}")
    print(f"Plateaus detected:  {len(plateau_rows)}")

    if plateau_rows:
        top = sorted(plateau_rows, key=lambda r: r["plateau_gv"], reverse=True)[:10]
        print("\nTop plateau cases:")
        print(
            f"{'C':>6} {'inj/S':>8} {'GV*':>8} {'step':>8} "
            f"{'final_gv':>10} {'max_S':>10}"
        )
        print("-" * 60)
        for row in top:
            print(
                f"{row['constraint']:>6.3f} "
                f"{row['inj_over_entropy']:>8.3f} "
                f"{row['plateau_gv']:>8.4f} "
                f"{row['plateau_step']:>8d} "
                f"{row['final_gv']:>10.4f} "
                f"{row['max_entropy']:>10.4f}"
            )

    if stable_rows:
        nearest = min(stable_rows, key=lambda r: abs(r["plateau_gv"] - 0.962))
        print("\nNearest non-collapsed case to GV = 0.962:")
        print(
            f"  C={nearest['constraint']:.3f}, "
            f"inj/S={nearest['inj_over_entropy']:.3f}, "
            f"plateau_gv={nearest['plateau_gv']:.4f}, "
            f"final_gv={nearest['final_gv']:.4f}, "
            f"collapsed={nearest['collapsed']}"
        )


def main() -> None:
    # Keep entropy_input fixed so we can test the ratio hypothesis cleanly:
    # GV* ~ f(C, injection / entropy_input)
    entropy_input = 0.03

    constraints = np.round(np.linspace(0.05, 1.50, 20), 3).tolist()
    ratios = np.round(np.linspace(0.20, 8.00, 24), 3).tolist()

    rows: List[Dict] = []

    for constraint in constraints:
        for ratio in ratios:
            row = run_case(
                constraint=constraint,
                entropy_input=entropy_input,
                inj_over_entropy=ratio,
                steps=3000,
                dt=0.01,
                init_gv=0.95,
                init_entropy=0.02,
                beta=1.2,
                gamma=0.65,
                entropy_drag=0.08,
                collapse_threshold=0.35,
                collapse_rate=2.5,
                collapse_floor=0.0,
                collapse_entropy_drag=0.20,
            )
            rows.append(row)

    summary_csv = os.path.join(OUT_DIR, "gv_equilibrium_sweep.csv")
    plateau_png = os.path.join(FIG_DIR, "gv_equilibrium_heatmap.png")
    collapse_png = os.path.join(FIG_DIR, "gv_collapse_heatmap.png")

    save_summary_csv(rows, summary_csv)

    plateau_matrix = build_heatmap_matrix(
        rows=rows,
        constraints=constraints,
        ratios=ratios,
        value_key="plateau_gv",
    )
    save_heatmap(
        matrix=plateau_matrix,
        constraints=constraints,
        ratios=ratios,
        path=plateau_png,
        title="GV plateau heatmap vs constraint and injection/entropy",
        vmin=0.0,
        vmax=1.0,
    )

    save_collapse_heatmap(
        rows=rows,
        constraints=constraints,
        ratios=ratios,
        path=collapse_png,
    )

    print_summary(rows)
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved plateau heatmap: {plateau_png}")
    print(f"Saved collapse heatmap: {collapse_png}")


if __name__ == "__main__":
    main()
