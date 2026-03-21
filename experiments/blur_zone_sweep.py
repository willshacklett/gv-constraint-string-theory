import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print("NEW BLUR SWEEP FILE RUNNING")


# -----------------------------
# Tunable model parameters
# -----------------------------
BETA = 1.2
GAMMA = 0.15

# Push wider/harder so we can actually find a boundary
NOISE_VALS = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.90, 1.10]
RAMP_VALS = [20, 40, 60, 80, 100, 150, 200, 300]
SEEDS = list(range(10))

# Simulation controls
N_STEPS = 220
DT = 0.08

# Classification thresholds
COLLAPSE_THRESHOLD = 4.0       # hard fail if |x| blows past this
PARTIAL_THRESHOLD = 1.25       # "blur zone" if |x| gets above this but recovers
SETTLE_THRESHOLD = 0.60        # if final |x| is below this, call it damped


def ensure_output_dir() -> str:
    out_dir = os.path.join("data", "logs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def smoothstep(z: float) -> float:
    """
    Smooth ramp from 0 to 1 as z goes from 0 to 1.
    """
    z = max(0.0, min(1.0, z))
    return z * z * (3.0 - 2.0 * z)


def ramp_force(step: int, ramp_steps: int) -> float:
    """
    External forcing ramps from 0 to 1 over ramp_steps.
    """
    if ramp_steps <= 0:
        return 1.0
    return smoothstep(step / float(ramp_steps))


def simulate_run(noise_sigma: float, ramp_steps: int, seed: int) -> dict:
    """
    Nonlinear driven/damped system with stochastic forcing.

    x'' + gamma*x' - beta*x + x^3 = forcing + noise

    Interpretation:
    - damping gamma tries to stabilize
    - negative linear term with cubic restores/creates bistable-like behavior
    - forcing ramps the system toward instability
    - noise can kick it across thresholds
    """
    rng = np.random.default_rng(seed)

    x = 0.05
    v = 0.0

    max_abs_x = abs(x)
    crossed_partial = False
    collapsed = False
    collapse_step = None

    series_x = []
    series_v = []

    for step in range(N_STEPS):
        force = ramp_force(step, ramp_steps)
        noise = rng.normal(0.0, noise_sigma)

        # Dynamics
        # x'' = beta*x - x^3 - gamma*v + force + noise
        a = (BETA * x) - (0.2 * x ** 3) - (GAMMA * v) + (2.5 * force) + noise

        v = v + DT * a
        x = x + DT * v

        ax = abs(x)
        max_abs_x = max(max_abs_x, ax)

        if ax >= PARTIAL_THRESHOLD:
            crossed_partial = True

        if ax >= COLLAPSE_THRESHOLD:
            collapsed = True
            collapse_step = step
            series_x.append(x)
            series_v.append(v)
            break

        series_x.append(x)
        series_v.append(v)

    final_abs_x = abs(x)

    if collapsed:
        label = "collapse"
    elif crossed_partial:
        # It crossed into the blur zone but did not fully blow up.
        label = "partial"
    elif final_abs_x <= SETTLE_THRESHOLD:
        label = "damped"
    else:
        # fallback bucket if it neither settled tightly nor crossed partial
        label = "partial"

    return {
        "noise": noise_sigma,
        "ramp": ramp_steps,
        "seed": seed,
        "label": label,
        "max_abs_x": float(max_abs_x),
        "final_abs_x": float(final_abs_x),
        "collapse_step": collapse_step if collapse_step is not None else -1,
        "steps_completed": len(series_x),
    }


def summarize_cell(noise_sigma: float, ramp_steps: int, seeds: list[int]) -> dict:
    results = [simulate_run(noise_sigma, ramp_steps, seed) for seed in seeds]

    total = len(results)
    damped_n = sum(1 for r in results if r["label"] == "damped")
    partial_n = sum(1 for r in results if r["label"] == "partial")
    collapse_n = sum(1 for r in results if r["label"] == "collapse")

    return {
        "noise": noise_sigma,
        "ramp": ramp_steps,
        "damped_n": damped_n,
        "partial_n": partial_n,
        "collapse_n": collapse_n,
        "damped_pct": damped_n / total,
        "partial_pct": partial_n / total,
        "collapse_pct": collapse_n / total,
        "avg_max_abs_x": float(np.mean([r["max_abs_x"] for r in results])),
        "avg_final_abs_x": float(np.mean([r["final_abs_x"] for r in results])),
    }


def make_heatmap(summary_rows: list[dict], value_col: str, title: str, out_path: str) -> str:
    df = pd.DataFrame(summary_rows)

    pivot = df.pivot(index="ramp", columns="noise", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Ramp steps (ΔC rate)")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            txt = f"{val * 100:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fraction")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_summary_table(summary_rows: list[dict]) -> None:
    print("\nGV-CST BLUR-ZONE SWEEP\n")
    print(f"{'noise':<8}{'ramp':<8}{'damped%':<12}{'partial%':<12}{'collapse%':<12}")
    print("-" * 52)

    for row in summary_rows:
        print(
            f"{row['noise']:<8.2f}"
            f"{row['ramp']:<8}"
            f"{row['damped_pct'] * 100:<12.1f}"
            f"{row['partial_pct'] * 100:<12.1f}"
            f"{row['collapse_pct'] * 100:<12.1f}"
        )


def main():
    print(">>> NEW MAIN IS RUNNING <<<")

    out_dir = ensure_output_dir()
    summary_rows = []

    for ramp in RAMP_VALS:
        for noise in NOISE_VALS:
            row = summarize_cell(noise, ramp, SEEDS)
            summary_rows.append(row)

    # Sort nicely for console output and CSV
    summary_rows.sort(key=lambda r: (r["ramp"], r["noise"]))

    print_summary_table(summary_rows)

    summary_csv = os.path.join(out_dir, "blur_zone_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    partial_map = make_heatmap(
        summary_rows,
        "partial_pct",
        "GV-CST blur zone | partial-collapse fraction",
        os.path.join(out_dir, "blur_zone_partial_heatmap.png"),
    )

    collapse_map = make_heatmap(
        summary_rows,
        "collapse_pct",
        "GV-CST blur zone | collapse fraction",
        os.path.join(out_dir, "blur_zone_collapse_heatmap.png"),
    )

    damped_map = make_heatmap(
        summary_rows,
        "damped_pct",
        "GV-CST blur zone | damped fraction",
        os.path.join(out_dir, "blur_zone_damped_heatmap.png"),
    )

    max_amp_map = make_heatmap(
        summary_rows,
        "avg_max_abs_x",
        "GV-CST blur zone | average max |x|",
        os.path.join(out_dir, "blur_zone_maxamp_heatmap.png"),
    )

    print("\nSaved summary:", summary_csv)
    print("Saved partial heatmap:", partial_map)
    print("Saved collapse heatmap:", collapse_map)
    print("Saved damped heatmap:", damped_map)
    print("Saved max amplitude heatmap:", max_amp_map)


if __name__ == "__main__":
    main()
