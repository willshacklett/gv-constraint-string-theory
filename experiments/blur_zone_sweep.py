import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("NEW BLUR SWEEP FILE RUNNING")

import csv

import numpy as np
import matplotlib.pyplot as plt

from src.noisy_pressure_simulation import NoisyPressureSimulation


OUT_DIR = "data/logs"
os.makedirs(OUT_DIR, exist_ok=True)


def classify_run(history):
    min_eff = min(row["effective_energy"] for row in history)
    final_eff = history[-1]["effective_energy"]
    min_gv = min(row["gv"] for row in history)
    final_gv = history[-1]["gv"]
    max_entropy = max(row["entropy"] for row in history)

    if final_eff < 0.0005 or min_gv < 0.12:
        return "collapse", min_eff, final_eff, min_gv, final_gv, max_entropy

    if (
        0.12 <= min_gv < 0.75
        or 0.45 <= final_gv < 0.90
        or max_entropy > 1.8
    ):
        return "partial", min_eff, final_eff, min_gv, final_gv, max_entropy

    return "damped", min_eff, final_eff, min_gv, final_gv, max_entropy


def ramp_schedule_factory(multiplier=2.5, warmup_steps=120, base_amp=0.10):
    def schedule(i, t):
        if i < warmup_steps:
            frac = i / max(1, warmup_steps)
            return base_amp * (1.0 + (multiplier - 1.0) * frac)
        return base_amp * multiplier
    return schedule


def no_overflow_dim_schedule(i, t):
    return 11


def moderate_injection(i, t):
    return 0.40


def run_single_trial(beta, gamma, noise_std, ramp_steps, seed):
    sim = NoisyPressureSimulation(
        beta=beta,
        gamma=gamma,
        dim_limit=11,
        base_amp=0.10,
        freq=1.0,
        entropy_drag=0.08,
        noise_std=noise_std,
        seed=seed,
    )

    history = sim.run(
        steps=300,
        dt=0.05,
        amp_schedule=ramp_schedule_factory(multiplier=2.5, warmup_steps=ramp_steps),
        dim_schedule=no_overflow_dim_schedule,
        injection_schedule=moderate_injection,
    )

    label, min_eff, final_eff, min_gv, final_gv, max_entropy = classify_run(history)

    return {
        "result": label,
        "min_gv": min_gv,
        "final_gv": final_gv,
        "max_entropy": max_entropy,
        "history": history,
    }


def aggregate_trials(beta, gamma, noise_std, ramp_steps, seeds):
    outcomes = {"damped": 0, "partial": 0, "collapse": 0}
    exemplar = None

    for seed in seeds:
        result = run_single_trial(beta, gamma, noise_std, ramp_steps, seed)
        outcomes[result["result"]] += 1
        if exemplar is None:
            exemplar = result["history"]

    n = len(seeds)

    return {
        "beta": beta,
        "gamma": gamma,
        "noise_std": noise_std,
        "ramp_steps": ramp_steps,
        "damped_pct": outcomes["damped"] / n,
        "partial_pct": outcomes["partial"] / n,
        "collapse_pct": outcomes["collapse"] / n,
        "history": exemplar,
    }


def make_heatmap(rows, metric_key, title, filename):
    noise_vals = sorted(set(row["noise_std"] for row in rows))
    ramp_vals = sorted(set(row["ramp_steps"] for row in rows))

    z = []
    for ramp in ramp_vals:
        z_row = []
        for noise in noise_vals:
            match = next(
                r for r in rows
                if abs(r["noise_std"] - noise) < 1e-6 and r["ramp_steps"] == ramp
            )
            z_row.append(match[metric_key])
        z.append(z_row)

    z = np.array(z)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(z, aspect="auto", origin="lower", vmin=0, vmax=1)

    plt.xticks(range(len(noise_vals)), [f"{v:.2f}" for v in noise_vals])
    plt.yticks(range(len(ramp_vals)), [str(v) for v in ramp_vals])

    plt.xlabel("Noise σ")
    plt.ylabel("Ramp steps (ΔC rate)")
    plt.title(title)

    cbar = plt.colorbar(im)
    cbar.set_label("Fraction")

    for i in range(len(ramp_vals)):
        for j in range(len(noise_vals)):
            val = z[i, j]
            plt.text(
                j,
                i,
                f"{val*100:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val > 0.5 else "black",
            )

    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=180)
    plt.close()

    return path


def main():
    beta = 1.2
    gamma = 0.65

    noise_vals = [0.25, 0.28, 0.30, 0.31, 0.32, 0.35, 0.40]
    ramp_vals = [20, 30, 40, 50, 55, 60, 65, 80]
    seeds = list(range(10))

    summary_rows = []

    print("\nGV-CST BLUR-ZONE SWEEP\n")
    print(f"{'noise':>6} {'ramp':>6} {'damped%':>8} {'partial%':>9} {'collapse%':>10}")
    print("-" * 70)

    for ramp_steps in ramp_vals:
        for noise_std in noise_vals:
            row = aggregate_trials(
                beta=beta,
                gamma=gamma,
                noise_std=noise_std,
                ramp_steps=ramp_steps,
                seeds=seeds,
            )
            summary_rows.append(row)

            print(
                f"{noise_std:>6.2f} {ramp_steps:>6} "
                f"{100*row['damped_pct']:>7.1f}% "
                f"{100*row['partial_pct']:>8.1f}% "
                f"{100*row['collapse_pct']:>9.1f}%"
            )

    summary_csv = os.path.join(OUT_DIR, "blur_zone_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "noise_std",
                "ramp_steps",
                "damped_pct",
                "partial_pct",
                "collapse_pct",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "noise_std": row["noise_std"],
                    "ramp_steps": row["ramp_steps"],
                    "damped_pct": row["damped_pct"],
                    "partial_pct": row["partial_pct"],
                    "collapse_pct": row["collapse_pct"],
                }
            )

    partial_map = make_heatmap(
        summary_rows,
        "partial_pct",
        "GV-CST blur zone | partial-collapse fraction",
        "blur_zone_partial_heatmap.png",
    )

    collapse_map = make_heatmap(
        summary_rows,
        "collapse_pct",
        "GV-CST blur zone | collapse fraction",
        "blur_zone_collapse_heatmap.png",
    )

    damped_map = make_heatmap(
        summary_rows,
        "damped_pct",
        "GV-CST blur zone | damped fraction",
        "blur_zone_damped_heatmap.png",
    )

    print("\nSaved summary:", summary_csv)
    print("Saved partial heatmap:", partial_map)
    print("Saved collapse heatmap:", collapse_map)
    print("Saved damped heatmap:", damped_map)


if __name__ == "__main__":
    main()
