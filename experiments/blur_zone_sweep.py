import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("NEW BLUR SWEEP FILE RUNNING v3")

from src.noisy_pressure_simulation import NoisyPressureSimulation


OUT_DIR = "data/logs"
os.makedirs(OUT_DIR, exist_ok=True)


def classify_run(history):
    min_eff = min(row["effective_energy"] for row in history)
    final_eff = history[-1]["effective_energy"]
    min_gv = min(row["gv"] for row in history)
    final_gv = history[-1]["gv"]
    max_entropy = max(row["entropy"] for row in history)
    max_overflow = max(row["overflow"] for row in history)
    min_constraint = min(row["constraint_noisy"] for row in history)

    if min_gv < 0.18 or final_gv < 0.22 or min_eff < 0.0002:
        return "collapse", min_eff, final_eff, min_gv, final_gv, max_entropy, max_overflow, min_constraint

    if (
        min_gv < 0.72
        or final_gv < 0.82
        or max_entropy > 1.35
        or max_overflow >= 2
        or min_constraint < 0.55
    ):
        return "partial", min_eff, final_eff, min_gv, final_gv, max_entropy, max_overflow, min_constraint

    return "damped", min_eff, final_eff, min_gv, final_gv, max_entropy, max_overflow, min_constraint


def ramp_schedule_factory(multiplier=5.0, warmup_steps=80, base_amp=0.10):
    def schedule(i, t):
        if i < warmup_steps:
            frac = i / max(1, warmup_steps)
            return base_amp * (1.0 + (multiplier - 1.0) * frac)
        return base_amp * multiplier
    return schedule


def overflow_dim_schedule_factory(ramp_steps, base_dim=11, max_extra=5):
    """
    Start safe, then ramp target_dim above the dim limit.
    Since dim_limit=11, returning >11 creates overflow pressure.
    """
    def schedule(i, t):
        if i < ramp_steps:
            frac = i / max(1, ramp_steps)
            extra = int(round(max_extra * frac))
        else:
            extra = max_extra
        return base_dim + extra
    return schedule


def collapsing_injection_schedule_factory(ramp_steps, start=0.35, end=0.02):
    """
    Injection starts supportive, then fades as the run gets more stressed.
    """
    def schedule(i, t):
        if i < ramp_steps:
            frac = i / max(1, ramp_steps)
            return start + (end - start) * frac
        return end
    return schedule


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
        steps=320,
        dt=0.05,
        amp_schedule=ramp_schedule_factory(
            multiplier=5.0,
            warmup_steps=ramp_steps,
            base_amp=0.10,
        ),
        dim_schedule=overflow_dim_schedule_factory(
            ramp_steps=ramp_steps,
            base_dim=11,
            max_extra=5,
        ),
        injection_schedule=collapsing_injection_schedule_factory(
            ramp_steps=ramp_steps,
            start=0.35,
            end=0.02,
        ),
    )

    label, min_eff, final_eff, min_gv, final_gv, max_entropy, max_overflow, min_constraint = classify_run(history)

    return {
        "result": label,
        "min_gv": min_gv,
        "final_gv": final_gv,
        "max_entropy": max_entropy,
        "max_overflow": max_overflow,
        "min_constraint": min_constraint,
        "history": history,
    }


def aggregate_trials(beta, gamma, noise_std, ramp_steps, seeds):
    outcomes = {"damped": 0, "partial": 0, "collapse": 0}
    exemplar = None
    min_gvs = []
    final_gvs = []
    max_entropies = []
    max_overflows = []
    min_constraints = []

    for seed in seeds:
        result = run_single_trial(beta, gamma, noise_std, ramp_steps, seed)
        outcomes[result["result"]] += 1

        min_gvs.append(result["min_gv"])
        final_gvs.append(result["final_gv"])
        max_entropies.append(result["max_entropy"])
        max_overflows.append(result["max_overflow"])
        min_constraints.append(result["min_constraint"])

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
        "avg_min_gv": float(np.mean(min_gvs)),
        "avg_final_gv": float(np.mean(final_gvs)),
        "avg_max_entropy": float(np.mean(max_entropies)),
        "avg_max_overflow": float(np.mean(max_overflows)),
        "avg_min_constraint": float(np.mean(min_constraints)),
        "history": exemplar,
    }


def make_heatmap(rows, metric_key, title, filename, vmin=None, vmax=None):
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
    im = plt.imshow(
        z,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    plt.xticks(range(len(noise_vals)), [f"{v:.2f}" for v in noise_vals])
    plt.yticks(range(len(ramp_vals)), [str(v) for v in ramp_vals])

    plt.xlabel("Noise σ")
    plt.ylabel("Ramp steps (ΔC rate)")
    plt.title(title)

    cbar = plt.colorbar(im)
    cbar.set_label("Value")

    for i in range(len(ramp_vals)):
        for j in range(len(noise_vals)):
            val = z[i, j]
            label = f"{val*100:.0f}%" if vmax == 1 else f"{val:.2f}"
            plt.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if (vmax == 1 and val > 0.5) or (vmax != 1 and val > np.mean(z)) else "black",
            )

    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=180)
    plt.close()

    return path


def main():
    print(">>> NEW MAIN IS RUNNING v3 <<<")

    beta = 1.2
    gamma = 0.65

    noise_vals = [0.25, 0.35, 0.50, 0.70, 0.90, 1.10]
    ramp_vals = [20, 40, 60, 80, 120, 160]
    seeds = list(range(12))

    summary_rows = []

    print("\nGV-CST BLUR-ZONE SWEEP v3\n")
    print(f"{'noise':>6} {'ramp':>6} {'damped%':>8} {'partial%':>9} {'collapse%':>10} {'minGV':>8} {'ovr':>6}")
    print("-" * 72)

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
                f"{100*row['collapse_pct']:>9.1f}% "
                f"{row['avg_min_gv']:>8.2f} "
                f"{row['avg_max_overflow']:>6.2f}"
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
                "avg_min_gv",
                "avg_final_gv",
                "avg_max_entropy",
                "avg_max_overflow",
                "avg_min_constraint",
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
                    "avg_min_gv": row["avg_min_gv"],
                    "avg_final_gv": row["avg_final_gv"],
                    "avg_max_entropy": row["avg_max_entropy"],
                    "avg_max_overflow": row["avg_max_overflow"],
                    "avg_min_constraint": row["avg_min_constraint"],
                }
            )

    partial_map = make_heatmap(
        summary_rows,
        "partial_pct",
        "GV-CST blur zone | partial-collapse fraction",
        "blur_zone_partial_heatmap.png",
        vmin=0,
        vmax=1,
    )

    collapse_map = make_heatmap(
        summary_rows,
        "collapse_pct",
        "GV-CST blur zone | collapse fraction",
        "blur_zone_collapse_heatmap.png",
        vmin=0,
        vmax=1,
    )

    damped_map = make_heatmap(
        summary_rows,
        "damped_pct",
        "GV-CST blur zone | damped fraction",
        "blur_zone_damped_heatmap.png",
        vmin=0,
        vmax=1,
    )

    min_gv_map = make_heatmap(
        summary_rows,
        "avg_min_gv",
        "GV-CST blur zone | average minimum GV",
        "blur_zone_min_gv_heatmap.png",
    )

    overflow_map = make_heatmap(
        summary_rows,
        "avg_max_overflow",
        "GV-CST blur zone | average max overflow",
        "blur_zone_overflow_heatmap.png",
    )

    print("\nSaved summary:", summary_csv)
    print("Saved partial heatmap:", partial_map)
    print("Saved collapse heatmap:", collapse_map)
    print("Saved damped heatmap:", damped_map)
    print("Saved min GV heatmap:", min_gv_map)
    print("Saved overflow heatmap:", overflow_map)


if __name__ == "__main__":
    main()
