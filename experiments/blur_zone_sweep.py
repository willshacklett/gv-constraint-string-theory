import csv
import math
import os

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

    # Full collapse
    if final_eff < 0.0005 or min_gv < 0.12:
        return "collapse", min_eff, final_eff, min_gv, final_gv, max_entropy

    # Partial collapse / blur zone
    if min_gv < 0.45 or final_gv < 0.60 or max_entropy > 2.5:
        return "partial", min_eff, final_eff, min_gv, final_gv, max_entropy

    # Damped / recovery
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
        amp_schedule=ramp_schedule_factory(multiplier=2.5, warmup_steps=ramp_steps, base_amp=0.10),
        dim_schedule=no_overflow_dim_schedule,
        injection_schedule=moderate_injection,
    )

    label, min_eff, final_eff, min_gv, final_gv, max_entropy = classify_run(history)

    return {
        "result": label,
        "min_eff": min_eff,
        "final_eff": final_eff,
        "min_gv": min_gv,
        "final_gv": final_gv,
        "max_entropy": max_entropy,
        "history": history,
    }


def aggregate_trials(beta, gamma, noise_std, ramp_steps, seeds):
    outcomes = {"damped": 0, "partial": 0, "collapse": 0}
    min_gvs = []
    final_gvs = []
    max_entropies = []
    exemplar = None

    for seed in seeds:
        result = run_single_trial(beta, gamma, noise_std, ramp_steps, seed)
        outcomes[result["result"]] += 1
        min_gvs.append(result["min_gv"])
        final_gvs.append(result["final_gv"])
        max_entropies.append(result["max_entropy"])

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
        "mean_min_gv": sum(min_gvs) / n,
        "mean_final_gv": sum(final_gvs) / n,
        "mean_max_entropy": sum(max_entropies) / n,
        "history": exemplar,
    }


def save_exemplar_csv(row):
    case_name = (
        f"blur_exemplar_beta{row['beta']}_gamma{row['gamma']}_"
        f"noise{str(row['noise_std']).replace('.', 'p')}_ramp{row['ramp_steps']}"
    )
    csv_path = os.path.join(OUT_DIR, f"{case_name}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "value",
                "amp",
                "target_dim",
                "overflow",
                "constraint_base",
                "constraint_noise",
                "constraint_noisy",
                "gv",
                "entropy",
                "raw_energy",
                "effective_energy",
                "injection",
            ],
        )
        writer.writeheader()
        writer.writerows(row["history"])

    return csv_path


def make_heatmap(rows, metric_key, title, filename):
    noise_vals = sorted(set(row["noise_std"] for row in rows))
    ramp_vals = sorted(set(row["ramp_steps"] for row in rows))

    z = []
    for ramp in ramp_vals:
        z_row = []
        for noise in noise_vals:
            match = next(
                r for r in rows
                if math.isclose(r["noise_std"], noise) and r["ramp_steps"] == ramp
            )
            z_row.append(match[metric_key])
        z.append(z_row)

    plt.figure(figsize=(9, 6))
    plt.imshow(z, aspect="auto", origin="lower")
    plt.xticks(range(len(noise_vals)), [f"{v:.2f}" for v in noise_vals])
    plt.yticks(range(len(ramp_vals)), [str(v) for v in ramp_vals])
    plt.xlabel("Noise σ")
    plt.ylabel("Ramp steps")
    plt.title(title)
    plt.colorbar(label=metric_key)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def make_line_plot(best_row):
    history = best_row["history"]
    times = [r["time"] for r in history]
    gvs = [r["gv"] for r in history]
    ents = [r["entropy"] for r in history]
    effs = [r["effective_energy"] for r in history]
    cvals = [r["constraint_noisy"] for r in history]

    plt.figure(figsize=(10, 6))
    plt.plot(times, gvs, label="GV")
    plt.plot(times, ents, label="Entropy")
    plt.plot(times, effs, label="E_eff")
    plt.plot(times, cvals, label="C_noisy")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(
        f"Blur-zone exemplar | beta={best_row['beta']}, gamma={best_row['gamma']}, "
        f"sigma={best_row['noise_std']}, ramp_steps={best_row['ramp_steps']}"
    )
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "blur_zone_exemplar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    beta = 1.2
    gamma = 0.65

    noise_vals = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]
    ramp_vals = [20, 40, 60, 80, 120, 160, 200]
    seeds = list(range(10))

    summary_rows = []

    print("\nGV-CST BLUR-ZONE SWEEP\n")
    print(
        f"{'noise':>6} {'ramp':>6} {'damped%':>8} {'partial%':>9} "
        f"{'collapse%':>10} {'minGV':>8} {'finalGV':>8} {'maxS':>8}"
    )
    print("-" * 80)

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
                f"{row['mean_min_gv']:>8.3f} "
                f"{row['mean_final_gv']:>8.3f} "
                f"{row['mean_max_entropy']:>8.3f}"
            )

    summary_csv = os.path.join(OUT_DIR, "blur_zone_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "beta",
                "gamma",
                "noise_std",
                "ramp_steps",
                "damped_pct",
                "partial_pct",
                "collapse_pct",
                "mean_min_gv",
                "mean_final_gv",
                "mean_max_entropy",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "beta": row["beta"],
                    "gamma": row["gamma"],
                    "noise_std": f"{row['noise_std']:.2f}",
                    "ramp_steps": row["ramp_steps"],
                    "damped_pct": f"{row['damped_pct']:.4f}",
                    "partial_pct": f"{row['partial_pct']:.4f}",
                    "collapse_pct": f"{row['collapse_pct']:.4f}",
                    "mean_min_gv": f"{row['mean_min_gv']:.6f}",
                    "mean_final_gv": f"{row['mean_final_gv']:.6f}",
                    "mean_max_entropy": f"{row['mean_max_entropy']:.6f}",
                }
            )

    damped_heatmap = make_heatmap(
        summary_rows,
        metric_key="damped_pct",
        title="GV-CST blur zone | damped fraction",
        filename="blur_zone_damped_heatmap.png",
    )

    partial_heatmap = make_heatmap(
        summary_rows,
        metric_key="partial_pct",
        title="GV-CST blur zone | partial-collapse fraction",
        filename="blur_zone_partial_heatmap.png",
    )

    collapse_heatmap = make_heatmap(
        summary_rows,
        metric_key="collapse_pct",
        title="GV-CST blur zone | collapse fraction",
        filename="blur_zone_collapse_heatmap.png",
    )

    # Pick the row with highest partial fraction as exemplar blur-zone case
    best_row = max(summary_rows, key=lambda r: r["partial_pct"])
    exemplar_csv = save_exemplar_csv(best_row)
    exemplar_plot = make_line_plot(best_row)

    print("\nSaved summary:", summary_csv)
    print("Saved damped heatmap:", damped_heatmap)
    print("Saved partial heatmap:", partial_heatmap)
    print("Saved collapse heatmap:", collapse_heatmap)
    print("Saved exemplar CSV:", exemplar_csv)
    print("Saved exemplar plot:", exemplar_plot)


if __name__ == "__main__":
    main()
