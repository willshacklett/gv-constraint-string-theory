import csv
import os

import matplotlib.pyplot as plt

from src.pressure_simulation import PressureSimulation


OUT_DIR = "data/logs"
os.makedirs(OUT_DIR, exist_ok=True)


def classify_run(history):
    min_eff = min(row["effective_energy"] for row in history)
    max_entropy = max(row["entropy"] for row in history)
    min_gv = min(row["gv"] for row in history)
    final_gv = history[-1]["gv"]
    final_eff = history[-1]["effective_energy"]

    # Heuristic labels for toy model
    if final_eff < 0.0005 or min_gv < 0.12:
        return "collapse", min_eff, max_entropy, min_gv, final_gv
    if max_entropy > 0.35 and final_gv > 0.45:
        return "damped", min_eff, max_entropy, min_gv, final_gv
    return "stable", min_eff, max_entropy, min_gv, final_gv


def gradual_amp_schedule_factory(multiplier=2.5, warmup_steps=120, base_amp=0.10):
    def schedule(i, t):
        if i < warmup_steps:
            frac = i / max(1, warmup_steps)
            return base_amp * (1.0 + (multiplier - 1.0) * frac)
        return base_amp * multiplier

    return schedule


def step_amp_schedule_factory(multiplier=2.5, step_at=50, base_amp=0.10):
    def schedule(i, t):
        return base_amp if i < step_at else base_amp * multiplier

    return schedule


def no_overflow_dim_schedule(i, t):
    return 11


def hard_overflow_dim_schedule(i, t):
    return 13


def delayed_overflow_dim_schedule_factory(step_at=50, pre_dim=11, post_dim=13):
    def schedule(i, t):
        return pre_dim if i < step_at else post_dim

    return schedule


def moderate_injection(i, t):
    return 0.40


def low_injection(i, t):
    return 0.25


def run_case(case_name, beta, gamma, amp_schedule, dim_schedule, inj_schedule):
    sim = PressureSimulation(beta=beta, gamma=gamma, dim_limit=11, base_amp=0.10, freq=1.0)
    history = sim.run(
        steps=300,
        dt=0.05,
        amp_schedule=amp_schedule,
        dim_schedule=dim_schedule,
        injection_schedule=inj_schedule,
    )
    label, min_eff, max_entropy, min_gv, final_gv = classify_run(history)

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
                "constraint",
                "gv",
                "entropy",
                "raw_energy",
                "effective_energy",
                "injection",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    return {
        "case": case_name,
        "beta": beta,
        "gamma": gamma,
        "result": label,
        "min_eff": min_eff,
        "max_entropy": max_entropy,
        "min_gv": min_gv,
        "final_gv": final_gv,
        "csv_path": csv_path,
        "history": history,
    }


def make_plot(case_result):
    history = case_result["history"]
    times = [r["time"] for r in history]
    gvs = [r["gv"] for r in history]
    ents = [r["entropy"] for r in history]
    effs = [r["effective_energy"] for r in history]

    plt.figure(figsize=(10, 6))
    plt.plot(times, gvs, label="GV")
    plt.plot(times, ents, label="Entropy")
    plt.plot(times, effs, label="E_eff")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(
        f"{case_result['case']} | beta={case_result['beta']}, gamma={case_result['gamma']} | {case_result['result']}"
    )
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, f"{case_result['case']}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


def main():
    # Grok-aligned ranges
    betas = [0.8, 1.0, 1.2, 1.4, 1.6]
    gammas = [0.4, 0.65, 0.9]

    summary_rows = []

    for beta in betas:
        for gamma in gammas:
            # Case A: gradual 2.5x ramp, no dimensional overflow
            case_a = run_case(
                case_name=f"gradual_beta{beta}_gamma{gamma}",
                beta=beta,
                gamma=gamma,
                amp_schedule=gradual_amp_schedule_factory(multiplier=2.5, warmup_steps=120),
                dim_schedule=no_overflow_dim_schedule,
                inj_schedule=moderate_injection,
            )
            case_a["plot_path"] = make_plot(case_a)
            summary_rows.append(case_a)

            # Case B: step overflow
            case_b = run_case(
                case_name=f"step_overflow_beta{beta}_gamma{gamma}",
                beta=beta,
                gamma=gamma,
                amp_schedule=step_amp_schedule_factory(multiplier=2.5, step_at=50),
                dim_schedule=delayed_overflow_dim_schedule_factory(step_at=50, pre_dim=11, post_dim=13),
                inj_schedule=moderate_injection,
            )
            case_b["plot_path"] = make_plot(case_b)
            summary_rows.append(case_b)

    summary_csv = os.path.join(OUT_DIR, "pressure_sweep_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "beta",
                "gamma",
                "result",
                "min_eff",
                "max_entropy",
                "min_gv",
                "final_gv",
                "csv_path",
                "plot_path",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "beta": row["beta"],
                    "gamma": row["gamma"],
                    "result": row["result"],
                    "min_eff": f"{row['min_eff']:.6f}",
                    "max_entropy": f"{row['max_entropy']:.6f}",
                    "min_gv": f"{row['min_gv']:.6f}",
                    "final_gv": f"{row['final_gv']:.6f}",
                    "csv_path": row["csv_path"],
                    "plot_path": row["plot_path"],
                }
            )

    print("\nGV-CST PRESSURE SWEEP\n")
    print(f"{'case':40} {'beta':>4} {'gamma':>5} {'result':>9} {'min_gv':>8} {'final_gv':>9} {'max_S':>8}")
    print("-" * 95)

    for row in summary_rows:
        print(
            f"{row['case'][:40]:40} "
            f"{row['beta']:>4.1f} "
            f"{row['gamma']:>5.2f} "
            f"{row['result']:>9} "
            f"{row['min_gv']:>8.3f} "
            f"{row['final_gv']:>9.3f} "
            f"{row['max_entropy']:>8.3f}"
        )

    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved logs/plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
