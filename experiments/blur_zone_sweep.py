import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== GV-CST NEW FILE LOADED v2 ===")

BETA = 1.5
GAMMA = 0.10

NOISE_VALS = [0.25, 0.35, 0.50, 0.70, 0.90, 1.20]
RAMP_VALS = [20, 50, 80, 120, 180, 250]
SEEDS = list(range(10))

N_STEPS = 220
DT = 0.08

COLLAPSE_THRESHOLD = 2.0
PARTIAL_THRESHOLD = 0.8
SETTLE_THRESHOLD = 0.5


def ensure_output_dir():
    out_dir = os.path.join("data", "logs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def smoothstep(z):
    z = max(0.0, min(1.0, z))
    return z * z * (3.0 - 2.0 * z)


def ramp_force(step, ramp_steps):
    if ramp_steps <= 0:
        return 1.0
    return smoothstep(step / float(ramp_steps))


def simulate_run(noise_sigma, ramp_steps, seed):
    rng = np.random.default_rng(seed)

    x = 0.05
    v = 0.0

    max_abs_x = abs(x)
    crossed_partial = False
    collapsed = False

    for step in range(N_STEPS):
        force = ramp_force(step, ramp_steps)
        noise = rng.normal(0.0, noise_sigma)

        nonlinear = x**3 - 0.5 * x

        a = (
            (BETA * x)
            + nonlinear
            - (GAMMA * v)
            + (3.5 * force)
            + (1.5 * noise)
        )

        v = v + DT * a
        x = x + DT * v

        ax = abs(x)
        max_abs_x = max(max_abs_x, ax)

        if ax >= PARTIAL_THRESHOLD:
            crossed_partial = True

        if ax >= COLLAPSE_THRESHOLD:
            collapsed = True
            break

    final_abs_x = abs(x)

    if collapsed:
        label = "collapse"
    elif crossed_partial:
        label = "partial"
    elif final_abs_x <= SETTLE_THRESHOLD:
        label = "damped"
    else:
        label = "partial"

    return {
        "noise": noise_sigma,
        "ramp": ramp_steps,
        "label": label,
        "max_abs_x": float(max_abs_x),
        "final_abs_x": float(final_abs_x),
    }


def summarize_cell(noise, ramp, seeds):
    results = [simulate_run(noise, ramp, s) for s in seeds]

    total = len(results)
    damped = sum(1 for r in results if r["label"] == "damped")
    partial = sum(1 for r in results if r["label"] == "partial")
    collapse = sum(1 for r in results if r["label"] == "collapse")

    return {
        "noise": noise,
        "ramp": ramp,
        "damped_pct": damped / total,
        "partial_pct": partial / total,
        "collapse_pct": collapse / total,
        "avg_max_abs_x": np.mean([r["max_abs_x"] for r in results]),
    }


def make_heatmap(rows, key, title, path):
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="ramp", columns="noise", values=key)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Ramp")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center", color="white")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def main():
    print(">>> GV-CST NEW MAIN v2 <<<")
    print("NOISE_VALS =", NOISE_VALS)
    print("RAMP_VALS  =", RAMP_VALS)

    out_dir = ensure_output_dir()
    rows = []

    for ramp in RAMP_VALS:
        for noise in NOISE_VALS:
            row = summarize_cell(noise, ramp, SEEDS)
            rows.append(row)

            print(
                f"{noise:.2f}  {ramp:<4}  "
                f"{row['damped_pct']*100:>5.1f}%  "
                f"{row['partial_pct']*100:>5.1f}%  "
                f"{row['collapse_pct']*100:>5.1f}%"
            )

    csv_path = os.path.join(out_dir, "blur_zone_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    partial_map = make_heatmap(
        rows,
        "partial_pct",
        "GV-CST | Partial (Blur Zone) v2",
        os.path.join(out_dir, "blur_zone_partial_heatmap.png"),
    )

    collapse_map = make_heatmap(
        rows,
        "collapse_pct",
        "GV-CST | Collapse v2",
        os.path.join(out_dir, "blur_zone_collapse_heatmap.png"),
    )

    damped_map = make_heatmap(
        rows,
        "damped_pct",
        "GV-CST | Damped v2",
        os.path.join(out_dir, "blur_zone_damped_heatmap.png"),
    )

    amp_map = make_heatmap(
        rows,
        "avg_max_abs_x",
        "GV-CST | Max Amplitude v2",
        os.path.join(out_dir, "blur_zone_maxamp_heatmap.png"),
    )

    print("\nSaved:", csv_path)
    print("Saved:", partial_map)
    print("Saved:", collapse_map)
    print("Saved:", damped_map)
    print("Saved:", amp_map)


if __name__ == "__main__":
    main()
