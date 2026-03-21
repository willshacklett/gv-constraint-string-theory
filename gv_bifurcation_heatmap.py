import numpy as np
import matplotlib.pyplot as plt

from src.gv_dynamics import GVDynamics


def simulate_point(constraint, injection, steps=600, dt=0.01):
    model = GVDynamics()
    model.reset(gv=0.95, entropy=0.01)

    for _ in range(steps):
        state = model.step(
            constraint=constraint,
            injection=injection,
            entropy_input=0.05,
            dt=dt,
        )

    return state.gv, state.collapsed


def run_heatmap():
    # Parameter ranges
    constraint_values = np.linspace(0.0, 1.2, 60)
    injection_values = np.linspace(0.0, 0.8, 60)

    gv_map = np.zeros((len(injection_values), len(constraint_values)))
    collapse_map = np.zeros_like(gv_map)

    # --- Sweep ---
    for i, inj in enumerate(injection_values):
        for j, c in enumerate(constraint_values):
            gv, collapsed = simulate_point(c, inj)
            gv_map[i, j] = gv
            collapse_map[i, j] = 1 if collapsed else 0

    # --- Critical boundary extraction ---
    critical_points = []

    for j, c in enumerate(constraint_values):
        boundary_inj = None

        for i in range(len(injection_values) - 1):
            below = collapse_map[i, j]
            above = collapse_map[i + 1, j]

            # detect transition from collapse → stable
            if below == 1 and above == 0:
                inj_low = injection_values[i]
                inj_high = injection_values[i + 1]

                # linear interpolation for smoother boundary
                boundary_inj = (inj_low + inj_high) / 2.0
                break

        if boundary_inj is not None:
            critical_points.append((c, boundary_inj))

    # --- Plot ---
    plt.figure(figsize=(10, 7))

    # Heatmap
    im = plt.imshow(
        gv_map,
        origin="lower",
        aspect="auto",
        extent=[
            constraint_values.min(),
            constraint_values.max(),
            injection_values.min(),
            injection_values.max(),
        ],
    )

    plt.colorbar(im, label="Final GV")

    # Collapse overlay (semi-transparent red)
    plt.imshow(
        collapse_map,
        origin="lower",
        aspect="auto",
        extent=[
            constraint_values.min(),
            constraint_values.max(),
            injection_values.min(),
            injection_values.max(),
        ],
        alpha=0.25,
    )

    # --- Plot critical boundary ---
    if critical_points:
        c_vals, inj_vals = zip(*critical_points)
        plt.plot(
            c_vals,
            inj_vals,
            linewidth=2,
            linestyle="--",
            label="Critical Boundary",
        )

    # Labels
    plt.xlabel("Constraint (C)")
    plt.ylabel("Injection (Support)")
    plt.title("GV Phase Map: Stability vs Collapse")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_heatmap()
