import numpy as np
import matplotlib.pyplot as plt

from gv_dynamics import GVDynamics


def run_trial(C, injection, steps=100, entropy_input=0.008):
    model = GVDynamics(
        beta=1.2,
        gamma=0.65,
        entropy_drag=0.08,
        gv_crit=0.42,
        collapse_rate=0.06,
        collapse_entropy_gain=0.03,
        collapse_feedback=0.20,
        dt=1.0,
    )
    model.reset(gv=1.0, entropy=0.0)

    collapsed_step = None
    final_gv = None

    for t in range(steps):
        out = model.step(C=C, injection=injection, entropy_input=entropy_input)
        final_gv = out["gv"]
        if out["collapsed"] and collapsed_step is None:
            collapsed_step = t

    if collapsed_step is None:
        collapsed_step = steps

    return collapsed_step, final_gv


def main():
    C_values = np.linspace(0.1, 1.2, 60)
    injection_values = np.linspace(0.0, 0.5, 60)

    heat = np.zeros((len(C_values), len(injection_values)))

    for i, C in enumerate(C_values):
        for j, injection in enumerate(injection_values):
            collapsed_step, final_gv = run_trial(C, injection)
            heat[i, j] = collapsed_step

    plt.figure(figsize=(10, 7))
    plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[
            injection_values[0],
            injection_values[-1],
            C_values[0],
            C_values[-1],
        ],
    )
    plt.colorbar(label="Collapse step (higher = more stable)")
    plt.xlabel("Injection")
    plt.ylabel("Constraint pressure C")
    plt.title("GV collapse boundary heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
