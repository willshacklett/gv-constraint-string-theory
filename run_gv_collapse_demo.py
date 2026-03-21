import matplotlib.pyplot as plt

from gv_dynamics import GVDynamics


def run_case(name, C, injection, entropy_schedule, steps=80):
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

    gv_series = []
    entropy_series = []
    regime_series = []

    for t in range(steps):
        entropy_input = entropy_schedule(t)
        out = model.step(C=C, injection=injection, entropy_input=entropy_input)
        gv_series.append(out["gv"])
        entropy_series.append(out["entropy"])
        regime_series.append(out["regime"])

    return gv_series, entropy_series, regime_series


def main():
    steps = 80

    cases = [
        ("mild_stress", 0.35, 0.12, lambda t: 0.004),
        ("borderline", 0.55, 0.08, lambda t: 0.007),
        ("collapse", 0.75, 0.03, lambda t: 0.012),
    ]

    plt.figure(figsize=(10, 6))

    for name, C, injection, entropy_schedule in cases:
        gv, entropy, regime = run_case(name, C, injection, entropy_schedule, steps=steps)
        plt.plot(gv, label=name)

    plt.axhline(0.42, linestyle="--", label="gv_crit")
    plt.xlabel("Step")
    plt.ylabel("GV")
    plt.title("GV dynamics: stable vs collapse")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
