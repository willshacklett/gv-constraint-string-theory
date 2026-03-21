import numpy as np
import matplotlib.pyplot as plt


class SpatialGVDynamics:
    """
    2D lattice GV dynamics with periodic boundaries and nearest-neighbor coupling.

    Local update:
        dGV/dt = -beta * C * (1 - GV) + gamma * injection - entropy_drag * entropy
                + neighbor_coupling * Laplacian(GV)

    Collapse rule:
        if GV < collapse_threshold:
            dGV/dt = collapse_rate * (collapse_floor - GV) - collapse_entropy_drag * entropy
                      + neighbor_coupling * Laplacian(GV)

    Notes:
    - Constraint can be spatially distributed
    - Entropy can be scalar or field
    - Injection is treated as global for the sweep
    """

    def __init__(
        self,
        shape=(64, 64),
        beta=1.2,
        gamma=0.65,
        entropy_drag=0.08,
        collapse_threshold=0.35,
        collapse_rate=2.5,
        collapse_floor=0.0,
        collapse_entropy_drag=0.20,
        neighbor_coupling=0.08,
        seed=42,
    ):
        self.shape = shape
        self.beta = beta
        self.gamma = gamma
        self.entropy_drag = entropy_drag
        self.collapse_threshold = collapse_threshold
        self.collapse_rate = collapse_rate
        self.collapse_floor = collapse_floor
        self.collapse_entropy_drag = collapse_entropy_drag
        self.neighbor_coupling = neighbor_coupling
        self.rng = np.random.default_rng(seed)

        self.gv = np.ones(shape, dtype=float)
        self.entropy = np.zeros(shape, dtype=float)

    def reset(self, gv0=0.95, entropy0=0.01, gv_noise=0.02, entropy_noise=0.002):
        self.gv = gv0 + gv_noise * self.rng.standard_normal(self.shape)
        self.entropy = entropy0 + entropy_noise * self.rng.standard_normal(self.shape)

        self.gv = np.clip(self.gv, self.collapse_floor, 1.0)
        self.entropy = np.clip(self.entropy, 0.0, None)

    def laplacian_periodic(self, x):
        return (
            np.roll(x, 1, axis=0)
            + np.roll(x, -1, axis=0)
            + np.roll(x, 1, axis=1)
            + np.roll(x, -1, axis=1)
            - 4.0 * x
        )

    def step(self, constraint_field, injection, entropy_input, dt=0.01):
        if np.isscalar(entropy_input):
            entropy_input_field = np.full(self.shape, entropy_input, dtype=float)
        else:
            entropy_input_field = np.array(entropy_input, dtype=float)

        self.entropy += entropy_input_field * dt
        self.entropy = np.clip(self.entropy, 0.0, None)

        lap = self.laplacian_periodic(self.gv)
        collapsed = self.gv < self.collapse_threshold

        stable_dgv = (
            -self.beta * constraint_field * (1.0 - self.gv)
            + self.gamma * injection
            - self.entropy_drag * self.entropy
            + self.neighbor_coupling * lap
        )

        collapse_dgv = (
            self.collapse_rate * (self.collapse_floor - self.gv)
            - self.collapse_entropy_drag * self.entropy
            + self.neighbor_coupling * lap
        )

        dgv = np.where(collapsed, collapse_dgv, stable_dgv)
        self.gv += dgv * dt
        self.gv = np.clip(self.gv, self.collapse_floor, 1.0)

    def run(self, constraint_field, injection, entropy_input, steps=800, dt=0.01):
        history_mean_gv = []
        history_collapse_fraction = []

        for _ in range(steps):
            self.step(
                constraint_field=constraint_field,
                injection=injection,
                entropy_input=entropy_input,
                dt=dt,
            )
            history_mean_gv.append(float(np.mean(self.gv)))
            history_collapse_fraction.append(float(np.mean(self.gv < self.collapse_threshold)))

        return {
            "mean_gv": np.array(history_mean_gv),
            "collapse_fraction": np.array(history_collapse_fraction),
            "final_mean_gv": float(np.mean(self.gv)),
            "final_collapse_fraction": float(np.mean(self.gv < self.collapse_threshold)),
            "final_gv_field": self.gv.copy(),
            "final_entropy_field": self.entropy.copy(),
        }


def make_constraint_field(shape, base_constraint, heterogeneity=0.03, seed=123):
    rng = np.random.default_rng(seed)
    field = base_constraint + heterogeneity * rng.standard_normal(shape)
    return np.clip(field, 0.0, None)


def classify_survival(final_mean_gv, final_collapse_fraction, mean_gv_threshold=0.40, collapse_fraction_threshold=0.50):
    """
    Survival criterion:
    - Survives if mean GV stays above threshold
    - and collapse fraction does not dominate the grid
    """
    survives = (
        final_mean_gv >= mean_gv_threshold
        and final_collapse_fraction < collapse_fraction_threshold
    )
    return survives


def simulate_spatial_point(
    base_constraint,
    injection,
    entropy_level,
    shape=(64, 64),
    steps=800,
    dt=0.01,
    neighbor_coupling=0.08,
    gv0=0.95,
    seed=42,
):
    model = SpatialGVDynamics(
        shape=shape,
        neighbor_coupling=neighbor_coupling,
        seed=seed,
    )
    model.reset(gv0=gv0, entropy0=0.01)

    constraint_field = make_constraint_field(
        shape=shape,
        base_constraint=base_constraint,
        heterogeneity=0.03,
        seed=seed + 1,
    )

    result = model.run(
        constraint_field=constraint_field,
        injection=injection,
        entropy_input=entropy_level,
        steps=steps,
        dt=dt,
    )

    survives = classify_survival(
        final_mean_gv=result["final_mean_gv"],
        final_collapse_fraction=result["final_collapse_fraction"],
    )

    return {
        "survives": survives,
        "final_mean_gv": result["final_mean_gv"],
        "final_collapse_fraction": result["final_collapse_fraction"],
        "final_gv_field": result["final_gv_field"],
    }


def extract_boundary(constraint_values, injection_values, survive_map):
    """
    For each constraint column, find the minimum injection that survives.
    """
    critical_points = []

    for j, c in enumerate(constraint_values):
        boundary_inj = None
        for i, inj in enumerate(injection_values):
            if survive_map[i, j] == 1:
                boundary_inj = inj
                break
        if boundary_inj is not None:
            critical_points.append((c, boundary_inj))

    return critical_points


def fit_power_law(critical_points):
    """
    Fit inj = A * C^alpha by log-log regression.
    """
    cps = np.array(critical_points, dtype=float)
    c = cps[:, 0]
    inj = cps[:, 1]

    mask = (c > 0) & (inj > 0)
    c = c[mask]
    inj = inj[mask]

    log_c = np.log(c)
    log_inj = np.log(inj)

    alpha, log_A = np.polyfit(log_c, log_inj, 1)
    A = np.exp(log_A)

    pred_log = alpha * log_c + log_A
    residuals = log_inj - pred_log

    ss_res = np.sum((log_inj - pred_log) ** 2)
    ss_tot = np.sum((log_inj - np.mean(log_inj)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "A": A,
        "alpha": alpha,
        "r2": r2,
        "c": c,
        "inj": inj,
        "pred_inj": A * (c ** alpha),
        "log_c": log_c,
        "log_inj": log_inj,
        "pred_log_inj": pred_log,
        "residuals": residuals,
    }


def fit_entropy_shift(entropy_values, boundary_at_fixed_c):
    """
    Fit inj_min(S) = B * exp(m * S)
    Expected here: m should be negative if higher entropy lowers required injection
    under this model's convention.
    """
    s = np.array(entropy_values, dtype=float)
    inj = np.array(boundary_at_fixed_c, dtype=float)

    mask = inj > 0
    s = s[mask]
    inj = inj[mask]

    log_inj = np.log(inj)
    m, log_B = np.polyfit(s, log_inj, 1)
    B = np.exp(log_B)
    pred = B * np.exp(m * s)

    ss_res = np.sum((log_inj - np.log(pred)) ** 2)
    ss_tot = np.sum((log_inj - np.mean(log_inj)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "B": B,
        "m": m,
        "r2": r2,
        "s": s,
        "inj": inj,
        "pred_inj": pred,
    }


def run_spatial_phase_map(
    shape=(64, 64),
    steps=800,
    dt=0.01,
    neighbor_coupling=0.08,
    entropy_level=0.5,
):
    constraint_values = np.linspace(0.10, 2.50, 32)
    injection_values = np.linspace(0.01, 2.20, 36)

    survive_map = np.zeros((len(injection_values), len(constraint_values)))
    mean_gv_map = np.zeros_like(survive_map)

    example_stable = None
    example_boundary = None
    example_collapse = None

    print("Running spatial phase sweep...")
    for i, inj in enumerate(injection_values):
        print(f"  injection row {i + 1}/{len(injection_values)}")
        for j, c in enumerate(constraint_values):
            result = simulate_spatial_point(
                base_constraint=c,
                injection=inj,
                entropy_level=entropy_level,
                shape=shape,
                steps=steps,
                dt=dt,
                neighbor_coupling=neighbor_coupling,
                seed=1000 + i * 100 + j,
            )

            survives = result["survives"]
            survive_map[i, j] = 1 if survives else 0
            mean_gv_map[i, j] = result["final_mean_gv"]

            if survives and example_stable is None and c < 1.0:
                example_stable = result["final_gv_field"]

            if (not survives) and example_collapse is None and c > 1.5 and inj < 0.5:
                example_collapse = result["final_gv_field"]

    critical_points = extract_boundary(constraint_values, injection_values, survive_map)

    if len(critical_points) < 3:
        raise RuntimeError("Not enough boundary points found to fit a power law.")

    fit = fit_power_law(critical_points)

    boundary_lookup = {round(c, 6): inj for c, inj in critical_points}
    mid_c_index = len(constraint_values) // 2
    mid_c = constraint_values[mid_c_index]
    if round(mid_c, 6) in boundary_lookup:
        boundary_inj = boundary_lookup[round(mid_c, 6)]
        boundary_result = simulate_spatial_point(
            base_constraint=mid_c,
            injection=boundary_inj,
            entropy_level=entropy_level,
            shape=shape,
            steps=steps,
            dt=dt,
            neighbor_coupling=neighbor_coupling,
            seed=9999,
        )
        example_boundary = boundary_result["final_gv_field"]

    return {
        "constraint_values": constraint_values,
        "injection_values": injection_values,
        "survive_map": survive_map,
        "mean_gv_map": mean_gv_map,
        "critical_points": critical_points,
        "fit": fit,
        "example_stable": example_stable,
        "example_boundary": example_boundary,
        "example_collapse": example_collapse,
        "entropy_level": entropy_level,
    }


def run_entropy_modulation_test(
    fixed_constraint=1.2,
    entropy_values=None,
    shape=(64, 64),
    steps=800,
    dt=0.01,
    neighbor_coupling=0.08,
):
    if entropy_values is None:
        entropy_values = np.linspace(0.0, 6.0, 9)

    injection_values = np.linspace(0.01, 2.20, 40)
    boundary_injections = []

    print("\nRunning entropy modulation test...")
    for s in entropy_values:
        print(f"  entropy={s:.2f}")
        found = None
        for inj in injection_values:
            result = simulate_spatial_point(
                base_constraint=fixed_constraint,
                injection=inj,
                entropy_level=s,
                shape=shape,
                steps=steps,
                dt=dt,
                neighbor_coupling=neighbor_coupling,
                seed=5000 + int(100 * s) + int(100 * inj),
            )
            if result["survives"]:
                found = inj
                break

        if found is None:
            found = np.nan

        boundary_injections.append(found)

    fit = fit_entropy_shift(entropy_values, boundary_injections)

    return {
        "entropy_values": np.array(entropy_values, dtype=float),
        "boundary_injections": np.array(boundary_injections, dtype=float),
        "fit": fit,
        "fixed_constraint": fixed_constraint,
    }


def plot_spatial_results(phase_result, entropy_result):
    constraint_values = phase_result["constraint_values"]
    injection_values = phase_result["injection_values"]
    survive_map = phase_result["survive_map"]
    mean_gv_map = phase_result["mean_gv_map"]
    critical_points = phase_result["critical_points"]
    fit = phase_result["fit"]

    plt.figure(figsize=(10, 7))
    plt.imshow(
        mean_gv_map,
        origin="lower",
        aspect="auto",
        extent=[
            constraint_values.min(),
            constraint_values.max(),
            injection_values.min(),
            injection_values.max(),
        ],
    )
    plt.colorbar(label="Final Mean GV")
    plt.contour(
        constraint_values,
        injection_values,
        survive_map,
        levels=[0.5],
        linewidths=2,
    )
    c_pts, inj_pts = zip(*critical_points)
    plt.plot(c_pts, inj_pts, "o", markersize=3, label="Boundary Points")
    plt.plot(fit["c"], fit["pred_inj"], linewidth=2, linestyle="--", label=f"Power fit: inj={fit['A']:.3f}·C^{fit['alpha']:.3f}")
    plt.xlabel("Constraint (C)")
    plt.ylabel("Injection")
    plt.title("Spatial GV Phase Map (2D lattice)")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.plot(fit["log_c"], fit["log_inj"], "o", label="Boundary data")
    plt.plot(fit["log_c"], fit["pred_log_inj"], linewidth=2, linestyle="--", label=f"Log-log fit (R²={fit['r2']:.4f})")
    plt.xlabel("log(C)")
    plt.ylabel("log(inj_min)")
    plt.title("Log-Log Boundary Fit")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.axhline(0.0, linewidth=1)
    plt.plot(fit["c"], fit["residuals"], "o-")
    plt.xlabel("Constraint (C)")
    plt.ylabel("Log residual")
    plt.title("Boundary Fit Residuals")
    plt.tight_layout()

    efit = entropy_result["fit"]
    plt.figure(figsize=(8, 6))
    plt.plot(efit["s"], efit["inj"], "o", label="Entropy boundary data")
    plt.plot(efit["s"], efit["pred_inj"], linewidth=2, linestyle="--", label=f"Exp fit: inj={efit['B']:.3f}·exp({efit['m']:.3f}·S)")
    plt.xlabel("Entropy (S)")
    plt.ylabel("inj_min")
    plt.title(f"Entropy Shift at Fixed Constraint C={entropy_result['fixed_constraint']:.2f}")
    plt.legend()
    plt.tight_layout()

    for title, field in [
        ("Example Stable GV Field", phase_result["example_stable"]),
        ("Example Boundary GV Field", phase_result["example_boundary"]),
        ("Example Collapse GV Field", phase_result["example_collapse"]),
    ]:
        if field is not None:
            plt.figure(figsize=(6, 5))
            plt.imshow(field, origin="lower", aspect="equal")
            plt.colorbar(label="GV")
            plt.title(title)
            plt.tight_layout()

    plt.show()


def main():
    phase_result = run_spatial_phase_map(
        shape=(64, 64),
        steps=700,
        dt=0.01,
        neighbor_coupling=0.08,
        entropy_level=0.5,
    )

    entropy_result = run_entropy_modulation_test(
        fixed_constraint=1.2,
        entropy_values=np.linspace(0.0, 6.0, 9),
        shape=(64, 64),
        steps=700,
        dt=0.01,
        neighbor_coupling=0.08,
    )

    fit = phase_result["fit"]
    efit = entropy_result["fit"]

    print("\n" + "=" * 70)
    print("SPATIAL FIT RESULTS")
    print("=" * 70)
    print(f"Power-law fit:")
    print(f"  inj_min(C) ≈ {fit['A']:.4f} * C^{fit['alpha']:.4f}")
    print(f"  R^2 = {fit['r2']:.6f}")

    print("\nEntropy modulation fit:")
    print(f"  inj_min(S) ≈ {efit['B']:.4f} * exp({efit['m']:.4f} * S)")
    print(f"  R^2 = {efit['r2']:.6f}")

    print("\nCombined working form:")
    print(f"  inj_min(C,S) ≈ A * C^alpha * exp(k*S)")
    print(f"  alpha ≈ {fit['alpha']:.4f}")
    print(f"  k ≈ {efit['m']:.4f}")
    print("=" * 70)

    plot_spatial_results(phase_result, entropy_result)


if __name__ == "__main__":
    main()
