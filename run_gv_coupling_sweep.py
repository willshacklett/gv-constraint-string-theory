import numpy as np
import matplotlib.pyplot as plt


class SpatialGVDynamics:
    """
    2D lattice GV dynamics with periodic boundaries and nearest-neighbor coupling.
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

    def run(self, constraint_field, injection, entropy_input, steps=700, dt=0.01):
        for _ in range(steps):
            self.step(
                constraint_field=constraint_field,
                injection=injection,
                entropy_input=entropy_input,
                dt=dt,
            )

        return {
            "final_mean_gv": float(np.mean(self.gv)),
            "final_collapse_fraction": float(np.mean(self.gv < self.collapse_threshold)),
            "final_gv_field": self.gv.copy(),
        }


def make_constraint_field(shape, base_constraint, heterogeneity=0.03, seed=123):
    rng = np.random.default_rng(seed)
    field = base_constraint + heterogeneity * rng.standard_normal(shape)
    return np.clip(field, 0.0, None)


def classify_survival(
    final_mean_gv,
    final_collapse_fraction,
    mean_gv_threshold=0.40,
    collapse_fraction_threshold=0.50,
):
    return (
        final_mean_gv >= mean_gv_threshold
        and final_collapse_fraction < collapse_fraction_threshold
    )


def simulate_spatial_point(
    base_constraint,
    injection,
    entropy_level,
    neighbor_coupling,
    shape=(64, 64),
    steps=700,
    dt=0.01,
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

    return survives, result


def extract_boundary(constraint_values, injection_values, survive_map):
    """
    For each constraint value, find the minimum injection that survives.
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
        "residuals": residuals,
    }


def fit_entropy_shift(entropy_values, boundary_injections):
    s = np.array(entropy_values, dtype=float)
    inj = np.array(boundary_injections, dtype=float)

    mask = np.isfinite(inj) & (inj > 0)
    s = s[mask]
    inj = inj[mask]

    if len(s) < 3:
        raise RuntimeError("Not enough valid entropy boundary points to fit exponential.")

    log_inj = np.log(inj)
    k, log_B = np.polyfit(s, log_inj, 1)
    B = np.exp(log_B)

    pred_inj = B * np.exp(k * s)

    ss_res = np.sum((log_inj - np.log(pred_inj)) ** 2)
    ss_tot = np.sum((log_inj - np.mean(log_inj)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "B": B,
        "k": k,
        "r2": r2,
        "s": s,
        "inj": inj,
        "pred_inj": pred_inj,
    }


def run_phase_boundary_for_lambda(
    lam,
    shape=(64, 64),
    steps=700,
    dt=0.01,
    entropy_level=0.5,
):
    constraint_values = np.linspace(0.10, 2.50, 28)
    injection_values = np.linspace(0.01, 2.20, 32)

    survive_map = np.zeros((len(injection_values), len(constraint_values)))

    for i, inj in enumerate(injection_values):
        for j, c in enumerate(constraint_values):
            survives, _ = simulate_spatial_point(
                base_constraint=c,
                injection=inj,
                entropy_level=entropy_level,
                neighbor_coupling=lam,
                shape=shape,
                steps=steps,
                dt=dt,
                seed=10000 + int(1000 * lam) + i * 100 + j,
            )
            survive_map[i, j] = 1 if survives else 0

    critical_points = extract_boundary(constraint_values, injection_values, survive_map)

    if len(critical_points) < 5:
        raise RuntimeError(f"Not enough critical points for lambda={lam:.4f}")

    fit = fit_power_law(critical_points)

    return {
        "lambda": lam,
        "constraint_values": constraint_values,
        "injection_values": injection_values,
        "survive_map": survive_map,
        "critical_points": critical_points,
        "fit": fit,
    }


def run_entropy_boundary_for_lambda(
    lam,
    fixed_constraint=1.2,
    entropy_values=None,
    shape=(64, 64),
    steps=700,
    dt=0.01,
):
    if entropy_values is None:
        entropy_values = np.linspace(0.0, 6.0, 9)

    injection_values = np.linspace(0.01, 2.20, 36)
    boundary_injections = []

    for s in entropy_values:
        found = None
        for inj in injection_values:
            survives, _ = simulate_spatial_point(
                base_constraint=fixed_constraint,
                injection=inj,
                entropy_level=s,
                neighbor_coupling=lam,
                shape=shape,
                steps=steps,
                dt=dt,
                seed=20000 + int(1000 * lam) + int(100 * s) + int(100 * inj),
            )
            if survives:
                found = inj
                break

        if found is None:
            found = np.nan

        boundary_injections.append(found)

    fit = fit_entropy_shift(entropy_values, boundary_injections)

    return {
        "lambda": lam,
        "fixed_constraint": fixed_constraint,
        "entropy_values": np.array(entropy_values, dtype=float),
        "boundary_injections": np.array(boundary_injections, dtype=float),
        "fit": fit,
    }


def run_coupling_sweep(
    lambda_values=None,
    shape=(64, 64),
    steps=700,
    dt=0.01,
):
    if lambda_values is None:
        lambda_values = np.array([0.00, 0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20])

    sweep_rows = []
    phase_details = []
    entropy_details = []

    print("Running coupling sweep...\n")

    for lam in lambda_values:
        print(f"lambda = {lam:.4f}")

        phase_result = run_phase_boundary_for_lambda(
            lam=lam,
            shape=shape,
            steps=steps,
            dt=dt,
            entropy_level=0.5,
        )
        entropy_result = run_entropy_boundary_for_lambda(
            lam=lam,
            fixed_constraint=1.2,
            entropy_values=np.linspace(0.0, 6.0, 9),
            shape=shape,
            steps=steps,
            dt=dt,
        )

        pfit = phase_result["fit"]
        efit = entropy_result["fit"]

        row = {
            "lambda": lam,
            "A": pfit["A"],
            "alpha": pfit["alpha"],
            "alpha_r2": pfit["r2"],
            "B": efit["B"],
            "k": efit["k"],
            "k_r2": efit["r2"],
        }
        sweep_rows.append(row)
        phase_details.append(phase_result)
        entropy_details.append(entropy_result)

        print(
            f"  alpha = {pfit['alpha']:.4f}, A = {pfit['A']:.4f}, R^2 = {pfit['r2']:.4f}"
        )
        print(
            f"  k     = {efit['k']:.4f}, B = {efit['B']:.4f}, R^2 = {efit['r2']:.4f}"
        )
        print()

    return sweep_rows, phase_details, entropy_details


def plot_coupling_sweep(sweep_rows):
    lambdas = np.array([row["lambda"] for row in sweep_rows])
    alphas = np.array([row["alpha"] for row in sweep_rows])
    ks = np.array([row["k"] for row in sweep_rows])
    As = np.array([row["A"] for row in sweep_rows])
    alpha_r2 = np.array([row["alpha_r2"] for row in sweep_rows])
    k_r2 = np.array([row["k_r2"] for row in sweep_rows])

    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, alphas, "o-", linewidth=2)
    plt.xlabel("Neighbor coupling λ")
    plt.ylabel("Power-law exponent α")
    plt.title("Exponent Flow: α(λ)")
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, ks, "o-", linewidth=2)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Neighbor coupling λ")
    plt.ylabel("Entropy exponent k")
    plt.title("Entropy Flow: k(λ)")
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, As, "o-", linewidth=2)
    plt.xlabel("Neighbor coupling λ")
    plt.ylabel("Prefactor A")
    plt.title("Prefactor Flow: A(λ)")
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, alpha_r2, "o-", linewidth=2, label="Power-law fit R²")
    plt.plot(lambdas, k_r2, "s--", linewidth=2, label="Entropy fit R²")
    plt.xlabel("Neighbor coupling λ")
    plt.ylabel("R²")
    plt.title("Fit Quality vs Coupling")
    plt.legend()
    plt.tight_layout()

    plt.show()


def print_summary_table(sweep_rows):
    print("\n" + "=" * 92)
    print("COUPLING SWEEP SUMMARY")
    print("=" * 92)
    header = f"{'lambda':>8} | {'A':>10} | {'alpha':>10} | {'R2(alpha)':>10} | {'B':>10} | {'k':>10} | {'R2(k)':>10}"
    print(header)
    print("-" * len(header))

    for row in sweep_rows:
        print(
            f"{row['lambda']:8.4f} | "
            f"{row['A']:10.4f} | "
            f"{row['alpha']:10.4f} | "
            f"{row['alpha_r2']:10.4f} | "
            f"{row['B']:10.4f} | "
            f"{row['k']:10.4f} | "
            f"{row['k_r2']:10.4f}"
        )

    print("=" * 92)
    print()


def main():
    lambda_values = np.array([0.00, 0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20])

    sweep_rows, phase_details, entropy_details = run_coupling_sweep(
        lambda_values=lambda_values,
        shape=(64, 64),
        steps=700,
        dt=0.01,
    )

    print_summary_table(sweep_rows)
    plot_coupling_sweep(sweep_rows)

    first = sweep_rows[0]
    last = sweep_rows[-1]

    print("Working effective-law view:")
    print(
        f"  low coupling  (lambda={first['lambda']:.2f}) -> alpha≈{first['alpha']:.4f}, k≈{first['k']:.4f}"
    )
    print(
        f"  high coupling (lambda={last['lambda']:.2f}) -> alpha≈{last['alpha']:.4f}, k≈{last['k']:.4f}"
    )
    print("\nCandidate form:")
    print("  inj_min(C,S; λ) = A(λ) * C^{α(λ)} * exp(k(λ) * S)")


if __name__ == "__main__":
    main()
