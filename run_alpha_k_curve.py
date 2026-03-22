import numpy as np


def laplacian_random(x, neighbors, rng):
    lap = np.zeros_like(x)

    for _ in range(neighbors):
        dx = rng.integers(-1, 2)
        dy = rng.integers(-1, 2)

        while dx == 0 and dy == 0:
            dx = rng.integers(-1, 2)
            dy = rng.integers(-1, 2)

        lap += np.roll(np.roll(x, dx, axis=0), dy, axis=1)

    return lap - neighbors * x


def simulate(C, k, inj, size=40, steps=220, lam=0.08, seed=42):
    rng = np.random.default_rng(seed)

    gv = 0.95 + 0.05 * rng.standard_normal((size, size))
    gv = np.clip(gv, 0.0, 1.0)

    constraint = C + 0.10 * rng.standard_normal(gv.shape)
    constraint = np.clip(constraint, 0.0, None)

    entropy = 0.5 + 0.2 * rng.standard_normal(gv.shape)
    entropy = np.clip(entropy, 0.0, None)

    for _ in range(steps):
        lap = laplacian_random(gv, k, rng)
        collapsed = gv < 0.35

        stable = (
            -1.2 * constraint * (1.0 - gv)
            + 0.65 * inj
            - 0.08 * entropy
            + lam * lap
        )

        collapse = (
            2.5 * (0.0 - gv)
            - 0.2 * entropy
            + lam * lap
        )

        gv = gv + 0.01 * np.where(collapsed, collapse, stable)
        gv = np.clip(gv, 0.0, 1.0)

    survives = (np.mean(gv < 0.35) < 0.4) and (np.mean(gv) > 0.4)
    return survives


def find_inj_min_for_seed(C, k, seed):
    inj_vals = np.linspace(0.01, 2.0, 60)

    for j, inj in enumerate(inj_vals):
        survives = simulate(
            C=C,
            k=k,
            inj=inj,
            size=40,
            steps=220,
            lam=0.08,
            seed=seed + 1000 * j,
        )
        if survives:
            return inj

    return np.nan


def find_inj_min_avg(C, k, base_seed=100):
    seed_list = [base_seed + 11, base_seed + 29, base_seed + 47]
    vals = []

    for s in seed_list:
        inj_min = find_inj_min_for_seed(C, k, s)
        if np.isfinite(inj_min):
            vals.append(inj_min)

    if not vals:
        return np.nan

    return float(np.mean(vals))


def extract_alpha(k):
    C_vals = np.linspace(0.2, 2.0, 10)
    points = []

    for i, C in enumerate(C_vals):
        inj_avg = find_inj_min_avg(C, k, base_seed=5000 + 137 * k + 23 * i)
        if np.isfinite(inj_avg) and inj_avg > 0:
            points.append((C, inj_avg))

    if len(points) < 4:
        raise RuntimeError(f"Not enough valid boundary points for k={k}")

    C_arr = np.array([p[0] for p in points], dtype=float)
    inj_arr = np.array([p[1] for p in points], dtype=float)

    logC = np.log(C_arr)
    logInj = np.log(inj_arr)

    alpha, intercept = np.polyfit(logC, logInj, 1)

    pred = alpha * logC + intercept
    ss_res = np.sum((logInj - pred) ** 2)
    ss_tot = np.sum((logInj - np.mean(logInj)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return alpha, r2, points


def main():
    print("\nRunning alpha(k) extraction...\n")

    k_vals = [2, 4, 6, 8, 10, 12, 14]

    for k in k_vals:
        alpha, r2, points = extract_alpha(k)
        print(f"k={k} -> alpha ≈ {alpha:.4f}, R^2 ≈ {r2:.4f}")
        print(f"  boundary points: {[(round(c, 2), round(i, 4)) for c, i in points]}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
