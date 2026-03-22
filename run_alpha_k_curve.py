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
    gv = np.clip(gv, 0, 1)

    constraint = C + 0.10 * rng.standard_normal(gv.shape)
    constraint = np.clip(constraint, 0, None)

    entropy = 0.5 + 0.2 * rng.standard_normal(gv.shape)
    entropy = np.clip(entropy, 0, None)

    for _ in range(steps):
        lap = laplacian_random(gv, k, rng)
        collapsed = gv < 0.35

        stable = (
            -1.2 * constraint * (1 - gv)
            + 0.65 * inj
            - 0.08 * entropy
            + lam * lap
        )

        collapse = (
            2.5 * (0 - gv)
            - 0.2 * entropy
            + lam * lap
        )

        gv = gv + 0.01 * np.where(collapsed, collapse, stable)
        gv = np.clip(gv, 0, 1)

    return (np.mean(gv < 0.35) < 0.4) and (np.mean(gv) > 0.4)


def find_inj_min_for_seed(C, k, seed):
    inj_vals = np.linspace(0.01, 2.0, 60)

    for j, inj in enumerate(inj_vals):
        survives = simulate(C, k, inj, seed=seed + j * 1000)
        if survives:
            return inj

    return np.nan


def find_inj_min_avg(C, k, base_seed):
    seeds = [base_seed, base_seed + 17, base_seed + 33]
    vals = []

    for s in seeds:
        v = find_inj_min_for_seed(C, k, s)
        if np.isfinite(v):
            vals.append(v)

    if not vals:
        return np.nan

    return np.mean(vals)


def extract_alpha(k):
    C_vals = np.linspace(0.2, 2.0, 10)
    pts = []

    for i, C in enumerate(C_vals):
        inj = find_inj_min_avg(C, k, base_seed=1000 + 97*k + 13*i)
        if np.isfinite(inj) and inj > 0:
            pts.append((C, inj))

    C_arr = np.array([p[0] for p in pts])
    inj_arr = np.array([p[1] for p in pts])

    logC = np.log(C_arr)
    logInj = np.log(inj_arr)

    alpha, intercept = np.polyfit(logC, logInj, 1)

    pred = alpha * logC + intercept
    ss_res = np.sum((logInj - pred) ** 2)
    ss_tot = np.sum((logInj - np.mean(logInj)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return alpha, r2, pts


def main():
    print("\nRunning alpha(k) extraction...\n")

    for k in [2, 4, 6, 8, 10, 12, 14]:
        alpha, r2, pts = extract_alpha(k)

        print(f"\nk={k} -> alpha ≈ {alpha:.4f}, R² ≈ {r2:.4f}")
        print("  points:", [(round(c,2), round(i,4)) for c,i in pts])

    print("\nDone.\n")


if __name__ == "__main__":
    main()
