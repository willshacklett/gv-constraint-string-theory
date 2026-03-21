import numpy as np


def laplacian_1d(x):
    return np.roll(x, 1) + np.roll(x, -1) - 2 * x


def laplacian_2d(x):
    return (
        np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
        - 4 * x
    )


def laplacian_3d(x):
    return (
        np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
        + np.roll(x, 1, axis=2)
        + np.roll(x, -1, axis=2)
        - 6 * x
    )


def simulate(dim, size, C, inj, steps=600, lam=0.08, seed=42):
    rng = np.random.default_rng(seed)

    # Initialize GV field
    if dim == 1:
        gv = 0.95 + 0.05 * rng.standard_normal(size)
        lap_fn = laplacian_1d
    elif dim == 2:
        gv = 0.95 + 0.05 * rng.standard_normal((size, size))
        lap_fn = laplacian_2d
    else:
        gv = 0.95 + 0.05 * rng.standard_normal((size, size, size))
        lap_fn = laplacian_3d

    gv = np.clip(gv, 0, 1)

    # Stronger heterogeneous constraint
    constraint = C + 0.10 * rng.standard_normal(gv.shape)
    constraint = np.clip(constraint, 0, None)

    # ENTROPY AS FIELD (key change)
    entropy = 0.5 + 0.2 * rng.standard_normal(gv.shape)
    entropy = np.clip(entropy, 0, None)

    # Noisy coupling field
    coupling_noise = 1.0 + 0.2 * rng.standard_normal(gv.shape)

    for _ in range(steps):
        lap = lap_fn(gv)

        collapsed = gv < 0.35

        stable = (
            -1.2 * constraint * (1 - gv)
            + 0.65 * inj
            - 0.08 * entropy
            + lam * coupling_noise * lap
        )

        collapse = (
            2.5 * (0 - gv)
            - 0.2 * entropy
            + lam * coupling_noise * lap
        )

        gv = gv + np.where(collapsed, collapse, stable) * 0.01
        gv = np.clip(gv, 0, 1)

    collapse_fraction = np.mean(gv < 0.35)
    mean_gv = np.mean(gv)

    survives = (collapse_fraction < 0.4) and (mean_gv > 0.4)

    return survives


def extract_alpha(dim, size):
    C_vals = np.linspace(0.1, 2.5, 25)
    inj_vals = np.linspace(0.01, 2.0, 30)

    points = []

    for C in C_vals:
        for inj in inj_vals:
            if simulate(dim, size, C, inj):
                points.append((C, inj))
                break

    C_arr = np.array([p[0] for p in points])
    inj_arr = np.array([p[1] for p in points])

    logC = np.log(C_arr)
    loginj = np.log(inj_arr)

    alpha, _ = np.polyfit(logC, loginj, 1)
    return alpha


def main():
    dims = [1, 2, 3]
    sizes = {1: 128, 2: 64, 3: 24}

    print("Running HETEROGENEOUS dimension sweep...\n")

    results = []

    for d in dims:
        print(f"{d}D...")
        alpha = extract_alpha(d, sizes[d])
        results.append((d, alpha))
        print(f"  alpha ≈ {alpha:.4f}\n")

    print("Final Results:")
    for d, a in results:
        print(f"{d}D → alpha ≈ {a:.4f}")


if __name__ == "__main__":
    main()
