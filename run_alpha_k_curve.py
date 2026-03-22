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


def simulate(C, k, inj, size=40, steps=200, lam=0.08, seed=42):
    rng = np.random.default_rng(seed)

    gv = 0.95 + 0.05 * rng.standard_normal((size, size))
    gv = np.clip(gv, 0, 1)

    constraint = C + 0.1 * rng.standard_normal(gv.shape)
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

    survives = (np.mean(gv < 0.35) < 0.4) and (np.mean(gv) > 0.4)
    return survives


def extract_alpha(k):
    C_vals = np.linspace(0.2, 2.0, 10)
    inj_vals = np.linspace(0.01, 2.0, 50)

    points = []

    for C in C_vals:
        for i, inj in enumerate(inj_vals):
            if simulate(C, k, inj, seed=42 + i):
                points.append((C, inj))
                break

    C_arr = np.array([p[0] for p in points])
    inj_arr = np.array([p[1] for p in points])

    logC = np.log(C_arr)
    loginj = np.log(inj_arr)

    alpha, _ = np.polyfit(logC, loginj, 1)
    return alpha


def main():
    k_vals = [2, 4, 6, 8, 10, 12, 14]
    alphas = []

    print("\nRunning alpha(k) extraction...\n")

    for k in k_vals:
        alpha = extract_alpha(k)
        alphas.append(alpha)
        print(f"k={k} -> alpha ≈ {alpha:.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
