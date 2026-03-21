import numpy as np
import matplotlib.pyplot as plt


def laplacian_neumann(x):
    return (
        np.roll(x, 1, 0)
        + np.roll(x, -1, 0)
        + np.roll(x, 1, 1)
        + np.roll(x, -1, 1)
        - 4 * x
    )


def laplacian_moore(x):
    shifts = [
        (1,0),(-1,0),(0,1),(0,-1),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]
    total = sum(np.roll(np.roll(x, dx, 0), dy, 1) for dx, dy in shifts)
    return total - 8 * x


def laplacian_random(x, neighbors=6, rng=None):
    h, w = x.shape
    lap = np.zeros_like(x)

    for _ in range(neighbors):
        dx = rng.integers(-1, 2)
        dy = rng.integers(-1, 2)
        lap += np.roll(np.roll(x, dx, 0), dy, 1)

    return lap - neighbors * x


def simulate(lap_fn, degree, C, inj, size=64, steps=600, lam=0.08, seed=42):
    rng = np.random.default_rng(seed)

    gv = 0.95 + 0.05 * rng.standard_normal((size, size))
    gv = np.clip(gv, 0, 1)

    constraint = C + 0.1 * rng.standard_normal(gv.shape)
    constraint = np.clip(constraint, 0, None)

    entropy = 0.5 + 0.2 * rng.standard_normal(gv.shape)
    entropy = np.clip(entropy, 0, None)

    for _ in range(steps):
        lap = lap_fn(gv)

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

        gv = gv + np.where(collapsed, collapse, stable) * 0.01
        gv = np.clip(gv, 0, 1)

    survives = (np.mean(gv < 0.35) < 0.4) and (np.mean(gv) > 0.4)
    return survives


def extract_alpha(lap_fn, degree):
    C_vals = np.linspace(0.1, 2.5, 25)
    inj_vals = np.linspace(0.01, 2.0, 30)

    pts = []

    for C in C_vals:
        for inj in inj_vals:
            if simulate(lap_fn, degree, C, inj):
                pts.append((C, inj))
                break

    C_arr = np.array([p[0] for p in pts])
    inj_arr = np.array([p[1] for p in pts])

    logC = np.log(C_arr)
    loginj = np.log(inj_arr)

    alpha, _ = np.polyfit(logC, loginj, 1)
    return alpha


def main():
    results = []

    print("Running connectivity sweep...\n")

    # Neumann (k=4)
    alpha = extract_alpha(laplacian_neumann, 4)
    results.append((4, alpha))
    print(f"Neumann (k=4): alpha ≈ {alpha:.4f}")

    # Moore (k=8)
    alpha = extract_alpha(laplacian_moore, 8)
    results.append((8, alpha))
    print(f"Moore (k=8): alpha ≈ {alpha:.4f}")

    # Random degrees
    rng = np.random.default_rng(123)

    for k in [2, 6, 10, 14]:
        lap_fn = lambda x, k=k, rng=rng: laplacian_random(x, neighbors=k, rng=rng)
        alpha = extract_alpha(lap_fn, k)
        results.append((k, alpha))
        print(f"Random (k={k}): alpha ≈ {alpha:.4f}")

    # Plot
    ks = [r[0] for r in results]
    alphas = [r[1] for r in results]

    plt.plot(ks, alphas, "o-")
    plt.xlabel("Connectivity (k)")
    plt.ylabel("Alpha")
    plt.title("Alpha vs Connectivity")
    plt.show()

    print("\nFinal Results:")
    for k, a in results:
        print(f"k={k} → alpha ≈ {a:.4f}")


if __name__ == "__main__":
    main()
