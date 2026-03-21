import numpy as np
import matplotlib.pyplot as plt


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


def simulate(dim, size, C, inj, steps=500, lam=0.08):
    if dim == 1:
        gv = np.ones(size)
        lap_fn = laplacian_1d
    elif dim == 2:
        gv = np.ones((size, size))
        lap_fn = laplacian_2d
    else:
        gv = np.ones((size, size, size))
        lap_fn = laplacian_3d

    entropy = 0.5

    for _ in range(steps):
        lap = lap_fn(gv)

        stable = -1.2 * C * (1 - gv) + 0.65 * inj - 0.08 * entropy + lam * lap
        collapse = 2.5 * (0 - gv) - 0.2 * entropy + lam * lap

        gv = np.where(gv < 0.35, gv + collapse * 0.01, gv + stable * 0.01)
        gv = np.clip(gv, 0, 1)

    survives = np.mean(gv) > 0.4
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

    alpha, logA = np.polyfit(logC, loginj, 1)
    return alpha


def main():
    dims = [1, 2, 3]
    sizes = {1: 128, 2: 64, 3: 24}

    results = []

    for d in dims:
        print(f"Running {d}D...")
        alpha = extract_alpha(d, sizes[d])
        results.append((d, alpha))
        print(f"  alpha ≈ {alpha:.4f}")

    dims_arr = [r[0] for r in results]
    alpha_arr = [r[1] for r in results]

    plt.plot(dims_arr, alpha_arr, "o-")
    plt.xlabel("Dimension")
    plt.ylabel("Alpha")
    plt.title("Alpha vs Dimension")
    plt.show()

    print("\nResults:")
    for d, a in results:
        print(f"{d}D → alpha ≈ {a:.4f}")


if __name__ == "__main__":
    main()
