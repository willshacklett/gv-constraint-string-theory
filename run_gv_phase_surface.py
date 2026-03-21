import numpy as np
import csv


def laplacian_random(x, neighbors, rng):
    lap = np.zeros_like(x)

    for _ in range(neighbors):
        dx = rng.integers(-1, 2)
        dy = rng.integers(-1, 2)

        # avoid zero-shift so a "neighbor" is actually a neighbor
        while dx == 0 and dy == 0:
            dx = rng.integers(-1, 2)
            dy = rng.integers(-1, 2)

        lap += np.roll(np.roll(x, dx, axis=0), dy, axis=1)

    return lap - neighbors * x


def simulate(C, S, k, inj, size=64, steps=500, lam=0.08, seed=42):
    rng = np.random.default_rng(seed)

    gv = 0.95 + 0.05 * rng.standard_normal((size, size))
    gv = np.clip(gv, 0.0, 1.0)

    constraint = C + 0.1 * rng.standard_normal(gv.shape)
    constraint = np.clip(constraint, 0.0, None)

    entropy = S + 0.2 * rng.standard_normal(gv.shape)
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

        dgv = np.where(collapsed, collapse, stable)
        gv = gv + 0.01 * dgv
        gv = np.clip(gv, 0.0, 1.0)

    survives = (np.mean(gv < 0.35) < 0.4) and (np.mean(gv) > 0.4)
    return survives


def find_inj_min(C, S, k, size=64, steps=500, lam=0.08, seed=42):
    inj_vals = np.linspace(0.01, 2.0, 30)

    for idx, inj in enumerate(inj_vals):
        survives = simulate(
            C=C,
            S=S,
            k=k,
            inj=inj,
            size=size,
            steps=steps,
            lam=lam,
            seed=seed + idx,
        )
        if survives:
            return inj

    return np.nan


def main():
    C_vals = np.linspace(0.2, 2.0, 10)
    S_vals = np.linspace(0.0, 2.0, 8)
    k_vals = [2, 4, 6, 8, 10]

    rows = []

    print("Running phase surface sweep...\n")

    for k in k_vals:
        print(f"k = {k}")
        for C in C_vals:
            for S in S_vals:
                inj_min = find_inj_min(
                    C=C,
                    S=S,
                    k=k,
                    size=64,
                    steps=500,
                    lam=0.08,
                    seed=1000 + int(100 * C) + int(100 * S) + 10 * k,
                )
                rows.append([C, S, k, inj_min])
                print(f"  C={C:.2f}, S={S:.2f}, k={k} -> inj_min={inj_min:.4f}")

    with open("gv_phase_surface.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "S", "k", "inj_min"])
        writer.writerows(rows)

    print("\nSaved: gv_phase_surface.csv")


if __name__ == "__main__":
    main()
