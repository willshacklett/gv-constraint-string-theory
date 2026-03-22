import numpy as np
import csv


# =========================
# CONFIG
# =========================
GRID_SIZE = 40


# =========================
# KERNEL
# =========================
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_influence(lam):
    center = GRID_SIZE // 2
    total = 0.0

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            r = distance(center, center, i, j)
            total += np.exp(-r / lam)

    return total


# =========================
# LAMBDA SWEEP
# =========================
def run_lambda_sweep():
    lambda_vals = np.linspace(0.15, 1.2, 15)

    results = []

    print("\n=== INFLUENCE(λ) CURVE ===\n")

    for lam in lambda_vals:
        infl = compute_influence(lam)
        results.append((lam, infl))

        print(f"λ = {lam:.3f} → influence ≈ {infl:.4f}")

    # save
    with open("lambda_influence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "influence"])
        writer.writerows(results)

    print("\nSaved: lambda_influence.csv")

    return results


# =========================
# LOAD PHASE SURFACE
# =========================
def load_phase_surface():
    rows = []

    try:
        with open("gv_phase_surface.csv", "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except:
        print("\n⚠️ gv_phase_surface.csv not found (skip surface stats)")
        return []

    print(f"\nLoaded {len(rows)} surface rows")

    return rows


# =========================
# SIMPLE SURFACE STATS
# =========================
def summarize_surface(rows):
    if not rows:
        return

    inj_vals = []

    for r in rows:
        try:
            inj = float(r["inj_min"])
            inj_vals.append(inj)
        except:
            continue

    inj_vals = np.array(inj_vals)

    print("\n=== SURFACE STATS ===")
    print(f"count = {len(inj_vals)}")
    print(f"mean  = {np.mean(inj_vals):.4f}")
    print(f"std   = {np.std(inj_vals):.4f}")
    print(f"min   = {np.min(inj_vals):.4f}")
    print(f"max   = {np.max(inj_vals):.4f}")


# =========================
# MAIN
# =========================
def main():
    lambda_results = run_lambda_sweep()

    rows = load_phase_surface()
    summarize_surface(rows)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
