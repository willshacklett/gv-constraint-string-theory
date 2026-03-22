import numpy as np

grid_size = 40

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def compute_effective_influence(lambda_prop):
    center = grid_size // 2
    total = 0.0

    for i in range(grid_size):
        for j in range(grid_size):
            r = distance(center, center, i, j)
            total += np.exp(-r / lambda_prop)

    return total

def alpha_from_k(k):
    return 0.9613 - 0.1630 * np.exp(-0.2776 * k)

def main():
    lambda_vals = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00])

    print("\n=== LAMBDA TO K BRIDGE ===\n")

    influences = []
    alpha_vals = []

    for lam in lambda_vals:
        infl = compute_effective_influence(lam)
        influences.append(infl)

        alpha_eff = alpha_from_k(infl)
        alpha_vals.append(alpha_eff)

        print(
            f"lambda = {lam:.3f} | "
            f"influence ≈ {infl:.4f} | "
            f"alpha_eff ≈ {alpha_eff:.4f}"
        )

    influences = np.array(influences)
    log_infl = np.log(influences)

    b, log_a = np.polyfit(lambda_vals, log_infl, 1)
    a = np.exp(log_a)

    infl_pred = a * np.exp(b * lambda_vals)

    ss_res = np.sum((influences - infl_pred) ** 2)
    ss_tot = np.sum((influences - np.mean(influences)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    print("\n--- Influence Fit ---")
    print(f"influence(lambda) ≈ {a:.4f} * exp({b:.4f} * lambda)")
    print(f"R^2 ≈ {r2:.4f}")

    print("\n--- Interpretation ---")
    print("k_eff ≈ influence(lambda)")
    print("alpha driven by propagation reach (lambda), not raw topology")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
