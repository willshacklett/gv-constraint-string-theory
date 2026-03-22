import numpy as np

# -----------------------------
# Parameters
grid_size = 32
lambdas = [0.2, 0.3, 0.4, 0.5]  # propagation lengths

# -----------------------------
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# -----------------------------
def compute_effective_influence(lambda_prop):
    center = grid_size // 2
    total = 0.0

    for i in range(grid_size):
        for j in range(grid_size):
            r = distance(center, center, i, j)
            weight = np.exp(-r / lambda_prop)
            total += weight

    return total

# -----------------------------
print("\n=== KERNEL VALIDATION ===\n")

for lam in lambdas:
    influence = compute_effective_influence(lam)
    print(f"lambda = {lam:.3f} → total influence ≈ {influence:.4f}")
