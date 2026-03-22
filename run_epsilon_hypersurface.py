import numpy as np
import csv
import matplotlib.pyplot as plt

# -----------------------------
# Locked values from your runs
# -----------------------------
A = 1.70

lambda_c_values = np.array([0.4964, 0.4812, 0.4716, 0.4624, 0.4492])
k_c_values = np.array([3.7919, 3.6758, 3.6023, 3.5317, 3.4308])

# Proxy C,S axes for now
# These can be replaced later with true extracted slices
C_values = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
S_values = np.array([0.80, 0.90, 1.00, 1.10, 1.20])

# -----------------------------
# Compute epsilon
# -----------------------------
products = k_c_values * lambda_c_values
epsilon_values = products - A

# -----------------------------
# Save pointwise hypersurface CSV
# -----------------------------
with open("epsilon_surface.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["C", "S", "lambda_c", "k_c", "product", "epsilon"])
    for i in range(len(lambda_c_values)):
        writer.writerow([
            C_values[i],
            S_values[i],
            lambda_c_values[i],
            k_c_values[i],
            products[i],
            epsilon_values[i]
        ])

# -----------------------------
# Build a smooth grid
# -----------------------------
C_grid = np.linspace(C_values.min(), C_values.max(), 80)
S_grid = np.linspace(S_values.min(), S_values.max(), 80)
CC, SS = np.meshgrid(C_grid, S_grid)

# Simple plane fit epsilon(C,S) = a*C + b*S + c
X = np.column_stack([C_values, S_values, np.ones_like(C_values)])
coeffs, _, _, _ = np.linalg.lstsq(X, epsilon_values, rcond=None)
a, b, c = coeffs

epsilon_grid = a * CC + b * SS + c

# -----------------------------
# Save gridded heatmap data
# -----------------------------
with open("epsilon_heatmap_grid.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["C", "S", "epsilon"])
    for i in range(CC.shape[0]):
        for j in range(CC.shape[1]):
            writer.writerow([CC[i, j], SS[i, j], epsilon_grid[i, j]])

# -----------------------------
# Plot heatmap
# -----------------------------
plt.figure(figsize=(8, 6))
im = plt.imshow(
    epsilon_grid,
    origin="lower",
    aspect="auto",
    extent=[C_grid.min(), C_grid.max(), S_grid.min(), S_grid.max()]
)
plt.colorbar(im, label="epsilon")
plt.scatter(C_values, S_values, c=epsilon_values, edgecolors="white", s=100)
plt.xlabel("C")
plt.ylabel("S")
plt.title("GV-CST Epsilon Hypersurface")
plt.tight_layout()
plt.savefig("epsilon_hypersurface.png", dpi=200)
plt.close()

# -----------------------------
# Plot epsilon vs lambda_c
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(lambda_c_values, epsilon_values, marker="o")
plt.xlabel("lambda_c")
plt.ylabel("epsilon")
plt.title("GV-CST Residual vs Propagation Depth")
plt.tight_layout()
plt.savefig("epsilon_vs_lambda.png", dpi=200)
plt.close()

# -----------------------------
# Terminal summary
# -----------------------------
print("\n=== EPSILON HYPERSURFACE ===\n")
print(f"Plane fit: epsilon(C,S) ≈ {a:.4f}*C + {b:.4f}*S + {c:.4f}")
print("\nPoint values:")
for i in range(len(epsilon_values)):
    print(
        f"C={C_values[i]:.2f}, "
        f"S={S_values[i]:.2f}, "
        f"lambda_c={lambda_c_values[i]:.4f}, "
        f"epsilon={epsilon_values[i]:.6f}"
    )

print("\nSaved files:")
print("- epsilon_surface.csv")
print("- epsilon_heatmap_grid.csv")
print("- epsilon_hypersurface.png")
print("- epsilon_vs_lambda.png")
print("\nDone.\n")
