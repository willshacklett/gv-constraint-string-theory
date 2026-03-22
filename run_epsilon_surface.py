import numpy as np
import csv

# === Base values from your runs ===
A = 1.70

# Your observed data (from duality sweep)
lambda_c_values = np.array([0.4964, 0.4812, 0.4716, 0.4624, 0.4492])
k_c_values = np.array([3.7919, 3.6758, 3.6023, 3.5317, 3.4308])

# Simulated C,S axes (proxy dimensions for now)
C_values = np.linspace(0.8, 1.2, len(lambda_c_values))
S_values = np.linspace(0.8, 1.2, len(lambda_c_values))

# === Compute epsilon ===
products = k_c_values * lambda_c_values
epsilon = products - A

# === Save hypersurface ===
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
            epsilon[i]
        ])

print("\n=== EPSILON SURFACE GENERATED ===")
for i in range(len(lambda_c_values)):
    print(f"C={C_values[i]:.3f}, S={S_values[i]:.3f}, λ={lambda_c_values[i]:.4f}, ε={epsilon[i]:.6f}")

print("\nSaved: epsilon_surface.csv")
