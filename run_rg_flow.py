import numpy as np
import csv
import matplotlib.pyplot as plt

# ---------------------------------
# Try to load epsilon data from CSV
# ---------------------------------
lambda_c = []
epsilon = []

with open("epsilon_surface.csv", "r") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []

    # Handle either "lambda" or "lambda_c"
    lambda_key = "lambda_c" if "lambda_c" in fieldnames else "lambda"

    for row in reader:
        lambda_c.append(float(row[lambda_key]))
        epsilon.append(float(row["epsilon"]))

lambda_c = np.array(lambda_c)
epsilon = np.array(epsilon)

# ---------------------------------
# Locked exponential fit constants
# ---------------------------------
alpha = 1.285511
beta = -2.091392
gamma = -3.448006

# ---------------------------------
# RG-like flow from epsilon kernel
# epsilon(lambda) = alpha * exp(-beta * lambda) + gamma
# ---------------------------------
def flow_fn(l):
    return alpha * np.exp(-beta * l) + gamma

lambda_grid = np.linspace(lambda_c.min(), lambda_c.max(), 200)
flow = flow_fn(lambda_grid)

# ---------------------------------
# Estimate fixed point where flow ~ 0
# ---------------------------------
idx = np.argmin(np.abs(flow))
lambda_fixed = lambda_grid[idx]
flow_fixed = flow[idx]

# ---------------------------------
# Save sampled flow
# ---------------------------------
with open("rg_flow.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda_c", "flow"])
    for l, y in zip(lambda_grid, flow):
        writer.writerow([l, y])

# ---------------------------------
# Plot
# ---------------------------------
plt.figure(figsize=(8, 5))
plt.plot(lambda_grid, flow, label="RG flow")
plt.scatter(lambda_c, epsilon, label="epsilon data")
plt.axhline(0.0, linestyle="--")
plt.axvline(lambda_fixed, linestyle=":")
plt.xlabel("lambda_c")
plt.ylabel("epsilon / flow")
plt.title("GV-CST RG Flow Toward Fixed Point")
plt.legend()
plt.tight_layout()
plt.savefig("rg_flow.png", dpi=200)
plt.close()

# ---------------------------------
# Terminal summary
# ---------------------------------
print("\n=== RG FLOW ===")
print(f"alpha ≈ {alpha:.6f}")
print(f"beta  ≈ {beta:.6f}")
print(f"gamma ≈ {gamma:.6f}")
print(f"\nEstimated fixed point: lambda_c ≈ {lambda_fixed:.6f}, flow ≈ {flow_fixed:.6f}")

print("\nSaved:")
print("- rg_flow.csv")
print("- rg_flow.png")
print("\nDone.\n")
