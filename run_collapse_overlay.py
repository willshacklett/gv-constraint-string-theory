import numpy as np
import matplotlib.pyplot as plt

# Load your data
data = np.genfromtxt("epsilon_surface.csv", delimiter=",", names=True)

lambda_c = data["lambda"]
epsilon = data["epsilon"]

# Your fitted constants (update if needed)
A = 1.70
alpha = 1.2855
beta = 2.0913
gamma = -3.4480

# Collapse transformation
collapsed = epsilon - (alpha * np.exp(-beta * lambda_c) + gamma)

plt.figure(figsize=(8,5))

# Original
plt.scatter(lambda_c, epsilon, label="epsilon (data)")

# Collapse curve
plt.scatter(lambda_c, collapsed, label="collapsed residual")

# Ideal line (should flatten if collapse works)
plt.axhline(0, linestyle="--")

plt.xlabel("lambda_c")
plt.ylabel("value")
plt.title("Collapse Overlay Test")
plt.legend()
plt.tight_layout()

plt.savefig("collapse_overlay.png", dpi=200)
print("Saved: collapse_overlay.png")
