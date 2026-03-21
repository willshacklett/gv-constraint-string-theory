import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Your measured data (from sweep)
k_vals = np.array([2, 4, 6, 8, 10, 14])
alpha_vals = np.array([0.8660, 0.9146, 0.9223, 0.9454, 0.9520, 0.9594])


# Model: exponential saturation
def alpha_model(k, alpha_max, A, b):
    return alpha_max - A * np.exp(-b * k)


# Fit
params, _ = curve_fit(alpha_model, k_vals, alpha_vals, p0=[1.0, 0.2, 0.3])

alpha_max, A, b = params


# Generate smooth curve
k_smooth = np.linspace(1, 15, 100)
alpha_fit = alpha_model(k_smooth, *params)


# Plot
plt.scatter(k_vals, alpha_vals, label="Data")
plt.plot(k_smooth, alpha_fit, label="Fit", linewidth=2)

plt.xlabel("Connectivity (k)")
plt.ylabel("Alpha")
plt.title("Alpha vs Connectivity (Fit)")
plt.legend()
plt.show()


# Print results
print("\nFitted parameters:")
print(f"alpha_max ≈ {alpha_max:.4f}")
print(f"A ≈ {A:.4f}")
print(f"b ≈ {b:.4f}")

print("\nModel:")
print("alpha(k) ≈ alpha_max - A * exp(-b k)")
