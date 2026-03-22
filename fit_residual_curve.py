import numpy as np
from scipy.optimize import curve_fit

# Data from your run (copy exact values)
lambda_c = np.array([0.4964, 0.4812, 0.4716, 0.4624, 0.4492])
residual = np.array([0.1824, 0.0689, -0.0011, -0.0671, -0.1591])

def model(x, a, b):
    return a * x + b  # start simple (linear)

params, _ = curve_fit(model, lambda_c, residual)

a, b = params

print("\n=== RESIDUAL FIT ===")
print(f"a ≈ {a:.4f}")
print(f"b ≈ {b:.4f}")

pred = model(lambda_c, a, b)
rmse = np.sqrt(np.mean((pred - residual)**2))

print(f"RMSE ≈ {rmse:.6f}")

print("\nInterpretation:")
print("If fit is strong → epsilon is deterministic propagation term")

print("\nDone.\n")
