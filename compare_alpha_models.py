import numpy as np
from scipy.optimize import curve_fit

# Your measured data (replace with your actual extracted values if needed)
k_vals = np.array([2, 4, 6, 8, 10, 12, 14])
alpha_vals = np.array([
    0.8678, 0.9074, 0.9304, 0.9438, 0.9515, 0.955, 0.958
])

# -----------------------------
# Model 1: Exponential saturation
def exp_model(k, alpha_max, A, b):
    return alpha_max - A * np.exp(-b * k)

# Model 2: Spectral-gap / rational
def rational_model(k, c):
    return 1 - 1 / (1 + c * k)

# -----------------------------
# Fit exponential
popt_exp, _ = curve_fit(exp_model, k_vals, alpha_vals, p0=[1.0, 0.2, 0.2])
alpha_exp_fit = exp_model(k_vals, *popt_exp)

# Fit rational
popt_rat, _ = curve_fit(rational_model, k_vals, alpha_vals, p0=[0.2])
alpha_rat_fit = rational_model(k_vals, *popt_rat)

# -----------------------------
# Compute errors
def rmse(y, yfit):
    return np.sqrt(np.mean((y - yfit) ** 2))

rmse_exp = rmse(alpha_vals, alpha_exp_fit)
rmse_rat = rmse(alpha_vals, alpha_rat_fit)

# -----------------------------
print("\n=== MODEL COMPARISON ===\n")

print("Exponential model:")
print(f"alpha_max ≈ {popt_exp[0]:.4f}, A ≈ {popt_exp[1]:.4f}, b ≈ {popt_exp[2]:.4f}")
print(f"RMSE ≈ {rmse_exp:.6f}")

print("\nRational (spectral-gap) model:")
print(f"c ≈ {popt_rat[0]:.4f}")
print(f"RMSE ≈ {rmse_rat:.6f}")

# -----------------------------
# Winner
print("\n--- RESULT ---")
if rmse_exp < rmse_rat:
    print("Exponential model fits better → empirical saturation dominates.")
else:
    print("Rational model fits better → topology/spectral origin confirmed.")
