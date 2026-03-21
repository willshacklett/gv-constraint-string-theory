import numpy as np
import csv


# α(k) from your fit
def alpha_k(k):
    return 0.9623 - 0.1627 * np.exp(-0.2715 * k)


# Load data
C_vals = []
S_vals = []
k_vals = []
inj_vals = []

with open("gv_phase_surface.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        C_vals.append(float(row["C"]))
        S_vals.append(float(row["S"]))
        k_vals.append(float(row["k"]))
        inj_vals.append(float(row["inj_min"]))

C_vals = np.array(C_vals)
S_vals = np.array(S_vals)
k_vals = np.array(k_vals)
inj_vals = np.array(inj_vals)


# Remove NaNs
mask = ~np.isnan(inj_vals)
C_vals = C_vals[mask]
S_vals = S_vals[mask]
k_vals = k_vals[mask]
inj_vals = inj_vals[mask]


# Build model without A0
model_base = (C_vals ** alpha_k(k_vals)) * np.exp(0.89 * S_vals)


# Fit A0 (least squares)
A0 = np.sum(inj_vals * model_base) / np.sum(model_base ** 2)


# Predictions
pred = A0 * model_base


# Errors
abs_err = np.abs(pred - inj_vals)
rel_err = abs_err / inj_vals

rmse = np.sqrt(np.mean((pred - inj_vals) ** 2))
mean_rel = np.mean(rel_err)
max_rel = np.max(rel_err)


print("\n===== GV SURFACE FIT =====\n")
print(f"A0 ≈ {A0:.4f}")
print(f"RMSE ≈ {rmse:.5f}")
print(f"Mean relative error ≈ {mean_rel:.4f}")
print(f"Max relative error ≈ {max_rel:.4f}")

print("\nModel:")
print("inj_min ≈ A0 · C^{α(k)} · exp(0.89 S)")
