import numpy as np
import csv
from scipy.optimize import curve_fit

# -------------------------
# Load data
# -------------------------
lambdas = []
influences = []

with open("lambda_influence.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lambdas.append(float(row["lambda"]))
        influences.append(float(row["influence"]))

lambdas = np.array(lambdas)
influences = np.array(influences)

# -------------------------
# Exponential model
# -------------------------
def model(l, a, b):
    return a * np.exp(b * l)

params, _ = curve_fit(model, lambdas, influences, p0=[1.0, 2.0])
a, b = params

lambda_c = 1.0 / b

print("\n=== LAMBDA CRITICAL SCALE ===")
print(f"a ≈ {a:.4f}")
print(f"b ≈ {b:.4f}")
print(f"lambda_c ≈ {lambda_c:.4f}")

# -------------------------
# Save result
# -------------------------
with open("lambda_c_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda_c"])
    writer.writerow([lambda_c])
