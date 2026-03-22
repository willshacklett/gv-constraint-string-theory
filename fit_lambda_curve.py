import numpy as np
import csv
from scipy.optimize import curve_fit

# load data
lambdas = []
influences = []

with open("lambda_influence.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lambdas.append(float(row["lambda"]))
        influences.append(float(row["influence"]))

lambdas = np.array(lambdas)
influences = np.array(influences)

# model: divergence-style (rational)
def model(l, a, lc):
    return a / (1 - l / lc)

params, _ = curve_fit(model, lambdas, influences, p0=[1.0, 0.3])

a, lc = params

print("\n=== LAMBDA FIT ===")
print(f"a ≈ {a:.4f}")
print(f"lambda_c ≈ {lc:.4f}")

# compute simple error
pred = model(lambdas, a, lc)
rmse = np.sqrt(np.mean((pred - influences)**2))
print(f"RMSE ≈ {rmse:.6f}")
