import numpy as np
import csv
from scipy.optimize import curve_fit

lambdas = []
influences = []

with open("lambda_influence.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lambdas.append(float(row["lambda"]))
        influences.append(float(row["influence"]))

lambdas = np.array(lambdas)
influences = np.array(influences)

def model(l, a, b):
    return a * np.exp(b * l)

params, _ = curve_fit(model, lambdas, influences, p0=[1.0, 2.0])

a, b = params

print("\n=== EXPONENTIAL FIT ===")
print(f"a ≈ {a:.4f}")
print(f"b ≈ {b:.4f}")

pred = model(lambdas, a, b)
rmse = np.sqrt(np.mean((pred - influences) ** 2))

print(f"RMSE ≈ {rmse:.6f}")
