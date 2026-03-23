import csv
import matplotlib.pyplot as plt
import numpy as np

DATA = "data/logs/gv_constraint_pressure.csv"

# Load data
rows = []
with open(DATA, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Extract columns safely
t = np.array([float(r["t"]) for r in rows])
gv = np.array([float(r["gv"]) for r in rows])
constraint = np.array([float(r["constraint_pressure"]) for r in rows])

# ---- Compute ξ (xi) proxy ----
# Idea: inverse of local instability (variance + gradient)
grad = np.gradient(gv)
variance = np.array([
    np.var(gv[max(0, i-5):i+5]) for i in range(len(gv))
])

xi = 1 / (1 + np.abs(grad) + variance)

# Normalize for plotting
xi_norm = (xi - np.min(xi)) / (np.max(xi) - np.min(xi))

# ---- Plot ----
plt.figure(figsize=(12, 6))

plt.plot(t, gv, label="GV", linewidth=2)
plt.plot(t, constraint, label="Constraint Pressure", alpha=0.7)
plt.plot(t, xi_norm, label="ξ (coherence proxy)", linestyle="--")

plt.title("GV vs Constraint vs ξ Overlay")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
