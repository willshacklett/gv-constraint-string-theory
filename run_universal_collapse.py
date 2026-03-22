import numpy as np

# Your fitted values
A = 1.70
a = 7.2387
b = -3.4130

lambda_c = np.array([0.4964, 0.4812, 0.4716, 0.4624, 0.4492])

print("\n=== UNIVERSAL COLLAPSE TEST ===\n")

for l in lambda_c:
    k_pred = ((A + (a*l + b)) / l)
    print(f"lambda={l:.4f} => k_pred={k_pred:.4f}")

print("\nInterpretation:")
print("If k_pred matches observed k_c → full law is closed")

print("\nDone.\n")
