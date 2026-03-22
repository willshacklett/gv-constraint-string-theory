import numpy as np
import csv
import matplotlib.pyplot as plt

A = 1.70

lambda_c = np.array([0.4964, 0.4812, 0.4716, 0.4624, 0.4492])
k_c = np.array([3.7919, 3.6758, 3.6023, 3.5317, 3.4308])

products = k_c * lambda_c
epsilon = products - A

with open("epsilon_surface.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda_c", "k_c", "product", "epsilon"])
    for i in range(len(lambda_c)):
        writer.writerow([lambda_c[i], k_c[i], products[i], epsilon[i]])

coeffs = np.polyfit(lambda_c, epsilon, 1)
a, b = coeffs

lambda_grid = np.linspace(lambda_c.min(), lambda_c.max(), 100)
epsilon_fit = a * lambda_grid + b

plt.figure()
plt.scatter(lambda_c, epsilon)
plt.plot(lambda_grid, epsilon_fit)
plt.xlabel("lambda_c")
plt.ylabel("epsilon")
plt.title("Epsilon vs Lambda")
plt.tight_layout()
plt.savefig("epsilon_vs_lambda.png", dpi=200)
plt.close()

print("\n=== EPSILON MODEL ===")
print(f"epsilon ≈ {a:.4f} * lambda + {b:.4f}")
print("\nValues:")
for i in range(len(epsilon)):
    print(f"lambda={lambda_c[i]:.4f}, epsilon={epsilon[i]:.6f}")

print("\nSaved:")
print("- epsilon_surface.csv")
print("- epsilon_vs_lambda.png")
print("\nDone.\n")
