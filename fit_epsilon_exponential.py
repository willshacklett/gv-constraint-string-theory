import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

lambda_c = np.array([0.4964, 0.4812, 0.4716, 0.4624, 0.4492])
epsilon = np.array([0.182299, 0.068795, -0.001155, -0.066942, -0.158885])

def model(l, alpha, beta):
    return alpha * np.exp(-beta * l)

params, _ = curve_fit(model, lambda_c, epsilon, p0=[1.0, 1.0])

alpha, beta = params

lambda_grid = np.linspace(lambda_c.min(), lambda_c.max(), 100)
fit = model(lambda_grid, alpha, beta)

plt.figure()
plt.scatter(lambda_c, epsilon)
plt.plot(lambda_grid, fit)
plt.xlabel("lambda_c")
plt.ylabel("epsilon")
plt.title("Exponential Fit: epsilon ~ alpha exp(-beta lambda)")
plt.savefig("epsilon_exponential_fit.png")
plt.close()

print("\n=== EXPONENTIAL EPSILON FIT ===")
print(f"alpha ≈ {alpha:.4f}")
print(f"beta ≈ {beta:.4f}")
print("\nDone.\n")
