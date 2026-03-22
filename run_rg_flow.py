import numpy as np
import matplotlib.pyplot as plt

# Load epsilon surface
data = np.genfromtxt("epsilon_surface.csv", delimiter=",", names=True)

lambda_c = data["lambda"]
epsilon = data["epsilon"]

# Fitted constants
alpha = 1.2855
beta = 2.0913
gamma = -3.4480

# Define RG flow step (gradient toward fixed point)
def flow_step(l):
    return alpha * np.exp(-beta * l) + gamma

# Simulate flow
lambda_vals = np.linspace(min(lambda_c), max(lambda_c), 100)
flow = flow_step(lambda_vals)

plt.figure(figsize=(8,5))

# Plot flow field
plt.plot(lambda_vals, flow, label="RG flow")

# Plot data
plt.scatter(lambda_c, epsilon, alpha=0.6, label="data")

plt.xlabel("lambda_c")
plt.ylabel("epsilon")
plt.title("RG Flow Toward Fixed Point")
plt.legend()
plt.tight_layout()

plt.savefig("rg_flow.png", dpi=200)
print("Saved: rg_flow.png")
