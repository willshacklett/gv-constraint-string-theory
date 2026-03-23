#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

SIZE = 64

def generate_field(size):
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)

    # Synthetic constraint field (you can swap later)
    Z = np.sin(X*2) * np.cos(Y*2) + np.exp(-(X**2 + Y**2))

    return Z

def compute_stability(Z):
    gx, gy = np.gradient(Z)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # normalize
    grad_mag /= np.max(grad_mag)

    stability = 1 - grad_mag
    return stability

def classify(stability):
    states = np.zeros_like(stability)

    states[stability > 0.7] = 0  # stable
    states[(stability <= 0.7) & (stability > 0.4)] = 1  # boundary
    states[(stability <= 0.4) & (stability > 0.2)] = 2  # transition
    states[stability <= 0.2] = 3  # collapse

    return states

def plot(states):
    plt.figure(figsize=(6,6))
    plt.imshow(states)
    plt.title("GV Basin Grid (64x64)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("figures/gv_basin_grid.png")
    plt.close()

def main():
    Z = generate_field(SIZE)
    stability = compute_stability(Z)
    states = classify(stability)
    plot(states)

    unique, counts = np.unique(states, return_counts=True)
    print("State distribution:")
    for u, c in zip(unique, counts):
        print(f"{int(u)}: {c}")

if __name__ == "__main__":
    main()
