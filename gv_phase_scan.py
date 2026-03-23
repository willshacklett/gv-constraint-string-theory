import csv
import os

import matplotlib.pyplot as plt
import numpy as np

# ================================
# CONFIG
# ================================
GRID_SIZE = 64
STEPS = 120

INJECTION_RANGE = np.linspace(0.0, 1.2, 9)
ENTROPY_RANGE = np.linspace(0.0, 1.0, 9)
BOUNDARY_STRESS_RANGE = np.linspace(0.0, 0.8, 5)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# HELPERS
# ================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_lambda_c(field):
    """
    Rough correlation-length proxy from nearest-neighbor agreement.
    Kept simple on purpose for first-pass exploration.
    """
    dx = np.mean(np.abs(field - np.roll(field, 1, axis=0)))
    dy = np.mean(np.abs(field - np.roll(field, 1, axis=1)))
    local_diff = 0.5 * (dx + dy)
    return 1.0 / (local_diff + 1e-6)


def compute_k_c(field):
    """
    Basin / structure proxy from variance after lock.
    """
    return np.var(field)


# ================================
# CORE MODEL
# ================================
def run_simulation(injection, entropy, boundary_stress):
    """
    Goal:
    Move from smooth runaway response toward constraint-selected locking.

    Mechanisms added:
    - sharper nonlinear gating
    - double-well attractor term
    - stronger entropy drag
    - explicit collapse penalty
    - hard clipping to suppress runaway growth
    """

    gv = np.ones((GRID_SIZE, GRID_SIZE), dtype=float) * 0.55
    gv += 0.02 * np.random.randn(GRID_SIZE, GRID_SIZE)
    entropy_field = np.zeros_like(gv)

    mean_history = []
    invariant_history = []

    for _ in range(STEPS):
        laplacian = (
            np.roll(gv, 1, axis=0)
            + np.roll(gv, -1, axis=0)
            + np.roll(gv, 1, axis=1)
            + np.roll(gv, -1, axis=1)
            - 4.0 * gv
        )

        # Sharper selector gate:
        # near 0 below threshold, near 1 above threshold
        gate = sigmoid(12.0 * (gv - 0.5))

        # Double-well style locking:
        # pushes field toward low or high stable basin
        basin_force = -6.0 * gv * (gv - 0.35) * (gv - 1.0)

        # Collapse penalty:
        # if gv gets too high, suppress it hard
        collapse_penalty = 1.8 * np.maximum(gv - 1.05, 0.0)

        # Entropy drag grows with motion
        entropy_drag = 0.20 * entropy_field

        d_gv = (
            basin_force
            + 0.18 * laplacian
            + 0.30 * injection * gate
            - entropy_drag
            - collapse_penalty
        )

        gv += d_gv

        # Boundary stress applied after update
        edge_factor = (1.0 - boundary_stress)
        gv[0, :] *= edge_factor
        gv[-1, :] *= edge_factor
        gv[:, 0] *= edge_factor
        gv[:, -1] *= edge_factor

        # Entropy accumulates from local motion
        entropy_field += entropy * np.abs(d_gv)

        # Hard clip to prevent runaway
        gv = np.clip(gv, 0.0, 1.2)

        lambda_c = compute_lambda_c(gv)
        k_c = compute_k_c(gv)
        invariant = k_c * lambda_c

        mean_history.append(np.mean(gv))
        invariant_history.append(invariant)

    final_lambda_c = compute_lambda_c(gv)
    final_k_c = compute_k_c(gv)
    final_invariant = final_k_c * final_lambda_c

    return {
        "final_field": gv.copy(),
        "mean_history": np.array(mean_history),
        "invariant_history": np.array(invariant_history),
        "lambda_c": final_lambda_c,
        "k_c": final_k_c,
        "invariant": final_invariant,
    }


# ================================
# FULL SWEEP
# ================================
print("Running locking-enabled phase sweep...")

results = []
phase_map = np.zeros((len(INJECTION_RANGE), len(ENTROPY_RANGE)))
lambda_map = np.zeros_like(phase_map)
k_map = np.zeros_like(phase_map)

for i, injection in enumerate(INJECTION_RANGE):
    for j, entropy in enumerate(ENTROPY_RANGE):
        invs = []
        lambdas = []
        ks = []

        for boundary_stress in BOUNDARY_STRESS_RANGE:
            run = run_simulation(injection, entropy, boundary_stress)

            invs.append(run["invariant"])
            lambdas.append(run["lambda_c"])
            ks.append(run["k_c"])

            results.append(
                {
                    "injection": float(injection),
                    "entropy": float(entropy),
                    "boundary_stress": float(boundary_stress),
                    "lambda_c": float(run["lambda_c"]),
                    "k_c": float(run["k_c"]),
                    "invariant": float(run["invariant"]),
                }
            )

        phase_map[i, j] = np.mean(invs)
        lambda_map[i, j] = np.mean(lambdas)
        k_map[i, j] = np.mean(ks)

print("Sweep complete.")

# ================================
# CURVATURE MAP
# ================================
curvature_map = np.zeros_like(phase_map)

for i in range(1, phase_map.shape[0] - 1):
    for j in range(1, phase_map.shape[1] - 1):
        curvature_map[i, j] = (
            phase_map[i + 1, j]
            + phase_map[i - 1, j]
            + phase_map[i, j + 1]
            + phase_map[i, j - 1]
            - 4.0 * phase_map[i, j]
        )

# ================================
# SAVE CSVS
# ================================
with open("lambda_c_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda_c"])
    writer.writerow([float(np.mean(lambda_map))])

with open("lambda_influence.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda", "influence"])
    for lam in np.linspace(0.15, 1.2, 15):
        # simple carried-over diagnostic curve for quick glance
        influence = lam / max(np.mean(lambda_map), 1e-6)
        writer.writerow([float(lam), float(influence)])

with open(os.path.join(OUTPUT_DIR, "phase_scan_results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "injection",
            "entropy",
            "boundary_stress",
            "lambda_c",
            "k_c",
            "invariant",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

# ================================
# SAVE NUMPY ARRAYS
# ================================
np.save(os.path.join(OUTPUT_DIR, "phase_map.npy"), phase_map)
np.save(os.path.join(OUTPUT_DIR, "curvature_map.npy"), curvature_map)
np.save(os.path.join(OUTPUT_DIR, "lambda_map.npy"), lambda_map)
np.save(os.path.join(OUTPUT_DIR, "k_map.npy"), k_map)

# ================================
# PLOTS
# ================================
plt.figure(figsize=(7, 6))
plt.imshow(phase_map, origin="lower", aspect="auto")
plt.colorbar(label="Mean invariant")
plt.xticks(range(len(ENTROPY_RANGE)), [f"{x:.2f}" for x in ENTROPY_RANGE], rotation=45)
plt.yticks(range(len(INJECTION_RANGE)), [f"{x:.2f}" for x in INJECTION_RANGE])
plt.xlabel("Entropy")
plt.ylabel("Injection")
plt.title("Invariant Phase Map")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase_map.png"))
plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(curvature_map, origin="lower", aspect="auto")
plt.colorbar(label="Local curvature")
plt.xticks(range(len(ENTROPY_RANGE)), [f"{x:.2f}" for x in ENTROPY_RANGE], rotation=45)
plt.yticks(range(len(INJECTION_RANGE)), [f"{x:.2f}" for x in INJECTION_RANGE])
plt.xlabel("Entropy")
plt.ylabel("Injection")
plt.title("Phase Boundary Curvature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "curvature_map.png"))
plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(lambda_map, origin="lower", aspect="auto")
plt.colorbar(label="Mean lambda_c")
plt.xticks(range(len(ENTROPY_RANGE)), [f"{x:.2f}" for x in ENTROPY_RANGE], rotation=45)
plt.yticks(range(len(INJECTION_RANGE)), [f"{x:.2f}" for x in INJECTION_RANGE])
plt.xlabel("Entropy")
plt.ylabel("Injection")
plt.title("Lambda Map")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lambda_map.png"))
plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(k_map, origin="lower", aspect="auto")
plt.colorbar(label="Mean k_c")
plt.xticks(range(len(ENTROPY_RANGE)), [f"{x:.2f}" for x in ENTROPY_RANGE], rotation=45)
plt.yticks(range(len(INJECTION_RANGE)), [f"{x:.2f}" for x in INJECTION_RANGE])
plt.xlabel("Entropy")
plt.ylabel("Injection")
plt.title("k_c Map")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "k_map.png"))
plt.close()

print("Saved outputs to /outputs")
print("Done.")
