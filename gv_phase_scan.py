import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# CONFIG
# ================================
GRID_SIZE = 64
STEPS = 80

INJECTION_RANGE = np.linspace(0.0, 2.0, 12)
ENTROPY_RANGE = np.linspace(0.0, 1.0, 12)
BOUNDARY_STRESS_RANGE = np.linspace(0.0, 1.0, 6)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# CORE MODEL
# ================================
def run_sim(injection, entropy, boundary_stress):
    gv = np.ones((GRID_SIZE, GRID_SIZE))
    entropy_field = np.zeros_like(gv)

    history = []

    for t in range(STEPS):
        laplacian = (
            np.roll(gv, 1, axis=0) +
            np.roll(gv, -1, axis=0) +
            np.roll(gv, 1, axis=1) +
            np.roll(gv, -1, axis=1) -
            4 * gv
        )

        constraint = np.clip(gv, 0, 1)

        d_gv = (
            -1.2 * constraint * (1 - gv)
            + 0.6 * injection
            - 0.08 * entropy_field
            + 0.2 * laplacian
        )

        gv += d_gv

        # boundary stress (edges lose stability)
        gv[0, :] *= (1 - boundary_stress)
        gv[-1, :] *= (1 - boundary_stress)
        gv[:, 0] *= (1 - boundary_stress)
        gv[:, -1] *= (1 - boundary_stress)

        entropy_field += entropy * np.abs(d_gv)

        history.append(np.mean(gv))

    gv = np.clip(gv, 0, None)

    # proxy k_c and lambda_c
    k_c = np.var(gv)
    lambda_c = np.mean(gv)

    invariant = k_c * lambda_c

    return invariant, np.array(history), gv


# ================================
# SWEEP
# ================================
results = []

print("Running full phase sweep...")

for inj in INJECTION_RANGE:
    for ent in ENTROPY_RANGE:
        for bound in BOUNDARY_STRESS_RANGE:
            inv, hist, field = run_sim(inj, ent, bound)

            results.append({
                "inj": inj,
                "ent": ent,
                "bound": bound,
                "inv": inv
            })

print("Sweep complete.")

# ================================
# BUILD PHASE MAP (mean invariant)
# ================================
phase_map = np.zeros((len(INJECTION_RANGE), len(ENTROPY_RANGE)))

for i, inj in enumerate(INJECTION_RANGE):
    for j, ent in enumerate(ENTROPY_RANGE):
        subset = [r["inv"] for r in results if r["inj"] == inj and r["ent"] == ent]
        phase_map[i, j] = np.mean(subset)

# ================================
# CURVATURE (2nd derivative approx)
# ================================
curvature = np.zeros_like(phase_map)

for i in range(1, len(INJECTION_RANGE)-1):
    for j in range(1, len(ENTROPY_RANGE)-1):
        curvature[i, j] = (
            phase_map[i+1, j] + phase_map[i-1, j] +
            phase_map[i, j+1] + phase_map[i, j-1] -
            4 * phase_map[i, j]
        )

# ================================
# SAVE OUTPUTS
# ================================
np.save(f"{OUTPUT_DIR}/phase_map.npy", phase_map)
np.save(f"{OUTPUT_DIR}/curvature.npy", curvature)

# ================================
# PLOTS
# ================================
plt.figure()
plt.imshow(phase_map, origin='lower')
plt.colorbar()
plt.title("Invariant Phase Map")
plt.savefig(f"{OUTPUT_DIR}/phase_map.png")

plt.figure()
plt.imshow(curvature, origin='lower')
plt.colorbar()
plt.title("Phase Boundary Curvature")
plt.savefig(f"{OUTPUT_DIR}/curvature.png")

print("Saved outputs to /outputs")
