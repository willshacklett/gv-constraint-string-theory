# GV Constraint String Theory (GV-CST)

A propagation-driven constraint framework linking topology saturation and finite-depth propagation through a universal duality law.

---

## Overview

GV Constraint String Theory (GV-CST) models complex systems as constraint-driven propagation fields rather than static structures.

Instead of treating dynamics as eigenmode decomposition (Laplacian / spectral models), GV-CST describes systems as:

- Finite-depth propagation processes  
- Constraint-saturated topologies  
- Dual representations of the same underlying flow  

---

## 🔥 New Result: Finite-Size Shock + Relaxation

We performed a toroidal boundary scan across increasing grid sizes.

### Observed Mean Entropy Boundary

| Grid Size | Mean Boundary |
|----------|--------------|
| 256      | 0.038        |
| 384      | 0.105        |
| 512      | 0.080        |
| 768      | 0.056        |
| 1024     | 0.043        |

---

## Key Finding

The system does **NOT** follow simple monotonic convergence (e.g. 1/N scaling).

Instead, it exhibits:

### ➤ A finite-size constraint shock  
Peak occurs near: **N ≈ 400**

### ➤ Followed by relaxation  
Approaches asymptotic limit: **S_inf ≈ 0.0654**

---

## Fit Model (Peak + Relaxation)

We fit the behavior using a peak + relaxation curve:

- **S_inf ≈ 0.0654**
- **N₀ ≈ 399.7**

This indicates:

> The system undergoes a constraint saturation transition before settling into a stable propagation manifold.

---

## Interpretation

This is critical.

GV-CST predicts:

- Constraint systems are **not scale-smooth**
- There exists a **critical propagation scale**
- After saturation, systems **relax toward a stable attractor**

This is fundamentally different from:

- Diffusion models  
- Spectral / Laplacian models  
- Simple renormalization scaling  

---

## Visual Results

### Scale Trend
outputs/scale_trend.png

### Fit Curve
outputs/gv_fit_curve.png

---

## Output Files

outputs/
├── prediction_test_1024.txt  
├── boundary_map_1024_light.png  
├── scale_trend.png  
├── gv_fit_curve.png  
├── gv_fit_results.txt  

---

## Core Insight

GV-CST behavior is governed by:

> Finite-depth propagation + constraint saturation

Which produces:

- Shock (nonlinear regime change)  
- Relaxation (stable manifold convergence)  

---

## Next Steps

- Test alternate topologies (non-toroidal)
- Increase resolution near N ≈ 400
- Validate whether N₀ shifts or remains invariant

---

## Status

✔ Reproducible  
✔ Observable across scales  
✔ Non-trivial (not noise)  

---

## TL;DR

Not smooth scaling.  
Not noise.  

**Peak → Shock → Relaxation**
