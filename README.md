# GV Constraint String Theory (GV-CST)

A propagation-driven constraint framework linking topology saturation and finite-depth propagation through a universal duality law.

---

## Overview

GV Constraint String Theory (GV-CST) models complex systems as constraint-driven propagation fields rather than static structures.

Instead of treating dynamics as eigenmode decomposition (Laplacian / spectral models), GV-CST describes systems as:

- Finite-depth propagation processes
- Constraint-saturated topologies
- Dual representations of the same underlying flow

The key result:

> Topology scale and propagation depth are not independent — they are dual.

---

## Core Concept

Two fundamental quantities emerge:

- `k_c` → topology saturation scale  
- `lambda_c` → propagation depth scale  

These are linked through a universal bridge:

\[
k_c \cdot \lambda_c \approx \text{constant}
\]

Empirically:

\[
k_c \cdot \lambda_c \approx 1.70
\]

This is the first-order GV-CST duality.

---

## Propagation Law (Empirical Closure)

### Primary Relation

\[
k_c \cdot \lambda_c = A + \epsilon(C,S,\lambda)
\]

Where:

- `A ≈ 1.70` (leading invariant)
- `epsilon` = structured correction term (NOT noise)

---

## Extracted Fits

### Lambda influence (finite-depth kernel)

\[
\mathrm{influence}(\lambda) \approx 0.7528 \cdot e^{2.1204 \lambda}
\]

→ gives:

\[
\lambda_c \approx 0.4716
\]

---

### Topology response

\[
\alpha(k) = 0.9623 - 0.1630 e^{-0.2776 k}
\]

→ gives:

\[
k_c \approx 3.60
\]

---

### Duality Check

\[
k_c \cdot \lambda_c \approx 3.60 \cdot 0.4716 \approx 1.6989
\]

→ locks to:

\[
A \approx 1.70
\]

---

## Residual Structure

Residual analysis shows:

- corr(lambda_c, residual) ≈ 0.9999  
- bounded variance  
- smooth monotonic structure  

Conclusion:

> epsilon is a deterministic propagation correction — not random error.

---

## Residual Fit (Closure Term)

Fitted correction model:

\[
\epsilon(\lambda) \approx a\lambda + b
\]

with:

- `a ≈ 7.2387`
- `b ≈ -3.4130`

---

## Closed Predictive Form

Full propagation law becomes:

\[
k_c(\lambda) = \frac{A + a\lambda + b}{\lambda}
\]

This predicts observed `k_c` across all tested perturbations.

---

## Duality Sweep Results

Across ±5% perturbations:

- product range: 1.54 → 1.88  
- mean ≈ 1.7048  
- std ≈ 0.116  

Interpretation:

> Duality remains stable under perturbation.

---

## Cross-Regime Validation

Measured:

- mean error ≈ 0.0039  
- max error ≈ 0.0059  
- mean relative error ≈ 0.0011  

Conclusion:

> The law is stable across regimes — not overfit.

---

## Universal Collapse Test

Using:

\[
A = 1.70
\]

Predicted:

- k_c reconstructed from lambda_c across slices
- Matches observed topology scale with high accuracy

Conclusion:

> The system collapses to a single propagation law.

---

## Interpretation

GV-CST shows:

- Systems behave as **finite-depth propagation fields**
- Stability is governed by **constraint flow**, not eigenmodes
- Laplacian models are insufficient in this regime
- Topology and propagation are dual descriptions

---

## Key Insight

> GV-CST is propagation-first, not structure-first.

Topology is what propagation leaves behind.

---

## Summary

\[
k_c \cdot \lambda_c = 1.70 + \epsilon(C,S,\lambda)
\]

- Leading invariant: **1.70**
- Correction: **deterministic, propagation-driven**
- Behavior: **universal under perturbation**

---

## Repo Structure
 src/
fit_lambda_curve.py
extract_lambda_c.py
run_lambda_k_duality.py
run_duality_sweep.py
run_residual_analysis.py
fit_residual_curve.py
run_universal_collapse.py
run_cross_regime_validation.py

data/
lambda_influence.csv
lambda_c_results.csv


---

## Status

- Duality: locked  
- Residual: extracted  
- Closure: validated  
- Universality: confirmed  

---

## Next Steps

- Extract ε(C,S) hypersurface
- Fit exponential correction:
  \[
  \epsilon \sim \alpha e^{-\beta \lambda}
  \]
- Validate RG-style flattening
- Extend to spatial / lattice systems

---

## Philosophy

GV-CST treats reality as:

> Constraint-regulated propagation across finite depth

Not static structure.

---

## License

MIT
