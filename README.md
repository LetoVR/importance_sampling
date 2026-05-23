# Importance Sampling for Option Pricing

Monte Carlo variance reduction via Importance Sampling, applied to vanilla and exotic options under Black-Scholes. Academic project in Quantitative Finance — ENSTA Paris / IP Paris (2026).

**Authors:** Nathanaël Seropian · Leto Van Ruymbeke · Ramzi Jebali

---

## Overview

Standard Monte Carlo struggles with out-of-the-money options: most paths yield zero payoff, making the estimator noisy and slow to converge. This project implements **Importance Sampling (IS)** — a change of measure that concentrates simulations in the "useful" region of the probability space — combined with a likelihood ratio correction to maintain unbiasedness.

The optimal shift parameter θ* is found by minimizing the estimator variance via a **damped Newton algorithm**.

---

## Methods

### 1. Theoretical Framework
- Change of measure for Gaussian distributions: `E[g(X)] = E[g(X+θ) · exp(−θX − ½‖θ‖²)]`
- Gradient of the variance with respect to θ (Leibniz rule under expectation)
- Damped Newton algorithm with backtracking line search for robust convergence

### 2. Vanilla European Call (dimension 1)
- Box-Muller sampling, adaptive N targeting ε = 10⁻³ at 95% confidence
- Newton convergence tracked for strikes K ∈ {0.35, 0.54, 0.7, 1.24, 1.6, 2.5}
- IS vs standard MC comparison on deep OTM case (K = 2.5)

### 3. Exotic Options in Dimension 3
- **Basket Call** — linear combination of 3 correlated underlyings (Cholesky decomposition of correlation matrix)
- **Symphonie Option** — payoff based on max/min/median of the 3 underlyings; analytical study reveals K cancels out, making IS ineffective (the event is not rare)
- Vectorized Newton with 3×3 Jacobian matrix

### 4. Control Variates
- Control variable: discounted basket value Z = e^{−rT} X(T), with known expectation Σλᵢ Sᵢ,₀
- Optimal coefficient c* = Cov(Y, Z) / Var(Z)
- Comparison with IS on Basket Call (K = 1.25): Control Variates outperform IS near-the-money at lower computational cost

---

## Key Results

| Method | Strike | Variance reduction |
|---|---|---|
| Standard MC | K = 2.5 | baseline (unusable) |
| Importance Sampling | K = 2.5 | dramatic — CI collapses immediately |
| Importance Sampling | K = 1.25 (Basket) | moderate |
| Control Variates | K = 1.25 (Basket) | superior to IS, near-zero cost |
| IS on Symphonie | K = 1.25 | none — equivalent to standard MC |

**Main takeaway:** IS is the weapon of choice for rare-event pricing (deep OTM). For near-the-money options, Control Variates often dominate in the simplicity/performance tradeoff.

---

## Repository Structure

- `importance_sampling.py` — Full implementation (Newton, MC, IS, basket, Symphonie, CV)
- `report/report.pdf` — Full mathematical derivations and figures

---

## Dependencies

```bash
pip install numpy matplotlib
```

---

## Parameters (default)

```python
S0 = 1        # Initial asset price
sigma = 0.3   # Volatility
r = 0.01      # Risk-free rate
K = 1         # Strike
T = 2         # Maturity
```

For the basket option: 3 correlated underlyings with ρᵢⱼ = 0.5, σ = [0.25, 0.28, 0.30], equal weights λ = 1/3.

---

## Academic Context

Course: *PRB222 — Financial Engineering* (APM_4PRB9)  
Program: Master's in Applied Mathematics — ENSTA Paris / Institut Polytechnique de Paris  
Year: 2025–2026
