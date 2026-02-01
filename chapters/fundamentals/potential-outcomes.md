# Potential Outcomes Framework

The potential outcomes framework, also known as the Rubin Causal Model, provides a rigorous way to define causal effects.

## Basic Setup

For each unit $i$, we define:

- $Y_i(1)$: The potential outcome if unit $i$ receives treatment
- $Y_i(0)$: The potential outcome if unit $i$ does not receive treatment
- $W_i$: Treatment indicator (1 if treated, 0 if not)

The **observed outcome** is:
$$Y_i = W_i \cdot Y_i(1) + (1 - W_i) \cdot Y_i(0)$$

## Individual Treatment Effect

The causal effect of the treatment for unit $i$ is:

$$\tau_i = Y_i(1) - Y_i(0)$$

The fundamental problem: we can never observe both $Y_i(1)$ and $Y_i(0)$ for the same unit!

## Average Treatment Effect (ATE)

Since we can't observe individual treatment effects, we focus on population averages:

$$\text{ATE} = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]$$

## Average Treatment Effect on the Treated (ATT)

Sometimes we're specifically interested in the effect on those who actually received treatment:

$$\text{ATT} = E[Y_i(1) - Y_i(0) | W_i = 1]$$

## Key Assumptions

### SUTVA (Stable Unit Treatment Value Assumption)

1. **No interference**: One unit's treatment doesn't affect another unit's outcome
2. **No hidden variations of treatment**: Treatment is well-defined and consistent

### Ignorability/Unconfoundedness

Treatment assignment is independent of potential outcomes (conditional on covariates):

$$(Y_i(1), Y_i(0)) \perp W_i | X_i$$

where $X_i$ are observed covariates.

## Example Code

```python
import numpy as np
import pandas as pd

# Simulate potential outcomes
np.random.seed(42)
n = 1000

# Generate covariates
X = np.random.normal(0, 1, n)

# Potential outcomes (unknown in real data!)
Y0 = 2 + 0.5 * X + np.random.normal(0, 1, n)  # Outcome without treatment
Y1 = Y0 + 1.5 + 0.3 * X  # Outcome with treatment (effect = 1.5 + 0.3*X)

# True ATE
true_ate = np.mean(Y1 - Y0)
print(f"True ATE: {true_ate:.3f}")

# Random treatment assignment
W = np.random.binomial(1, 0.5, n)

# Observed outcomes
Y_obs = W * Y1 + (1 - W) * Y0

# Naive estimator (difference in means)
ate_estimate = Y_obs[W==1].mean() - Y_obs[W==0].mean()
print(f"Estimated ATE: {ate_estimate:.3f}")
```

## References

- Rubin, D. B. (1974). "Estimating causal effects of treatments in randomized and nonrandomized studies"
- Imbens, G. W., & Rubin, D. B. (2015). "Causal Inference for Statistics, Social, and Biomedical Sciences"
