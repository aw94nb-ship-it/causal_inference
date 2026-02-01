# Mediation Analysis

Mediation analysis decomposes causal effects into direct and indirect pathways.

## The Question

**Does treatment $W$ affect outcome $Y$ directly, or through a mediator $M$?**

## Example

- $W$: Job training program
- $M$: Skills acquired
- $Y$: Employment

Does training affect employment directly, or only through increased skills?

## The Mediation Model

```
W → M → Y
W -------→ Y
```

- **Total Effect (TE)**: Overall effect of $W$ on $Y$
- **Direct Effect (DE)**: Effect of $W$ on $Y$ not through $M$
- **Indirect Effect (IE)**: Effect of $W$ on $Y$ through $M$

$$\text{TE} = \text{DE} + \text{IE}$$

## Natural Direct and Indirect Effects

### Natural Direct Effect (NDE)
Effect of treatment on outcome if we set the mediator to what it would be under control:

$$\text{NDE} = E[Y(W=1, M(0)) - Y(W=0, M(0))]$$

### Natural Indirect Effect (NIE)
Effect on outcome from changing the mediator from control to treatment level, holding treatment at treated:

$$\text{NIE} = E[Y(W=1, M(1)) - Y(W=1, M(0))]$$

## Identification Assumptions

1. **No unmeasured treatment-outcome confounding**
2. **No unmeasured mediator-outcome confounding**
3. **No unmeasured treatment-mediator confounding**
4. **No treatment-induced mediator-outcome confounding**

The fourth assumption is strong: no confounder of $M \to Y$ can be affected by treatment.

## Baron-Kenny Approach (Traditional)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Simulate mediation data
np.random.seed(42)
n = 1000

# Treatment
W = np.random.binomial(1, 0.5, n)

# Mediator (affected by treatment)
M = 10 + 3*W + np.random.normal(0, 2, n)

# Outcome (affected by both treatment and mediator)
Y = 5 + 2*W + 1.5*M + np.random.normal(0, 3, n)

# Step 1: Total effect (W -> Y)
model_total = sm.OLS(Y, sm.add_constant(W)).fit()
total_effect = model_total.params[1]

# Step 2: Treatment -> Mediator
model_mediator = sm.OLS(M, sm.add_constant(W)).fit()
a_path = model_mediator.params[1]

# Step 3: Mediator -> Outcome (controlling for treatment)
X = np.column_stack([W, M])
model_outcome = sm.OLS(Y, sm.add_constant(X)).fit()
direct_effect = model_outcome.params[1]  # c' path
b_path = model_outcome.params[2]

# Indirect effect
indirect_effect = a_path * b_path

print("Mediation Analysis Results:")
print(f"Total effect: {total_effect:.2f}")
print(f"Direct effect: {direct_effect:.2f}")
print(f"Indirect effect: {indirect_effect:.2f}")
print(f"Proportion mediated: {indirect_effect/total_effect:.2%}")
```

## Modern Approach: Causal Mediation Analysis

```python
# Using potential outcomes framework
# Requires additional packages like 'mediation' in R or custom implementation

def estimate_natural_effects(W, M, Y, X=None):
    """
    Estimate natural direct and indirect effects.

    Uses parametric regression-based approach.
    """
    # Model for mediator
    if X is not None:
        mediator_model = sm.OLS(M, sm.add_constant(np.column_stack([W, X]))).fit()
    else:
        mediator_model = sm.OLS(M, sm.add_constant(W)).fit()

    # Model for outcome
    if X is not None:
        outcome_model = sm.OLS(Y, sm.add_constant(np.column_stack([W, M, X]))).fit()
    else:
        outcome_model = sm.OLS(Y, sm.add_constant(np.column_stack([W, M]))).fit()

    # Predict potential outcomes
    n = len(W)

    # Y(1, M(1))
    M1_pred = mediator_model.predict(sm.add_constant(np.column_stack([np.ones(n), X]) if X is not None else np.ones((n, 1))))
    Y11_pred = outcome_model.predict(sm.add_constant(np.column_stack([np.ones(n), M1_pred, X]) if X is not None else np.column_stack([np.ones(n), M1_pred])))

    # Y(0, M(0))
    M0_pred = mediator_model.predict(sm.add_constant(np.column_stack([np.zeros(n), X]) if X is not None else np.zeros((n, 1))))
    Y00_pred = outcome_model.predict(sm.add_constant(np.column_stack([np.zeros(n), M0_pred, X]) if X is not None else np.column_stack([np.zeros(n), M0_pred])))

    # Y(1, M(0))
    Y10_pred = outcome_model.predict(sm.add_constant(np.column_stack([np.ones(n), M0_pred, X]) if X is not None else np.column_stack([np.ones(n), M0_pred])))

    # Effects
    total_effect = np.mean(Y11_pred - Y00_pred)
    nde = np.mean(Y10_pred - Y00_pred)
    nie = np.mean(Y11_pred - Y10_pred)

    return {
        'total_effect': total_effect,
        'natural_direct_effect': nde,
        'natural_indirect_effect': nie,
        'prop_mediated': nie / total_effect if total_effect != 0 else 0
    }

results = estimate_natural_effects(W, M, Y)
for key, val in results.items():
    print(f"{key}: {val:.2f}")
```

## Sensitivity Analysis for Mediation

The sequential ignorability assumption is strong. Sensitivity analysis can assess robustness:

- **Correlation between errors**: How correlated can $\epsilon_M$ and $\epsilon_Y$ be?
- **Unmeasured confounding**: How strong would $U$ need to be?

## Applications

- **Psychology**: Does therapy reduce anxiety through improved coping skills?
- **Medicine**: Does a drug improve survival through reduced inflammation?
- **Economics**: Does education increase earnings through skill development?

## Limitations

- Strong assumptions about confounding
- Requires correct model specification
- Cross-sectional data provides weaker evidence than longitudinal

## Further Reading

- Imai, K., Keele, L., & Tingley, D. (2010). "A general approach to causal mediation analysis"
- Pearl, J. (2001). "Direct and indirect effects"
- VanderWeele, T. J. (2015). "Explanation in Causal Inference"
