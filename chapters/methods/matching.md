# Matching Methods

When randomization isn't possible, matching tries to approximate it by comparing similar units.

## Basic Idea

Match each treated unit with one or more control units that have similar characteristics (covariates).

## Types of Matching

### Exact Matching
- Find control units with exactly the same covariate values
- Difficult with many covariates (curse of dimensionality)

### Propensity Score Matching
- **Propensity score**: Probability of receiving treatment given covariates
  $$e(X_i) = P(W_i = 1 | X_i)$$
- Match on this single score instead of all covariates
- Reduces dimensionality problem

### Nearest Neighbor Matching
- For each treated unit, find the k nearest control units
- Can use different distance metrics (Euclidean, Mahalanobis)

## Propensity Score Methods

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Generate sample data
np.random.seed(42)
n = 1000

# Covariates
age = np.random.normal(45, 10, n)
income = np.random.normal(50000, 20000, n)

# Treatment depends on covariates (selection bias!)
prob_treatment = 1 / (1 + np.exp(-(-2 + 0.05*age + 0.00002*income)))
treatment = np.random.binomial(1, prob_treatment)

# Outcomes
y0 = 100 + 0.5*age + 0.0005*income + np.random.normal(0, 10, n)
y1 = y0 + 20  # Treatment effect = 20

outcome = treatment * y1 + (1 - treatment) * y0

# Estimate propensity scores
X = np.column_stack([age, income])
ps_model = LogisticRegression()
ps_model.fit(X, treatment)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# Match treated to control units
treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

# For each treated unit, find nearest control
nn = NearestNeighbors(n_neighbors=1)
nn.fit(propensity_scores[control_idx].reshape(-1, 1))
distances, matches = nn.kneighbors(
    propensity_scores[treated_idx].reshape(-1, 1)
)

# Calculate ATT
att_estimate = (
    outcome[treated_idx].mean() -
    outcome[control_idx[matches.flatten()]].mean()
)

print(f"Estimated ATT: {att_estimate:.2f}")
print(f"True treatment effect: 20.00")
```

## Common Variants

1. **Caliper matching**: Only match if distance is below threshold
2. **Kernel matching**: Weighted average of multiple controls
3. **Coarsened Exact Matching (CEM)**: Coarsen variables then exact match

## Assumptions

- **Unconfoundedness**: All confounders are observed and included
- **Common support**: Overlap in covariate distributions between groups

## Limitations

- Can only control for observed confounders
- Results depend on model specification
- May discard data (e.g., unmatched units)
