# Instrumental Variables

Instrumental Variables (IV) is a method for handling unobserved confounding.

## The Problem: Unobserved Confounding

Sometimes we have confounders that we can't measure, making methods like matching insufficient.

## The Solution: An Instrument

An **instrument** $Z_i$ is a variable that:

1. **Relevance**: Affects the treatment $W_i$
2. **Exclusion**: Affects the outcome $Y_i$ only through its effect on treatment
3. **Exogeneity**: Is independent of unobserved confounders

## Classic Example: Returns to Education

**Question**: What is the causal effect of education on earnings?

**Problem**: Unobserved ability affects both education and earnings (confounding)

**Instrument**: Distance to nearest college
- Affects education (relevance)
- Doesn't directly affect earnings, only through education (exclusion)
- Arguably exogenous

## Two-Stage Least Squares (2SLS)

### Stage 1: Predict treatment using instrument
$$\widehat{W_i} = \alpha_0 + \alpha_1 Z_i + \epsilon_i$$

### Stage 2: Regress outcome on predicted treatment
$$Y_i = \beta_0 + \beta_1 \widehat{W_i} + u_i$$

The coefficient $\beta_1$ estimates the **Local Average Treatment Effect (LATE)**.

## Example Code

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

# Simulate IV data
np.random.seed(42)
n = 1000

# Instrument (e.g., randomized encouragement)
Z = np.random.binomial(1, 0.5, n)

# Unobserved confounder
U = np.random.normal(0, 1, n)

# Treatment (affected by instrument and confounder)
W_latent = -0.5 + 1.5*Z + 0.8*U + np.random.normal(0, 0.5, n)
W = (W_latent > 0).astype(int)

# Outcome (affected by treatment and confounder)
Y = 50 + 10*W + 5*U + np.random.normal(0, 2, n)

# Naive regression (biased due to unobserved U)
naive_model = LinearRegression()
naive_model.fit(W.reshape(-1, 1), Y)
naive_effect = naive_model.coef_[0]

# Two-Stage Least Squares
# Stage 1: Regress treatment on instrument
stage1 = LinearRegression()
stage1.fit(Z.reshape(-1, 1), W)
W_hat = stage1.predict(Z.reshape(-1, 1))

# Stage 2: Regress outcome on predicted treatment
stage2 = LinearRegression()
stage2.fit(W_hat.reshape(-1, 1), Y)
iv_effect = stage2.coef_[0]

print(f"Naive estimate: {naive_effect:.2f}")
print(f"IV estimate: {iv_effect:.2f}")
print(f"True effect: 10.00")

# Check instrument strength
f_stat = np.var(W_hat) / np.var(W - W_hat) * (n - 2)
print(f"\nFirst-stage F-statistic: {f_stat:.2f}")
print("(Rule of thumb: F > 10 for strong instrument)")
```

## The LATE Interpretation

IV estimates the effect for **compliers** - those whose treatment status is affected by the instrument.

## Potential Problems

1. **Weak instruments**: If $Z$ barely affects $W$, estimates are unreliable
2. **Invalid instruments**: If exclusion restriction violated, estimates are biased
3. **Heterogeneous effects**: LATE may not equal ATE

## Testing and Diagnostics

- **First-stage F-test**: Test instrument strength (want F > 10)
- **Overidentification test**: If multiple instruments, test exclusion restriction
- **Durbin-Wu-Hausman test**: Test whether IV is necessary
