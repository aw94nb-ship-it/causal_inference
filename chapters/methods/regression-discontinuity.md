# Regression Discontinuity Design

Regression Discontinuity Design (RDD) exploits a threshold rule for treatment assignment.

## The Setup

Treatment is assigned based on whether a "running variable" crosses a threshold:

$$W_i = \begin{cases} 1 & \text{if } X_i \geq c \\ 0 & \text{if } X_i < c \end{cases}$$

where $X_i$ is the running variable and $c$ is the cutoff.

## Key Insight

Units just above and just below the threshold are (arguably) very similar, except for treatment assignment. This creates a "local randomization" around the cutoff.

## Estimation

The treatment effect is estimated as the discontinuity in the outcome at the cutoff:

$$\tau_{RDD} = \lim_{x \downarrow c} E[Y_i | X_i = x] - \lim_{x \uparrow c} E[Y_i | X_i = x]$$

## Types of RDD

### Sharp RDD
Treatment is a deterministic function of the running variable (as above).

### Fuzzy RDD
The probability of treatment jumps at the cutoff, but isn't 0 or 1:

$$P(W_i = 1 | X_i) \text{ jumps at } X_i = c$$

## Example: Test Score Cutoff

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulate RDD data
np.random.seed(42)
n = 1000

# Running variable (test score)
score = np.random.uniform(40, 60, n)

# Cutoff at 50
cutoff = 50
treatment = (score >= cutoff).astype(int)

# Outcome: continuous function + jump at cutoff
outcome = (
    80 + 0.5 * score +  # Baseline trend
    5 * treatment +  # Treatment effect (discontinuity)
    np.random.normal(0, 3, n)  # Noise
)

# Visualize the discontinuity
plt.figure(figsize=(10, 6))
plt.scatter(score[treatment==0], outcome[treatment==0],
           alpha=0.5, label='Control', s=20)
plt.scatter(score[treatment==1], outcome[treatment==1],
           alpha=0.5, label='Treated', s=20)
plt.axvline(cutoff, color='red', linestyle='--', label='Cutoff')
plt.xlabel('Running Variable (Score)')
plt.ylabel('Outcome')
plt.legend()
plt.title('Regression Discontinuity Design')
plt.savefig('rdd_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Estimate treatment effect using local linear regression
bandwidth = 5
local = (score >= cutoff - bandwidth) & (score <= cutoff + bandwidth)

from sklearn.linear_model import LinearRegression

# Separate regressions on each side
below = local & (score < cutoff)
above = local & (score >= cutoff)

model_below = LinearRegression()
model_below.fit(score[below].reshape(-1, 1), outcome[below])

model_above = LinearRegression()
model_above.fit(score[above].reshape(-1, 1), outcome[above])

# Treatment effect = difference at cutoff
te_estimate = (
    model_above.predict([[cutoff]])[0] -
    model_below.predict([[cutoff]])[0]
)

print(f"Estimated treatment effect: {te_estimate:.2f}")
print(f"True treatment effect: 5.00")
```

## Assumptions

1. **Continuity**: All other factors vary smoothly at the cutoff
2. **No manipulation**: Units can't precisely control the running variable
3. **Local effect**: Identifies treatment effect near the cutoff only

## Validity Checks

- **McCrary density test**: Check for manipulation (bunching at cutoff)
- **Covariate balance**: Covariates should be continuous at cutoff
- **Placebo tests**: No discontinuity at other thresholds
