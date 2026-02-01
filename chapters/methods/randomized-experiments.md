# Randomized Experiments

Randomized Controlled Trials (RCTs) are the gold standard for causal inference.

## Why Randomization?

Random assignment of treatment ensures that:

1. Treatment is independent of potential outcomes
2. Treated and control groups are balanced in expectation
3. We can estimate causal effects with simple comparisons

## The Magic of Randomization

Under random assignment:

$$E[Y_i(1) | W_i = 1] = E[Y_i(1)]$$
$$E[Y_i(0) | W_i = 0] = E[Y_i(0)]$$

Therefore, the difference in means is an unbiased estimator of the ATE:

$$\widehat{\text{ATE}} = \frac{1}{n_1}\sum_{i:W_i=1} Y_i - \frac{1}{n_0}\sum_{i:W_i=0} Y_i$$

## Example: A/B Testing

A/B testing is a common application of RCTs in industry.

```python
import numpy as np
import pandas as pd
from scipy import stats

# Simulate an A/B test
np.random.seed(42)
n_control = 1000
n_treatment = 1000

# Control group (version A)
conversion_control = np.random.binomial(1, 0.10, n_control)

# Treatment group (version B) - 2% absolute increase
conversion_treatment = np.random.binomial(1, 0.12, n_treatment)

# Calculate conversion rates
rate_control = conversion_control.mean()
rate_treatment = conversion_treatment.mean()

print(f"Control conversion rate: {rate_control:.3f}")
print(f"Treatment conversion rate: {rate_treatment:.3f}")
print(f"Lift: {(rate_treatment - rate_control):.3f}")

# Statistical test
t_stat, p_value = stats.ttest_ind(conversion_treatment, conversion_control)
print(f"P-value: {p_value:.4f}")
```

## Challenges with RCTs

While RCTs are ideal, they have limitations:

1. **Cost**: Can be expensive to run
2. **Ethics**: Not always ethical to randomize (e.g., smoking)
3. **Feasibility**: Some treatments can't be randomized (e.g., gender, race)
4. **External validity**: Results may not generalize beyond the experimental setting
5. **Compliance**: People may not follow assigned treatment

## Variations

- **Stratified randomization**: Randomize within strata to ensure balance
- **Block randomization**: Ensure equal group sizes
- **Cluster randomization**: Randomize groups rather than individuals
