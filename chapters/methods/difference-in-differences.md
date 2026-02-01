# Difference-in-Differences

Difference-in-Differences (DiD) compares changes over time between treated and control groups.

## The Setup

- **Treatment group**: Receives intervention at time $t$
- **Control group**: Does not receive intervention
- **Pre-period**: Before intervention
- **Post-period**: After intervention

## The DiD Estimator

$$\widehat{\tau}_{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$$

This "differences out" time-invariant differences between groups and common time trends.

## Key Assumption: Parallel Trends

In the absence of treatment, the treatment and control groups would have followed **parallel trends**.

$$E[Y_{it}(0) | G=1] - E[Y_{it'}(0) | G=1] = E[Y_{it}(0) | G=0] - E[Y_{it'}(0) | G=0]$$

## Visual Intuition

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate DiD data
np.random.seed(42)
n_per_group = 500

# Time periods
time_periods = [0, 1, 2, 3, 4, 5]  # Treatment occurs at t=3

# Control group (no treatment)
control_data = []
for t in time_periods:
    y = 10 + 2*t + np.random.normal(0, 1, n_per_group)
    control_data.append(pd.DataFrame({
        'time': t,
        'outcome': y,
        'group': 'Control',
        'treated': 0
    }))

# Treatment group (treatment at t=3)
treatment_data = []
for t in time_periods:
    treatment_effect = 5 if t >= 3 else 0  # Effect starts at t=3
    y = 15 + 2*t + treatment_effect + np.random.normal(0, 1, n_per_group)
    treatment_data.append(pd.DataFrame({
        'time': t,
        'outcome': y,
        'group': 'Treatment',
        'treated': 1 if t >= 3 else 0
    }))

df = pd.concat(control_data + treatment_data, ignore_index=True)

# Calculate means
means = df.groupby(['time', 'group'])['outcome'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
for group in ['Control', 'Treatment']:
    data = means[means['group'] == group]
    plt.plot(data['time'], data['outcome'], marker='o', label=group, linewidth=2)

plt.axvline(x=2.5, color='red', linestyle='--', label='Treatment starts')
plt.xlabel('Time Period')
plt.ylabel('Outcome')
plt.title('Difference-in-Differences: Parallel Trends')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('did_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Estimate DiD
pre_period = df['time'] < 3
post_period = df['time'] >= 3

did_estimate = (
    df[post_period & (df['group'] == 'Treatment')]['outcome'].mean() -
    df[pre_period & (df['group'] == 'Treatment')]['outcome'].mean() -
    (df[post_period & (df['group'] == 'Control')]['outcome'].mean() -
     df[pre_period & (df['group'] == 'Control')]['outcome'].mean())
)

print(f"DiD estimate: {did_estimate:.2f}")
print(f"True treatment effect: 5.00")
```

## Regression Specification

$$Y_{it} = \alpha + \beta \cdot \text{Treated}_i + \gamma \cdot \text{Post}_t + \tau \cdot (\text{Treated}_i \times \text{Post}_t) + \epsilon_{it}$$

The coefficient $\tau$ on the interaction term is the DiD estimator.

```python
import statsmodels.formula.api as smf

# Create indicator variables
df['post'] = (df['time'] >= 3).astype(int)
df['treated_group'] = (df['group'] == 'Treatment').astype(int)

# Regression
model = smf.ols('outcome ~ treated_group + post + treated_group:post', data=df)
results = model.fit()

print(results.summary())
print(f"\nDiD coefficient: {results.params['treated_group:post']:.2f}")
```

## Extensions

### Multiple Time Periods
Can include multiple pre and post periods for robustness.

### Multiple Groups
Can have staggered treatment adoption across groups.

### Event Study
Estimate effects for each time period relative to treatment:

$$Y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \delta_k \cdot \mathbb{1}[t - T_i = k] + \epsilon_{it}$$

## Validity Checks

1. **Pre-trends test**: Test for parallel trends in pre-treatment periods
2. **Placebo tests**: Test for "effects" before treatment
3. **Sensitivity analysis**: Test robustness to violations of parallel trends

## Common Violations

- **Differential pre-trends**: Groups on different trajectories before treatment
- **Spillovers**: Control group affected by treatment
- **Composition changes**: Different units in treatment/control over time
