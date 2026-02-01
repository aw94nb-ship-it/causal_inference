# Sensitivity Analysis

Sensitivity analysis examines how robust causal estimates are to violations of key assumptions.

## Why Sensitivity Analysis?

Most causal inference methods rely on untestable assumptions (e.g., unconfoundedness, parallel trends). Sensitivity analysis helps answer:

**"How strong would an unobserved confounder need to be to change my conclusions?"**

## Omitted Variable Bias

Consider a regression:

$$Y_i = \beta_0 + \beta_1 W_i + \epsilon_i$$

If there's an unobserved confounder $U_i$, the bias in $\hat{\beta}_1$ is approximately:

$$\text{Bias} \approx \frac{\text{Cov}(W, U)}{\text{Var}(W)} \cdot \gamma$$

where $\gamma$ is the effect of $U$ on $Y$.

## Rosenbaum Sensitivity Analysis

For matched studies, Rosenbaum's approach asks:

**How much hidden bias (odds ratio) would be needed to explain away the observed effect?**

Sensitivity parameter $\Gamma$:
- $\Gamma = 1$: No hidden bias (perfect randomization)
- $\Gamma = 2$: Matched pairs could differ by 2:1 odds in treatment probability
- Larger $\Gamma$ = more robust to hidden bias

## Example: E-value

The **E-value** is the minimum strength of association (risk ratio scale) that an unmeasured confounder would need to have with both treatment and outcome to explain away the observed effect.

```python
import numpy as np

def calculate_e_value(rr, ci_lower=None):
    """
    Calculate E-value for a risk ratio.

    Parameters:
    rr: observed risk ratio
    ci_lower: lower bound of confidence interval (optional)

    Returns:
    E-value and E-value for CI lower bound
    """
    if rr < 1:
        rr = 1 / rr
        if ci_lower is not None:
            ci_lower = 1 / ci_lower

    e_value = rr + np.sqrt(rr * (rr - 1))

    if ci_lower is not None and ci_lower > 1:
        e_value_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
        return e_value, e_value_ci

    return e_value, None

# Example
observed_rr = 2.5
ci_lower = 1.8

e_val, e_val_ci = calculate_e_value(observed_rr, ci_lower)

print(f"Observed Risk Ratio: {observed_rr}")
print(f"E-value: {e_val:.2f}")
print(f"E-value for CI lower bound: {e_val_ci:.2f}")
print(f"\nInterpretation: An unmeasured confounder would need to be")
print(f"associated with both treatment and outcome by a risk ratio of")
print(f"{e_val:.2f} to fully explain away the observed effect.")
```

## Sensitivity Analysis for DiD

For difference-in-differences, we can test sensitivity to violations of parallel trends:

```python
import numpy as np
import pandas as pd

def did_sensitivity_linear_trend(df, delta):
    """
    DiD sensitivity to linear pre-trend.

    Parameters:
    df: dataframe with columns ['time', 'treated', 'outcome']
    delta: assumed difference in linear trends

    Returns:
    Adjusted DiD estimate
    """
    # De-trend treatment group
    treatment_df = df[df['treated'] == 1].copy()
    treatment_df['outcome_adjusted'] = treatment_df['outcome'] - delta * treatment_df['time']

    # Recalculate DiD
    pre = df['time'] < df['time'].median()
    post = ~pre

    did_original = (
        df[post & (df['treated'] == 1)]['outcome'].mean() -
        df[pre & (df['treated'] == 1)]['outcome'].mean() -
        (df[post & (df['treated'] == 0)]['outcome'].mean() -
         df[pre & (df['treated'] == 0)]['outcome'].mean())
    )

    did_adjusted = (
        treatment_df[post]['outcome_adjusted'].mean() -
        treatment_df[pre]['outcome_adjusted'].mean() -
        (df[post & (df['treated'] == 0)]['outcome'].mean() -
         df[pre & (df['treated'] == 0)]['outcome'].mean())
    )

    return did_original, did_adjusted

# Test various levels of trend violation
for delta in [0, 0.1, 0.2, 0.5]:
    # Simulate data with pre-trend
    # ... (simulation code)
    print(f"Pre-trend difference: {delta}")
    # Calculate sensitivity
```

## Best Practices

1. **Report E-values or sensitivity parameters** with your estimates
2. **Consider realistic confounders**: Think about what variables you might have missed
3. **Benchmark against observed confounders**: Compare sensitivity bounds to effects of measured variables
4. **Conduct multiple sensitivity analyses**: Test different assumptions

## Further Reading

- VanderWeele, T. J., & Ding, P. (2017). "Sensitivity analysis in observational research: introducing the E-value"
- Rosenbaum, P. R. (2002). "Observational Studies"
- Cinelli, C., & Hazlett, C. (2020). "Making sense of sensitivity: extending omitted variable bias"
