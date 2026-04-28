# Chapter 12: Regression Discontinuity Design

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

---

## Data Structure

RDD has a fundamentally different structure from DiD and SC. There is **no panel** — each unit appears once. Treatment is determined entirely by whether the running variable crosses the cutoff, not by group membership or time period.

| unit_id | score (running var) | W (=1 if score ≥ 50) | Y (outcome) |
|---------|--------------------|-----------------------|-------------|
| student_01 | 42.3 | 0 | 101.2 |
| student_02 | 47.8 | 0 | 103.4 |
| student_03 | 49.1 | 0 | 104.1 |
| student_04 | 49.9 | 0 | 104.8 |
| **student_05** | **50.1** | **1** | **110.0** |
| student_06 | 50.4 | 1 | 110.3 |
| student_07 | 52.6 | 1 | 111.5 |
| student_08 | 57.3 | 1 | 113.9 |

Key difference from DiD: **there is no pre/post period.** W is a deterministic function of score: `W = 1 if score ≥ 50, else 0`. The identification comes from units just below vs. just above the cutoff being otherwise comparable.

Notice the jump: student_04 (score=49.9, W=0, Y=104.8) vs. student_05 (score=50.1, W=1, Y=110.0). A 0.2-point score difference produces a ~5-point jump in Y — that discontinuity is the treatment effect.

What the data looks like visually:

```
Y
115 ─┤                      ....../ ← treated side (W=1)
     │                ......
110 ─┤          ......
     │                          ← JUMP at cutoff ≈ 5 pts (= a₁)
106 ─┤ × ← predicted from control side at score=50
     │.....
103 ─┤     control side (W=0)
     │
     +─────┬────┬────┬────┬──|──┬────┬────→ score
          44   46   48   50  52  54
                              ↑
                           cutoff
```

The regression model fit to this data — one row per observation:

`Y = a₀ + a₁·W + a₂·(score − 50) + a₃·(score − 50)·W + ε`

- `a₀` = predicted Y at the cutoff, control side
- `a₁` = **jump at cutoff = treatment effect** (what you report)
- `a₂` = slope of the control regression line
- `a₃` = additional slope change on the treated side

In practice, fit this model using only observations within a bandwidth window (e.g., score ∈ [45, 55]). Units far from the cutoff add bias without improving comparability.

---

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

## Formal Identification

### Continuity Assumption (Potential Outcomes Form)

RDD rests on two continuity conditions at the threshold T:

**Y₁ is continuous at T:**
$$\lim_{s \to 0} E[Y_1(T - s)] = \lim_{s \to 0} E[Y_1(T + s)]$$

**Y₀ is continuous at T, and equals observed Y from the left (control side):**
$$E[Y_0(T)] = \lim_{s \to 0} E[Y_0(T - s)] = \lim_{s \to 0} E[Y(T - s)]$$

The second equation is the key identification step: we cannot observe the counterfactual Y₀(T) for treated units, but continuity lets us use the observed outcomes of control units just below T as a stand-in.

### The RDD Estimand (Equation 11.1)

Combining both continuity conditions:

$$\text{ATE} = E[Y_1(T)] - E[Y_0(T)] = \lim_{s \to 0} E[Y(T+s)] - \lim_{s \to 0} E[Y(T-s)]$$

Both sides are now observable — right-limit minus left-limit of *observed* Y at the cutoff. **Important**: this is a *local* ATE at the threshold T, not the full population ATE. Units far from the cutoff tell you nothing.

---

## Linear Model Specification

The standard RDD regression is a single model with an interaction term — it looks like two lines in a plot but is one equation:

$$Y^i = a_0 + a_1 D^i + a_2(t^i - T) + a_3(t^i - T)D^i + \varepsilon^i(t)$$

| Term | Meaning |
|------|---------|
| $a_0$ | Baseline intercept (control group at cutoff) |
| $a_1 D$ | **Jump at the cutoff = treatment effect** |
| $a_2(t - T)$ | Pre-period slope (control group trend) |
| $a_3(t - T)D$ | Interaction — allows slope to differ post-treatment |

**D = 0 (pre/control side):** $Y = a_0 + a_2(t - T)$ — intercept $a_0$, slope $a_2$

**D = 1 (post/treated side):** $Y = (a_0 + a_1) + (a_2 + a_3)(t - T)$ — intercept shifts by $a_1$, slope shifts by $a_3$

**$a_1$ is what you report as the treatment effect.** Centering on $(t - T)$ ensures this cleanly equals the jump at the cutoff.

---

## Bandwidth Selection

In practice, you restrict estimation to units within a window $w$ of the cutoff (the bandwidth). There is a fundamental bias-variance tradeoff:

| Bandwidth | Bias | Variance |
|-----------|------|----------|
| Smaller $w$ | Lower — units near cutoff are more comparable | Higher — fewer observations |
| Larger $w$ | Higher — units farther from cutoff may differ on unobservables | Lower — more observations |

**The catch**: you cannot objectively optimize $w$ because the bias comes from *unobservable* factors — by definition, you can't measure what you can't observe. Data-driven bandwidth selectors (e.g., Imbens-Kalyanaraman optimal bandwidth) minimize MSE under assumptions about the smoothness of the outcome function, but the assumption itself is untestable.

**Practical approach**: report results across a range of bandwidths as a sensitivity check (bandwidth sensitivity plot). If the estimate is stable across different $w$, that's reassuring. If it's sensitive, acknowledge fragility. In synthetic data experiments, smaller bandwidths tend to produce less-biased estimates but with wider confidence intervals — in real data, you can't see the "true" line, so stability across bandwidths is your only diagnostic.

---

## Assumptions

1. **Continuity**: Potential outcomes Y₀ and Y₁ are continuous at the cutoff (formal form above)
2. **No manipulation**: Units can't precisely control the running variable to sort above/below cutoff
3. **Local effect**: Identifies treatment effect near the cutoff only — not generalizable to the full population

## Nonlinear Models and Local Linear Regression

**Can you use ML?** Yes — fit a separate model (random forest, boosting, deep learning) on each side of the cutoff using data before and after T. Use cross-validation to tune each model, then predict both at time T and take the difference. This gives a valid estimator of the local ATE at T.

**Why LLR is preferred over ML in practice**: ML models are black boxes — hard to analyze and interpret. **Local linear regression (LLR)** (or local polynomial regression, LPR) is the standard choice because:
- It provides explicit formulas
- Its behavior is well-understood analytically
- It is often *more accurate* than ML alternatives near a boundary

LLR fits a linear regression using only observations within the bandwidth window, separately on each side. The treatment effect is the difference in predicted values at the cutoff.

**Kernel weighting** (how `rdrobust` works): rather than treating all observations within the window equally, assign higher weights to observations closer to the cutoff t₀ and lower weights to those farther away. The formal objective function minimized on each side:

$$\sum_t K_h(t - t_0)\left[Y(t) - (a_0 + a_1(t - t_0))\right]^2$$

where $K_h(t - t_0)$ is the kernel weight (higher near the cutoff, lower farther away; $h$ is the bandwidth) and the bracketed term is the squared residual from the local linear model. Minimizing this yields $a_0$ and $a_1$ for each side. At $t = t_0$, the fitted value is simply $a_0$ (since $t - t_0 = 0$) — so **$a_0$ is the predicted Y at the cutoff**, and the treatment effect = $a_0^{\text{treated}} - a_0^{\text{control}}$.

**Local polynomial regression (LPR)**: LLR can be generalized to polynomials instead of lines. `rdrobust` supports this, but **use polynomials of degree ≤ 2 only** — higher-order polynomials are unstable near the boundary and can produce misleading estimates (Gelman and Imbens). In practice, local linear (degree 1) is the default and usually sufficient.

---

## Methods in Practice

Step-by-step checklist for running an RDD analysis:

1. **Identify the running variable and cutoff**: what continuous score determines treatment? What is the exact threshold c?
2. **Plot Y vs. running variable**: scatter plot with a vertical line at c. A visible jump at the cutoff is the treatment effect. If there's no visual discontinuity, the effect is likely small or zero.
3. **McCrary density test (manipulation check)**: test whether the density of the running variable is smooth at the cutoff. A spike just above c means units are sorting above the threshold — no-manipulation assumption violated, RDD invalid.
4. **Covariate balance at the cutoff**: check that pre-determined covariates (age, gender, baseline characteristics) are continuous at c. A covariate jump = a confound, not just treatment.
5. **Choose bandwidth h**: restrict to observations within [c−h, c+h]. Use `rdrobust` for the Imbens-Kalyanaraman optimal bandwidth. Smaller h → less bias, more variance. Larger h → more data, more bias from units less comparable.
6. **Fit local linear regression**: separate OLS on each side within the bandwidth window. With kernel weighting (triangular kernel), units closer to the cutoff get higher weight.
7. **Estimate treatment effect**: the jump at the cutoff = predicted Y just above c minus predicted Y just below c = coefficient a₁ in the regression `Y = a₀ + a₁·W + a₂·(X−c) + a₃·(X−c)·W + ε`.
8. **Bandwidth sensitivity plot**: rerun steps 5–7 for a range of bandwidths (e.g., h/2, h, 2h). Plot the estimate vs. bandwidth. Stable = robust; highly sensitive = fragile, report the variation.
9. **Placebo threshold tests**: apply the same estimation at fake cutoff values (e.g., c±5, c±10) where no treatment was assigned. Should find no significant discontinuities.

**What good output looks like:**
- Optimal bandwidth: h = 5 (from rdrobust)
- Treatment effect: 5.2, SE = 0.7, 95% CI [3.8, 6.6], p < 0.001
- McCrary test: p = 0.43 → no evidence of manipulation
- Bandwidth sensitivity: estimate stable from h = 3 to h = 8 → robust result
- Placebo thresholds at score = 45 and 55: p = 0.62, p = 0.81 → no spurious discontinuities

---

## Validity Checks

- **McCrary density test**: Check for manipulation (bunching at cutoff)
- **Covariate balance**: Covariates should be continuous at cutoff
- **Placebo tests**: No discontinuity at other thresholds
