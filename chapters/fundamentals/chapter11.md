# Chapter 11: Synthetic Control

When you have only a single treated unit — one country, one state, one company — standard DiD breaks down. You cannot reliably verify parallel trends with just one treated observation, and choosing a single control unit is arbitrary and potentially misleading. Synthetic control, introduced by Abadie and Gardeazabal (2003) and formalized by Abadie, Diamond, and Hainmueller (2010), sidesteps this problem by constructing a **weighted combination** of control units that mimics the pre-treatment trajectory of the treated unit as closely as possible. The resulting "synthetic" unit is the counterfactual — what the treated unit would have looked like without the treatment.

---

## Assumptions

Synthetic control works well as long as its assumptions hold. Before using SC, assess whether you can explain and defend each assumption to stakeholders — even if the assumptions are technically true, a lack of confidence in them undermines the credibility of your conclusions.

1. **Predictability**: X(t) (the donor/control time series) can be used to build a predictive model for Y(t) using pre-treatment data. The donor pool must actually track the treated unit's trajectory.

2. **No spillovers / no interference**: The event at time T has no effect whatsoever on the controls X(t). If treatment spills over into the donor pool (e.g., a competitor reacts to your product launch), the synthetic control is contaminated.

3. **Consistent relationship / no confounding events**: A stable relationship exists between Y(t) and X(t) — no other events near time T (besides the main event) distort the Y/X relationship. If another shock hits at the same time as treatment, you can't separate the two effects.

**For your Apple BD story**: be ready to explain why your donor pool satisfies assumption 2 (why the partners/markets you used as controls weren't affected by the same BD deals) and assumption 3 (no other concurrent events distorting the comparison).

---

## When DiD Fails: The Single Treated Unit Problem

DiD is powerful when you have many treated units, because averaging across them makes the parallel trends assumption more credible and gives you statistical power. But many important policy questions involve a single treated unit:

- California enacts a cigarette tax — how does it affect smoking?
- Germany reunifies — how does it affect West German GDP?
- One company launches a new product — what happens to its market share?

With a single treated unit, DiD faces severe limitations:

**Problem 1: Choosing controls is arbitrary.** If California is treated, which states do you use as controls? The Pacific states? All states? The choice is subjective, and different choices can yield very different estimates.

**Problem 2: Parallel trends is hard to justify.** With many treated units, pre-trends tests are informative. With one treated unit, the single pre-treatment trajectory could be due to noise, and there is no averaging over units to average out idiosyncratic shocks.

**Problem 3: Standard inference breaks down.** With one treated unit, you cannot use cluster-robust standard errors or central limit theorem arguments. You need a different inference strategy.

Synthetic control addresses all three problems:
- **Control selection is data-driven**: weights are chosen to minimize the pre-period discrepancy — no subjective choices.
- **Pre-period fit is explicit and auditable**: you can see exactly how well the synthetic unit matches the treated unit.
- **Inference via permutation**: a principled randomization-based approach to inference that doesn't rely on asymptotic theory.

---

## Selecting Training and Predicting Time Periods

Both periods must be chosen based on domain knowledge, but there's a tradeoff involved in selecting P_b^start (how far back the pre-treatment period begins):

| P_b^start choice | Effect |
|---|---|
| Far from T | More training data → lower variance, but Y/X relationship may weaken over time → more bias |
| Close to T | Relationship is stable and consistent → less bias, but small training set → high variance |

**Practical approach**: try different values of P_b^start, run cross-validation for each to estimate predictive capability, and choose the one with the lowest predictive error. This is valid because you're tuning a predictive model on pre-treatment data — not doing causal identification.

**Selecting P_a^start (post-treatment evaluation start)**: usually set immediately after T, but if the treatment effect is delayed (e.g., a campaign requires physical delivery before impact is measurable), set P_a^start farther from T to allow the effect to materialize. Choosing P_a^start too early when effects are delayed will underestimate the treatment effect. This should be driven by domain knowledge of the mechanism.

**Novelty vs. sustained effects — choosing P_a^end**: decide whether you want short-term, medium-term, or long-term effects. The farther P_a^start is from T, the more likely conditions surrounding Y(t) and X(t) will change, causing the model to lose accuracy.

**Critical limitation — post-treatment period is blind**: during pre-treatment training, you can use cross-validation to verify the model's predictive accuracy. In the post-treatment period, you're predicting a counterfactual that never happened — there's no ground truth to check against. You cannot validate the model's accuracy post-T. This means the validity of the post-treatment predictions rests entirely on domain expertise and the assumption that the Y/X relationship remains stable.

**Summary — consequences of moving each endpoint farther from T (Table 11.3):**

| Endpoint | Direction | Effect of moving farther from T |
|---|---|---|
| P_b^start | ← (further back) | More training data → lower variance, but context may change → less accuracy |
| P_b^end | ← (closer to T) | Less training data → higher variance, but more stable context → more accuracy. Moving P_b^end away from T risks capturing event anticipation effects |
| P_a^start | → (further from T) | Captures delayed effects, but less post-treatment data and higher risk of context change |
| P_a^end | → (further from T) | More data to evaluate long-term impact, but higher risk of Y/X relationship drifting |

**Key insight**: all four endpoints trade off sample size against context stability. The closer to T, the more stable the Y/X relationship — but the less data you have.

**Rule of thumb for pre-treatment length**: typically ≥ 10 periods to establish a reliable match (see DiD vs SC comparison table).

---

## The Synthetic Control Idea

The idea is disarmingly simple: instead of picking one control unit, build a **weighted average of all available control units** that best replicates the pre-treatment history of the treated unit.

**Intuition**: California in 1985 might be best approximated by a mixture of — say — 40% Colorado, 30% Utah, 20% Nevada, and 10% Montana. No single state matches California's pre-treatment trajectory, but a weighted combination can match it very closely. This weighted combination is the "synthetic California."

After treatment begins, synthetic California continues to evolve as a weighted average of the control states' outcomes. The gap between real California and synthetic California in the post period is the estimated treatment effect.

**Key properties of a good synthetic control:**
- Pre-treatment fit is close (low RMSPE)
- Weights are non-negative and sum to 1 (so the synthetic unit is an extrapolation-free convex combination)
- The donor pool consists of units not affected by the treatment

---

## Data Structure

Synthetic control data has the same unit × time structure as DiD, but with only **one treated unit**. Using the California Prop 99 smoking example:

**Long format** (for storage and exploration): one row per state × year.

| state | year | treated | post | Y (packs/capita/year) |
|-------|------|---------|------|-----------------------|
| California | 1985 | 1 | 0 | 116.1 |
| California | 1988 | 1 | 0 | 112.3 |
| California | 1989 | 1 | **1** | 103.5 |
| California | 2000 | 1 | 1 | 89.0 |
| Colorado | 1985 | 0 | 0 | 107.0 |
| Colorado | 1988 | 0 | 0 | 104.5 |
| Colorado | 1989 | 0 | 1 | 103.2 |
| Utah | 1985 | 0 | 0 | 78.2 |
| ... | ... | 0 | ... | ... |

**Wide format** (needed for weight optimization): one column per state, rows are time periods. The optimizer sees all donor states side-by-side.

| year | post | California | Colorado | Utah | Nevada | Montana |
|------|------|-----------|----------|------|--------|---------|
| 1970 | 0 | 127.1 | 120.3 | 95.4 | 130.2 | 88.1 |
| 1975 | 0 | 122.8 | 118.1 | 91.0 | 127.5 | 85.3 |
| 1988 | 0 | 116.0 | 115.2 | 89.3 | 125.1 | 82.4 |
| **1989** | **1** | 111.2 | 114.0 | 88.5 | 124.3 | 81.8 |
| 2000 | 1 | 90.1 | 112.0 | 86.0 | 122.0 | 80.1 |

The optimization uses **only the pre-treatment rows (post=0)** and **only the donor columns**. It finds weights w such that the weighted donor average matches the California column as closely as possible row by row:

```
w_CO × 120.3 + w_UT × 95.4 + w_NV × 130.2 + w_MT × 88.1 ≈ 127.1  (1970)
w_CO × 118.1 + w_UT × 91.0 + w_NV × 127.5 + w_MT × 85.3 ≈ 122.8  (1975)
w_CO × 115.2 + w_UT × 89.3 + w_NV × 125.1 + w_MT × 82.4 ≈ 116.0  (1988)
```

The weights are **frozen after training**. In post-treatment rows (1989+), apply the same weights to the donor columns to project the counterfactual. Treatment effect at each year:

`gap_t = Y_California_t − (w_CO × Y_Colorado_t + w_UT × Y_Utah_t + ...)`

At 2000: `89.0 − (w_CO × 112.0 + ...) ≈ 89.0 − 115.0 = −26 packs/capita`

---

## Formal Setup

### Notation

- $J+1$ units total: unit 1 is the **treated unit**, units $2, \ldots, J+1$ are the **donor pool** (controls)
- Time periods: $1, \ldots, T_0$ are pre-treatment; $T_0+1, \ldots, T$ are post-treatment
- $Y_{it}^N$: outcome for unit $i$ at time $t$ under no treatment (the potential outcome)
- $Y_{1t}^I$: outcome for the treated unit when treated (observed post-treatment)

The treatment effect at time $t > T_0$:

$$\tau_{1t} = Y_{1t}^I - Y_{1t}^N$$

We observe $Y_{1t}^I$ but need to estimate $Y_{1t}^N$ — the counterfactual. Synthetic control estimates it as:

$$\hat{Y}_{1t}^N = \sum_{j=2}^{J+1} \hat{w}_j \, Y_{jt}$$

### Finding the Weights

The weights $W = (w_2, \ldots, w_{J+1})$ are chosen to minimize the distance between the treated unit and the weighted combination of donors in the **pre-treatment period**:

$$\min_{W} \sum_{t=1}^{T_0} \left( Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt} \right)^2$$

subject to the constraints:
$$w_j \geq 0 \quad \text{for all } j, \qquad \sum_{j=2}^{J+1} w_j = 1$$

The non-negativity and sum-to-one constraints ensure the synthetic unit is a convex combination of donors — no extrapolation beyond the observed data.

In the original Abadie et al. formulation, the objective function also includes matching on pre-treatment **predictors** (not just outcome trajectories), using a $V$ matrix of feature weights:

$$\min_{W} \sum_{k=1}^{K} v_k \left( X_{1k} - \sum_{j=2}^{J+1} w_j X_{jk} \right)^2$$

where $X_{1k}$ are pre-treatment characteristics (including lagged outcomes). In practice, matching on lagged outcomes alone is often sufficient and simpler.

### The Estimated Treatment Effect

Once optimal weights $\hat{W}$ are found, the treatment effect at each post-treatment period is:

$$\hat{\tau}_{1t} = Y_{1t} - \sum_{j=2}^{J+1} \hat{w}_j Y_{jt}, \quad t = T_0+1, \ldots, T$$

The cumulative or average treatment effect:

$$\hat{\tau}_{\text{avg}} = \frac{1}{T - T_0} \sum_{t=T_0+1}^{T} \hat{\tau}_{1t}$$

---

## The California Smoking Example

Abadie, Diamond, and Hainmueller (2010) study California's **Proposition 99**, passed in November 1988. The proposition raised the cigarette tax by 25 cents per pack and allocated the revenue to anti-smoking programs. The goal: did Proposition 99 reduce per-capita cigarette consumption in California?

**The challenge**: every state experienced some trends in cigarette consumption during the 1980s–1990s (health awareness, other regulations, demographic changes). A naive before-after comparison for California would attribute all of the post-1988 decline to Prop 99, even if it was partly a national trend.

**The synthetic control approach**:
- **Treated unit**: California
- **Donor pool**: Other US states that did not pass major tobacco legislation during 1970–2000
- **Pre-treatment period**: 1970–1988
- **Post-treatment period**: 1989–2000
- **Outcome**: Per-capita cigarette sales (packs per year)

The algorithm finds that synthetic California is approximately a weighted average of Colorado, Connecticut, Montana, Nevada, New Mexico, and Utah — states that collectively matched California's smoking trajectory from 1970 to 1988.

**Results**:
- From 1970 to 1988, real and synthetic California track each other closely (pre-period RMSPE ≈ 1.6 packs)
- After 1988, real California's per-capita smoking falls sharply below synthetic California
- By 2000, the gap is about 26 packs per year per capita — this is the estimated effect of Prop 99

**Key finding**: California's smoking decline was substantially larger than what would have been expected based on comparable states. Prop 99 reduced per-capita consumption by roughly 26 packs per year — about a 25% reduction from the counterfactual.

---

## Python Implementation: Manual Synthetic Control

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATE DATA
# One treated unit (unit 0) receives treatment at period T0=15.
# 20 donor units form the control pool.
# True treatment effect: -5 per period post-treatment (e.g., reduced smoking).
# ══════════════════════════════════════════════════════════════════════════════

n_donors = 20
n_periods = 25
T0 = 15  # treatment starts after period 15 (periods 1..15 pre, 16..25 post)
true_effect = -5.0

# True weights used to generate the treated unit (mixture of donors 0, 3, 7)
true_weights = np.zeros(n_donors)
true_weights[0] = 0.40
true_weights[3] = 0.35
true_weights[7] = 0.25

# Generate donor trajectories (shared latent factors + unit-specific noise)
n_factors = 3
factor_loadings = np.random.normal(0, 1, (n_donors, n_factors))
factors = np.cumsum(np.random.normal(0, 0.5, (n_periods, n_factors)), axis=0)
donor_outcomes = factors @ factor_loadings.T  # shape: (n_periods, n_donors)
donor_outcomes += np.random.normal(0, 0.5, (n_periods, n_donors))  # idiosyncratic noise

# Treated unit = true weighted combination of donors + treatment effect + noise
treated_counterfactual = donor_outcomes @ true_weights
treatment_effect = np.array([true_effect if t >= T0 else 0.0 for t in range(n_periods)])
treated_outcomes = treated_counterfactual + treatment_effect + np.random.normal(0, 0.3, n_periods)

# Period index (1-based)
periods = np.arange(1, n_periods + 1)

# ══════════════════════════════════════════════════════════════════════════════
# FIND OPTIMAL WEIGHTS
# Minimize pre-treatment MSE subject to w >= 0, sum(w) = 1
# ══════════════════════════════════════════════════════════════════════════════

pre_mask = periods <= T0
Y_treated_pre = treated_outcomes[pre_mask]      # shape: (T0,)
Y_donors_pre  = donor_outcomes[pre_mask, :]     # shape: (T0, n_donors)

def objective(w):
    """Sum of squared pre-treatment prediction errors."""
    synth_pre = Y_donors_pre @ w
    return np.sum((Y_treated_pre - synth_pre) ** 2)

def gradient(w):
    synth_pre = Y_donors_pre @ w
    residuals = synth_pre - Y_treated_pre
    return 2 * (Y_donors_pre.T @ residuals)

# Constraints: w >= 0 (bounds) and sum(w) = 1
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * n_donors
w0 = np.ones(n_donors) / n_donors  # starting point: equal weights

result = minimize(
    objective,
    w0,
    jac=gradient,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 10000}
)

w_hat = result.x
print("Optimization converged:", result.success)
print(f"\nEstimated weights (donors with w > 0.01):")
for j, w in enumerate(w_hat):
    if w > 0.01:
        true_w = true_weights[j]
        print(f"  Donor {j:2d}: w_hat={w:.3f}  (true={true_w:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE SYNTHETIC CONTROL TRAJECTORY AND TREATMENT EFFECTS
# ══════════════════════════════════════════════════════════════════════════════

synth_outcomes = donor_outcomes @ w_hat       # synthetic unit for all periods
gap = treated_outcomes - synth_outcomes       # estimated treatment effect each period

# Pre-period RMSPE (goodness of fit)
pre_rmspe = np.sqrt(np.mean((treated_outcomes[pre_mask] - synth_outcomes[pre_mask]) ** 2))
post_rmspe = np.sqrt(np.mean((treated_outcomes[~pre_mask] - synth_outcomes[~pre_mask]) ** 2))
print(f"\nPre-period RMSPE:  {pre_rmspe:.4f}  (lower = better fit)")
print(f"Post-period RMSPE: {post_rmspe:.4f}")

avg_post_effect = gap[~pre_mask].mean()
print(f"\nAverage treatment effect (post): {avg_post_effect:.3f}  (true={true_effect})")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT: Actual vs. Synthetic
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(periods, treated_outcomes, "o-", color="coral", linewidth=2, label="Treated unit")
ax1.plot(periods, synth_outcomes, "s--", color="steelblue", linewidth=2, label="Synthetic control")
ax1.axvline(T0 + 0.5, color="gray", linestyle=":", linewidth=1.5, label=f"Treatment (T={T0})")
ax1.fill_betweenx([ax1.get_ylim()[0], ax1.get_ylim()[1]],
                   T0 + 0.5, n_periods + 0.5, alpha=0.07, color="coral")
ax1.set_xlabel("Period")
ax1.set_ylabel("Outcome")
ax1.set_title("Actual vs. Synthetic Control")
ax1.legend()

ax2 = axes[1]
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axvline(T0 + 0.5, color="gray", linestyle=":", linewidth=1.5, label=f"Treatment (T={T0})")
ax2.plot(periods, gap, "o-", color="darkgreen", linewidth=2, label="Gap (effect)")
ax2.fill_between(periods, gap, 0,
                  where=(periods > T0), alpha=0.15, color="darkgreen", label="Post-treatment gap")
ax2.set_xlabel("Period")
ax2.set_ylabel("Treated − Synthetic")
ax2.set_title("Treatment Effect Gap")
ax2.legend()

plt.suptitle("Synthetic Control: Actual vs. Synthetic Trajectory", fontsize=13)
plt.tight_layout()
plt.show()
```

---

## Inference via Permutation (Placebo Tests)

Standard statistical inference doesn't work with a single treated unit — we can't appeal to central limit theorems or asymptotic distributions. Instead, we use **permutation inference** (also called the "in-space placebo test").

### The Logic

Apply the exact same synthetic control procedure to every **donor unit**, treating each one as if it were the treated unit. For each donor unit, compute the "estimated treatment effect" gap at each period. These placebo gaps form a reference distribution under the null hypothesis of no treatment effect.

If the real treated unit's gap is unusually large compared to the placebo distribution, that's evidence of a genuine treatment effect.

**The key insight**: under the null hypothesis that treatment has no effect, the treated unit's gap should look like any other unit's gap. If the treated unit stands out, it's significant.

### The Spaghetti Plot

A standard visualization overlays all placebo paths (thin gray lines) with the real treated unit's path (bold colored line). If the treated unit's post-treatment gap is far outside the envelope of placebo paths, the effect is visually compelling.

### Excluding Poor-Fitting Placebos

Some donor units may have high pre-period RMSPE — the synthetic control didn't fit them well in the pre-period. These units have noisy, unreliable placebo gaps. Exclude donor units with pre-period RMSPE more than, say, 5× the treated unit's pre-period RMSPE from the reference distribution.

### p-value Calculation

A natural p-value: the fraction of units (treated + donors) whose post-treatment RMSPE exceeds the treated unit's post-treatment RMSPE (relative to their pre-period RMSPE):

$$p = \frac{\text{# units with post/pre RMSPE ratio} \geq \text{treated unit's ratio}}{J+1}$$

```python
# ══════════════════════════════════════════════════════════════════════════════
# PLACEBO TESTS: Apply synthetic control to each donor unit
# ══════════════════════════════════════════════════════════════════════════════

def fit_synthetic_control(Y_target_pre, Y_donors_pre, Y_donors_all):
    """
    Find optimal weights for a synthetic control and return the full synthetic path.
    Y_target_pre: (T0,) array of pre-treatment outcomes for the unit being matched
    Y_donors_pre: (T0, J) array of donor pre-treatment outcomes
    Y_donors_all: (T, J) array of all donor outcomes
    Returns: (weights, synthetic_full_trajectory)
    """
    n_d = Y_donors_pre.shape[1]

    def obj(w):
        return np.sum((Y_target_pre - Y_donors_pre @ w) ** 2)

    def grad(w):
        r = Y_donors_pre @ w - Y_target_pre
        return 2 * (Y_donors_pre.T @ r)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_d
    w0 = np.ones(n_d) / n_d
    res = minimize(obj, w0, jac=grad, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 5000})
    w_opt = res.x
    synth = Y_donors_all @ w_opt
    return w_opt, synth


# Pre-period RMSPE for the real treated unit
pre_rmspe_treated = pre_rmspe  # computed earlier

# Run placebo for each donor
placebo_gaps = []
placebo_pre_rmspes = []

for j in range(n_donors):
    # Donor j is treated; all other donors form the control pool
    other_donors = [d for d in range(n_donors) if d != j]
    Y_target_all = donor_outcomes[:, j]
    Y_pool_all   = donor_outcomes[:, other_donors]

    Y_target_pre_j = Y_target_all[pre_mask]
    Y_pool_pre_j   = Y_pool_all[pre_mask, :]

    _, synth_j = fit_synthetic_control(Y_target_pre_j, Y_pool_pre_j, Y_pool_all)
    gap_j = Y_target_all - synth_j
    pre_rmspe_j = np.sqrt(np.mean((Y_target_all[pre_mask] - synth_j[pre_mask]) ** 2))

    placebo_gaps.append(gap_j)
    placebo_pre_rmspes.append(pre_rmspe_j)
    print(f"  Donor {j:2d}: pre-RMSPE = {pre_rmspe_j:.3f}")

placebo_gaps = np.array(placebo_gaps)  # shape: (n_donors, n_periods)
placebo_pre_rmspes = np.array(placebo_pre_rmspes)

# ── Filter: exclude placebos with pre-RMSPE > 5x the treated unit's pre-RMSPE ─
rmspe_threshold = 5 * pre_rmspe_treated
good_placebos = placebo_pre_rmspes <= rmspe_threshold
print(f"\nRetaining {good_placebos.sum()}/{n_donors} placebo units (pre-RMSPE ≤ {rmspe_threshold:.2f})")

# ── Spaghetti plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for j in range(n_donors):
    if good_placebos[j]:
        ax.plot(periods, placebo_gaps[j], color="lightgray", linewidth=0.8, alpha=0.7, zorder=1)

# Real treated unit's gap
ax.plot(periods, gap, color="coral", linewidth=2.5, zorder=3, label="Treated unit")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(T0 + 0.5, color="gray", linestyle=":", linewidth=1.5, label=f"Treatment (T={T0})")

ax.set_xlabel("Period")
ax.set_ylabel("Gap (unit − synthetic)")
ax.set_title("Permutation Inference: Treated Unit vs. Placebo Distribution\n"
             "(Gray = donor placebo gaps; coral = real treated unit)")
ax.legend()
plt.tight_layout()
plt.show()

# ── Compute p-value: RMSPE ratio test ─────────────────────────────────────────
post_mask = periods > T0

def rmspe_ratio(outcomes, synth, pre_mask, post_mask):
    pre_r  = np.sqrt(np.mean((outcomes[pre_mask] - synth[pre_mask]) ** 2))
    post_r = np.sqrt(np.mean((outcomes[post_mask] - synth[post_mask]) ** 2))
    return post_r / (pre_r + 1e-10)

# Ratio for treated unit
ratio_treated = rmspe_ratio(treated_outcomes, synth_outcomes, pre_mask, post_mask)

# Ratios for good placebos
ratios_placebo = []
for j in range(n_donors):
    if good_placebos[j]:
        synth_j_path = donor_outcomes @ fit_synthetic_control(
            donor_outcomes[pre_mask, j],
            donor_outcomes[pre_mask, :][:, [d for d in range(n_donors) if d != j]],
            donor_outcomes[:, [d for d in range(n_donors) if d != j]]
        )[0]
        # Reuse already-computed gaps for efficiency
        target_j = donor_outcomes[:, j]
        synth_approx = target_j - placebo_gaps[j]
        ratios_placebo.append(rmspe_ratio(target_j, synth_approx, pre_mask, post_mask))

all_ratios = [ratio_treated] + ratios_placebo
p_value = np.mean([r >= ratio_treated for r in all_ratios])
print(f"\nPost/Pre RMSPE ratio — treated unit: {ratio_treated:.3f}")
print(f"Permutation p-value: {p_value:.3f}")
print(f"(Fraction of units with ratio >= treated unit's ratio)")
```

---

## Goodness of Fit: Pre-Period RMSPE

The **pre-period Root Mean Square Prediction Error (RMSPE)** measures how well the synthetic control fits the treated unit's pre-treatment history:

$$\text{RMSPE}_{\text{pre}} = \sqrt{\frac{1}{T_0} \sum_{t=1}^{T_0} \left(Y_{1t} - \sum_{j=2}^{J+1} \hat{w}_j Y_{jt}\right)^2}$$

**Why it matters**: if the pre-period RMSPE is large, the synthetic control is a poor match for the treated unit before treatment. In that case, the post-treatment gap could be driven by the pre-existing misfit rather than a genuine treatment effect. Results with poor pre-period fit are not credible.

**Practical thresholds**: there is no universal threshold, but as a guideline, the pre-period RMSPE should be small relative to the scale of the outcome and the post-period gap. If the pre-period RMSPE is comparable to the post-period gap, the synthetic control is not reliable.

**What to do if pre-period fit is poor**:
1. Expand the donor pool (more control units to draw from)
2. Extend the pre-treatment period (more periods to match on)
3. Include additional predictor variables in the matching criterion
4. Consider the Augmented Synthetic Control (see below) which adds a bias-correction term

```python
# Goodness-of-fit diagnostic
print(f"\nGoodness-of-fit summary:")
print(f"  Pre-period RMSPE:  {pre_rmspe:.4f}")
print(f"  Post-period RMSPE: {post_rmspe:.4f}")
print(f"  Ratio (post/pre):  {post_rmspe/pre_rmspe:.2f}")
print(f"  Average outcome scale: {np.abs(treated_outcomes).mean():.2f}")
print(f"  RMSPE as % of mean:    {100*pre_rmspe/np.abs(treated_outcomes).mean():.2f}%")

# Visual: pre-period fit
fig, ax = plt.subplots(figsize=(10, 4))
pre_periods = periods[pre_mask]
ax.plot(pre_periods, treated_outcomes[pre_mask], "o-", color="coral", label="Treated")
ax.plot(pre_periods, synth_outcomes[pre_mask], "s--", color="steelblue", label="Synthetic")
ax.fill_between(pre_periods,
                treated_outcomes[pre_mask], synth_outcomes[pre_mask],
                alpha=0.25, color="darkred", label="Prediction error")
ax.set_xlabel("Pre-treatment period")
ax.set_ylabel("Outcome")
ax.set_title(f"Pre-Period Fit: RMSPE = {pre_rmspe:.4f}")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Augmented Synthetic Control

The standard synthetic control works best when the donor pool is rich enough to match the treated unit's pre-treatment trajectory exactly. When exact pre-period match is impossible (e.g., the treated unit is an outlier), the synthetic control estimate has bias that comes from extrapolation.

**Ben-Michael, Feller, and Rothstein (2021)** propose the **Augmented Synthetic Control (ASC)**, which combines:

1. A standard synthetic control for the "interpolation" part
2. A **bias-correction term** estimated via an outcome model (typically a regularized regression like ridge regression) that accounts for the residual pre-period misfit

The ASC estimator:

$$\hat{\tau}_{1t}^{\text{ASC}} = \underbrace{\left(Y_{1t} - \sum_j \hat{w}_j Y_{jt}\right)}_{\text{standard SC gap}} + \underbrace{\hat{\mu}_{1t}^N - \sum_j \hat{w}_j \hat{\mu}_{jt}^N}_{\text{bias correction}}$$

where $\hat{\mu}_{it}^N$ is a predicted counterfactual from a ridge regression trained on the pre-treatment data.

**When to use ASC over standard SC:**
- When the donor pool is small or doesn't span the treated unit's space
- When the treated unit is unusual (extreme values of pre-treatment outcomes)
- When pre-period RMSPE is non-trivially large

The `SparseSC` package in Python implements the ASC and related methods.

---

## Improving Accuracy: Features, Regularization, and Practical Rules

### Adding Predictor Features (the V Matrix)

The original Abadie et al. (2010) formulation doesn't just match on outcome trajectory — it can also match on pre-treatment predictor variables (GDP, population, trade share, demographic characteristics) using a weighted feature matrix:

$$\min_W \sum_k v_k \left(X_{1k} - \sum_j w_j X_{jk}\right)^2$$

where $X_{1k}$ are pre-treatment characteristics of the treated unit, $X_{jk}$ are the same for each donor, and $v_k$ are feature weights (the V matrix). The outer optimization finds the V that minimizes pre-period prediction error; the inner optimization finds the W given V. In practice, matching on lagged outcomes alone is often sufficient — adding covariates helps mainly when outcome-only matching gives poor pre-period fit.

### Regularization on the Weights

The standard constraints (w_j ≥ 0 and Σw_j = 1) **are already a form of regularization.** They restrict the synthetic unit to a convex combination, preventing overfitting by design — no negative weights means no cancellation, and the synthetic unit stays within the space of observed donor trajectories.

When the donor pool is large relative to the number of pre-treatment periods, additional regularization options exist:

| Method | What it does | When to use |
|---|---|---|
| **Standard SC constraints** | Convex combination; prevents extrapolation | Default; always apply |
| **SparseSC (L1-style penalty)** | Pushes most weights to zero; produces sparse, interpretable weights | Large donor pool; want interpretability |
| **Augmented SC (ridge correction)** | Adds ridge regression bias-correction term on top of SC weights | Treated unit is outside donor convex hull; high pre-period RMSPE |
| **Matrix completion (Athey 2021)** | Frames SC as matrix completion with nuclear norm regularization | Missing data; panel with many units and times |

### Practical Decision Rule

**Start simple. Escalate only when pre-period fit is poor.**

1. **Run standard SC with outcome-only matching.** Check pre-period RMSPE relative to outcome scale.
   - RMSPE < 5% of mean outcome → fit is good, stop here
   - RMSPE ≥ 5% → proceed to step 2

2. **Expand the donor pool.** More control units give the optimizer more flexibility to match the treated unit's trajectory.

3. **Extend the pre-treatment period.** More training periods reduce variance in the weight estimates.

4. **Add predictor covariates (V matrix).** Include pre-treatment characteristics that explain the treated unit's trajectory and are available for all donors.

5. **Use Augmented SC.** If none of the above achieves low RMSPE, use ASC to add a ridge bias-correction term. This handles the case where the treated unit lies outside the donor convex hull.

**Key insight**: if pre-period RMSPE is already low with outcome-only matching, adding features or regularization adds complexity without meaningful benefit — and makes the method harder to explain to stakeholders. Transparency and simplicity are advantages of synthetic control; don't sacrifice them unnecessarily.

---

## Synthetic Control vs. DiD

| Dimension | DiD | Synthetic Control |
|---|---|---|
| **Number of treated units** | Many (or at least a few) | One (or very few) |
| **Control selection** | Researcher-chosen; parallel trends must hold | Data-driven weighted average of all donors |
| **Key assumption** | Parallel trends: same trend absent treatment | Pre-treatment fit: synthetic matches treated unit's trajectory |
| **Pre-treatment periods needed** | At least one; more is better | Many — typically ≥ 10 to establish a reliable match |
| **Inference** | Standard SE / clustered SE / bootstrap | Permutation (in-space placebo); no standard SE |
| **Transparency** | Treatment coefficient from regression | Explicit weights; which donors contribute is visible |
| **Effect dynamics** | Captured by event study | Estimated period-by-period post-treatment |
| **Applicable when** | Multiple treated units, parallel trends plausible | Single treated unit, rich pre-treatment history |
| **Extrapolation risk** | Regression can extrapolate | Convex weights prevent extrapolation |
| **Computation** | Simple OLS | Constrained optimization |

**Rule of thumb**: Use DiD when you have many treated units. Use synthetic control when you have one (or very few) treated units and a long pre-treatment history.

---

## Practical Considerations and Implementation

### The `pysyncon` Library

The `pysyncon` library provides a clean Python implementation of synthetic control following Abadie et al.:

```python
# pip install pysyncon
# from pysyncon import Dataprep, Synth
#
# dataprep = Dataprep(
#     foo=df,                        # your DataFrame
#     predictors=["GDP", "trade"],   # predictor variables
#     predictors_op="mean",          # how to aggregate predictors
#     time_predictors_prior=[1970, 1975, 1980, 1985],
#     special_predictors=[("Y", [1970, 1975, 1980, 1985, 1987], "mean")],
#     dependent="Y",
#     unit_variable="state",
#     time_variable="year",
#     treatment_identifier="California",
#     controls_identifier=["Colorado", "Utah", "Nevada", "Montana", ...]
# )
# synth = Synth()
# synth.fit(dataprep)
# synth.path_plot(time_period=range(1970, 2001), treatment_time=1988)
# synth.gaps_plot(time_period=range(1970, 2001), treatment_time=1988)
```

### Complete Self-Contained Example

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(99)

# ════════════════════════════════════════════════════════════════════════════════
# SCENARIO: A single market (Market A) launches a new product feature.
# 15 other markets serve as the donor pool.
# We want to estimate the feature's effect on weekly revenue (in $K).
# Treatment begins at week 20 (of 35 total weeks).
# ════════════════════════════════════════════════════════════════════════════════

n_donors  = 15
n_weeks   = 35
T0        = 20        # treatment begins after week 20
true_eff  = 8.0       # true treatment effect: +$8K/week

# Latent factor model for correlated market outcomes
n_factors = 4
F = np.cumsum(np.random.normal(0, 1, (n_weeks, n_factors)), axis=0)  # factor paths
Lambda_donors = np.abs(np.random.normal(1, 0.5, (n_donors, n_factors)))  # donor loadings
Lambda_treated = np.array([0.5, 0.3, 0.15, 0.05]) @ Lambda_donors[:4].T  # true treated = mixture

donor_Y = F @ Lambda_donors.T + np.random.normal(0, 1, (n_weeks, n_donors))
treated_base = F @ (0.5 * Lambda_donors[0] + 0.3 * Lambda_donors[3] +
                     0.15 * Lambda_donors[8] + 0.05 * Lambda_donors[12]) + \
               np.random.normal(0, 0.5, n_weeks)
# Add true effect post-treatment
treatment_vec = np.array([true_eff if t >= T0 else 0.0 for t in range(n_weeks)])
treated_Y = treated_base + treatment_vec

weeks = np.arange(1, n_weeks + 1)
pre_mask = weeks <= T0
post_mask = weeks > T0

# ── Fit synthetic control ─────────────────────────────────────────────────────
Y_treated_pre = treated_Y[pre_mask]
Y_donors_pre  = donor_Y[pre_mask, :]

def sc_objective(w):
    return np.sum((Y_treated_pre - Y_donors_pre @ w) ** 2)

def sc_gradient(w):
    r = Y_donors_pre @ w - Y_treated_pre
    return 2 * (Y_donors_pre.T @ r)

res = minimize(sc_objective, np.ones(n_donors)/n_donors, jac=sc_gradient,
               method="SLSQP",
               bounds=[(0,1)]*n_donors,
               constraints={"type":"eq","fun":lambda w: w.sum()-1},
               options={"ftol":1e-12, "maxiter":10000})

w_hat  = res.x
synth_Y = donor_Y @ w_hat
gap     = treated_Y - synth_Y

pre_rmspe  = np.sqrt(np.mean((treated_Y[pre_mask] - synth_Y[pre_mask])**2))
post_effect = gap[post_mask].mean()

print(f"Pre-period RMSPE: {pre_rmspe:.3f}")
print(f"Average post-treatment effect: {post_effect:.3f}  (true={true_eff})")
print(f"\nTop donor weights:")
top_donors = np.argsort(w_hat)[::-1][:5]
for d in top_donors:
    if w_hat[d] > 0.01:
        print(f"  Market {d+1}: weight = {w_hat[d]:.3f}")

# ── Plot: Actual vs. Synthetic ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(weeks, treated_Y, "o-", color="coral", linewidth=2, markersize=4, label="Market A (treated)")
ax1.plot(weeks, synth_Y,   "s--", color="steelblue", linewidth=2, markersize=4, label="Synthetic Market A")
ax1.axvline(T0+0.5, color="gray", linestyle=":", linewidth=1.5)
ax1.annotate("Feature launch", xy=(T0+0.5, ax1.get_ylim()[1]*0.95),
             xytext=(T0-6, ax1.get_ylim()[1]*0.95),
             arrowprops=dict(arrowstyle="->", color="gray"), color="gray")
ax1.fill_betweenx([min(treated_Y.min(), synth_Y.min())-1, max(treated_Y.max(), synth_Y.max())+1],
                   T0+0.5, n_weeks+0.5, alpha=0.06, color="coral")
ax1.set_ylabel("Revenue ($K/week)")
ax1.set_title("Synthetic Control: Market A Revenue\nActual vs. Synthetic")
ax1.legend()

ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.axvline(T0+0.5, color="gray", linestyle=":", linewidth=1.5)
ax2.plot(weeks, gap, "o-", color="darkgreen", linewidth=2, markersize=4, label="Gap (effect estimate)")
ax2.fill_between(weeks, gap, 0, where=post_mask, alpha=0.2, color="darkgreen")
ax2.set_xlabel("Week")
ax2.set_ylabel("Actual − Synthetic ($K)")
ax2.set_title(f"Treatment Effect Gap | Avg post-effect = {post_effect:.2f} | True = {true_eff}")
ax2.legend()

plt.tight_layout()
plt.show()

# ── Permutation inference ──────────────────────────────────────────────────────
placebo_gaps_mat = np.zeros((n_donors, n_weeks))
placebo_pre_rmspes = np.zeros(n_donors)

for j in range(n_donors):
    others = [d for d in range(n_donors) if d != j]
    Y_j_pre = donor_Y[pre_mask, j]
    Y_pool_pre = donor_Y[pre_mask, :][:, others]
    Y_pool_all = donor_Y[:, others]

    res_j = minimize(
        lambda w: np.sum((Y_j_pre - Y_pool_pre @ w)**2),
        np.ones(len(others))/len(others),
        method="SLSQP",
        bounds=[(0,1)]*len(others),
        constraints={"type":"eq","fun":lambda w: w.sum()-1},
        options={"ftol":1e-12, "maxiter":5000}
    )
    synth_j = Y_pool_all @ res_j.x
    placebo_gaps_mat[j] = donor_Y[:, j] - synth_j
    placebo_pre_rmspes[j] = np.sqrt(np.mean((donor_Y[pre_mask, j] - synth_j[pre_mask])**2))

# Spaghetti plot
fig, ax = plt.subplots(figsize=(12, 5))
rmspe_cutoff = 5 * pre_rmspe
for j in range(n_donors):
    if placebo_pre_rmspes[j] <= rmspe_cutoff:
        ax.plot(weeks, placebo_gaps_mat[j], color="lightgray", linewidth=0.8, alpha=0.8)
ax.plot(weeks, gap, color="coral", linewidth=2.5, label="Market A (treated)", zorder=5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(T0+0.5, color="gray", linestyle=":", linewidth=1.5, label="Feature launch")
ax.set_xlabel("Week")
ax.set_ylabel("Gap (unit − synthetic)")
ax.set_title("Permutation Test: Market A vs. Placebo Donor Gaps\n(gray = donor placebos, coral = Market A)")
ax.legend()
plt.tight_layout()
plt.show()

# p-value
post_rmspe_treated = np.sqrt(np.mean(gap[post_mask]**2))
rmspe_ratio_treated = post_rmspe_treated / (pre_rmspe + 1e-10)

placebo_ratios = []
for j in range(n_donors):
    if placebo_pre_rmspes[j] <= rmspe_cutoff:
        post_r_j = np.sqrt(np.mean(placebo_gaps_mat[j][post_mask]**2))
        placebo_ratios.append(post_r_j / (placebo_pre_rmspes[j] + 1e-10))

all_ratios = [rmspe_ratio_treated] + placebo_ratios
p_val = np.mean([r >= rmspe_ratio_treated for r in all_ratios])
print(f"\nPermutation p-value: {p_val:.3f}")
print(f"  (Fraction of units with post/pre RMSPE ratio >= treated unit's ratio)")
```

---

## Methods in Practice

Step-by-step checklist for running synthetic control:

1. **Identify treated unit and donor pool**: one treated unit; donors = units unaffected by treatment spillovers, with similar pre-treatment trajectories. Exclude donors with known interference.
2. **Choose time periods**: set P_b^start (training start), T (treatment date), P_a^end (evaluation end). Pre-treatment period should be ≥ 10 periods for a reliable match.
3. **Reshape to wide format**: rows = time periods, columns = units. The California column is the target; all other columns are donors.
4. **Optimize weights**: minimize pre-period MSE between the treated unit's column and a weighted average of donor columns, subject to w_j ≥ 0 and Σw_j = 1 (convex combination constraint). Use `scipy.optimize.minimize` with SLSQP.
5. **Check pre-period fit (RMSPE)**: compute `sqrt(mean((treated − synthetic)²))` over pre-treatment periods. If RMSPE is large relative to the outcome scale (>5%), the synthetic control is a poor match — expand donor pool, extend pre-period, or use Augmented SC.
6. **Project counterfactual**: apply frozen weights to donor columns in post-treatment rows. Do not refit.
7. **Compute gap**: `gap_t = treated_t − synthetic_t` at each post-period. Average gap = estimated treatment effect.
8. **Permutation inference**: apply steps 4–7 to each donor unit as a placebo (treating each donor as if it were the treated unit). Compare the treated unit's post/pre RMSPE ratio to the distribution of placebo ratios. Permutation p-value = fraction of units with ratio ≥ treated unit's ratio.
9. **Spaghetti plot**: overlay all placebo gap paths (gray lines) with the treated unit's gap path (colored line). Treated unit should stand clearly outside the placebo envelope post-treatment.

**What good output looks like:**
- Weights: CO=40%, UT=30%, NV=20%, MT=10% — sparse and interpretable
- Pre-period RMSPE: 1.6 packs (1.3% of mean outcome scale) → good fit
- Average treatment effect at 2000: −26 packs/capita
- Permutation p-value: p = 0.05 → treated unit's gap is unusual relative to all placebo paths

---

## Interview Questions

### Technical Questions

**Q1: What is the key assumption of synthetic control, and how does it differ from DiD's parallel trends assumption?**

Synthetic control assumes that the pre-treatment outcome trajectory of the treated unit can be well-approximated by a convex combination of donor units' trajectories. This is weaker than DiD's parallel trends assumption in one sense: you don't need the treated unit to have the same *level* or even the same *slope* as any single control unit — you need a weighted mixture to match the full trajectory. But it's stronger in another sense: you need the pre-treatment fit to be very close (low RMSPE) for the extrapolation to the post period to be credible. DiD's parallel trends is about a common trend; synthetic control is about matching the specific path of the treated unit.

---

**Q2: Why can't you use standard OLS standard errors for inference in synthetic control?**

Standard inference relies on asymptotics — either large samples (CLT) or large numbers of treated units (for clustering). With one treated unit, neither applies. There is no "average across many treated units" to average out idiosyncratic variation. The uncertainty comes entirely from the single treated unit's post-treatment outcome, which cannot be separated from noise. Permutation inference sidesteps this by asking: "Is the treated unit's estimated effect unusual relative to what we'd see if we applied the same method to untreated units?" This is a valid finite-sample test that does not require asymptotic arguments.

---

**Q3: What is the RMSPE ratio test and why is it preferred over raw post-treatment gap for the p-value?**

The raw post-treatment gap could be large simply because a unit is high-variance — not because it was actually affected by a treatment. The RMSPE ratio (post-period RMSPE / pre-period RMSPE) normalizes the post-treatment discrepancy by the pre-treatment discrepancy for each unit. For a placebo unit with a poor pre-period fit, the ratio would be near 1 (the same magnitude of error in both periods). For the truly treated unit, the ratio should be much greater than 1 (small pre-period error, large post-period effect). This normalization makes units comparable regardless of their individual variance levels.

---

**Q4: What are the constraints on the synthetic control weights and why?**

The constraints are $w_j \geq 0$ for all donors and $\sum_j w_j = 1$. Together these require the weights to form a **convex combination** — the synthetic unit lies within the convex hull of the donor units. The non-negativity constraint prevents the optimization from canceling out donor units with negative weights, which would amount to extrapolation beyond the observed data. If the treated unit lies outside the convex hull of the donors (e.g., it is an extreme outlier), no exact synthetic match exists and the pre-period RMSPE will be high — a diagnostic signal that the method may not be appropriate.

---

**Q5: A colleague suggests using DiD instead of synthetic control because it's simpler. When would you push back?**

Push back when: (1) you have only one or a handful of treated units — DiD with one treated unit has no basis for inference and the parallel trends assumption is very hard to verify; (2) the pre-treatment outcome trajectories of the treated and control units are not parallel — you'd fail the pre-trends test in DiD, whereas synthetic control can match a non-parallel trajectory by re-weighting; (3) you want the identifying assumption to be transparent and auditable — synthetic control makes explicit which donors are used and with what weights, whereas DiD just asserts parallel trends.

---

### Case Study Questions

**Case 1: Your company opens a flagship store in one city as a pilot. You have weekly sales data for 30 other cities going back 2 years. How would you use synthetic control to estimate the pilot's effect on sales?**

Setup: the pilot city is the treated unit ($J=1$); the 30 other cities are the donor pool ($J=30$). Pre-treatment period: the 2 years before store opening. Post-treatment: weeks after opening. Steps: (1) Collect weekly sales for all 31 cities. (2) Run the weight optimization to find the convex combination of the 30 cities that best matches the pilot city's pre-opening weekly trajectory. (3) Check pre-period RMSPE — if it's low (say, <5% of mean sales), the synthetic control is a good match. (4) Post-opening: compute the gap (actual pilot city − synthetic) each week — this is the treatment effect. (5) Run placebo tests on each donor city; check if the pilot city's gap stands out. (6) Report the average post-opening gap as the estimated sales lift, along with the permutation p-value. Important: check for spillovers (did the pilot city's store attract customers away from nearby donor cities?). If so, exclude geographically adjacent cities from the donor pool.

---

**Case 2: You apply synthetic control and find a large pre-period RMSPE. What does this mean and what do you do?**

A large pre-period RMSPE means the synthetic control doesn't fit the treated unit's pre-treatment history well — no convex combination of donors can reproduce the treated unit's trajectory. This makes the post-period estimate unreliable, because the "control" path is inaccurate even before treatment. Remedies: (1) Expand the donor pool — add more control units that are closer to the treated unit. (2) Extend the pre-treatment period — more data to match on. (3) Include additional predictor variables in the matching criterion (demographic features, lagged outcomes at specific time points). (4) Use the Augmented Synthetic Control, which adds a bias-correction term from a ridge regression to improve fit even when the convex hull constraint is binding. (5) If none of these work, acknowledge that synthetic control may not be applicable for this treated unit and consider other designs (DiD with a more restricted donor pool, interrupted time series).

---

**Case 3: You are presenting synthetic control results to a skeptical VP who asks "Why not just compare before vs. after for the one treated unit?" How do you respond?**

The before-after comparison for a single unit conflates the treatment effect with any other changes happening simultaneously — macroeconomic trends, seasonal patterns, competitor actions, platform-wide changes. For example, if the pilot city's sales increased 10% after the store opened, but all cities' sales increased 8% due to a national marketing campaign, the true effect of the pilot is only 2%. The synthetic control separates these by constructing a counterfactual that mimics what the pilot city's sales would have done absent the store opening, based on the trajectories of comparable cities. The pre-period fit shows explicitly how well this counterfactual tracks the pilot city historically, which is transparent and auditable — you can show the VP exactly how well the synthetic city matches the pilot city before the store opened. The before-after approach has no such check.

---

**Case 4: How does the synthetic control handle the case where the treatment gradually diffuses (not a sharp start date)?**

Standard synthetic control assumes a sharp, known treatment date — the weights are fixed at values learned in the pre-period, and the post-period gap is attributed entirely to treatment. With gradual rollout (e.g., a feature released to 20% of users in week 1, 60% in week 2, 100% in week 3), you have a few options: (1) Define the treatment date as the start of rollout, and interpret the post-period gaps as the average effect across adoption stages; (2) Use a time-varying treatment intensity measure and incorporate it into an extended model (though this departs from standard synthetic control); (3) Identify a "fully treated" period and focus the post-period analysis there; (4) Combine synthetic control with a dose-response model. The key is that the pre-period (before any rollout) must be clean — the weights should be estimated using only pre-rollout data, regardless of the rollout pace.
