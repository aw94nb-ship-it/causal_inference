# Chapter 10: Difference-in-Differences

Difference-in-Differences (DiD) is one of the most widely used quasi-experimental designs in economics, policy research, and industry. The core idea is elegant: when a treatment affects some units but not others, and you observe everyone before and after, you can use the change over time in the untreated group as a stand-in for what would have happened to the treated group had there been no treatment. The "difference in differences" is then the treated group's actual change minus that counterfactual change — isolating the treatment effect from any background trends.

---

## The Setup

DiD applies when you have all three of the following:

1. **Panel data** — the same units (states, users, stores, countries) are observed in multiple time periods.
2. **A known treatment event** — at some point in time $T_0$, a subset of units receives a treatment. The others do not.
3. **Variation in treatment assignment** — some units are treated ("treatment group"), others are not ("control group").

The key insight is that you don't need the treatment and control groups to look identical before treatment. You only need them to be **moving in parallel** — so that you can use the control group's trajectory as the counterfactual for the treated group.

**Business examples where DiD applies:**
- A product feature launches in some markets before others (phased rollout)
- A state raises its minimum wage while neighboring states do not
- A company runs a promotional email campaign for a random subset of users one week

**Why not just compare treated units before vs. after?**

That approach conflates the treatment effect with any other changes happening over time (seasonal trends, macroeconomic conditions, product changes). The control group absorbs all that background noise.

**Why not just compare treated vs. control in the post period?**

That approach confounds pre-existing differences between the groups with the treatment effect. DiD removes both sources of bias simultaneously.

---

## Data Structure

Every DiD dataset has the same core structure: one row per unit × time period. Using the NJ/PA minimum wage example with actual numbers:

| unit_id | period | treated | post | D (treated×post) | Y (FTE) |
|---------|--------|---------|------|------------------|---------|
| NJ_01 | Feb-1992 | 1 | 0 | 0 | 20.1 |
| NJ_01 | Nov-1992 | 1 | 1 | **1** | 21.5 |
| NJ_02 | Feb-1992 | 1 | 0 | 0 | 19.8 |
| NJ_02 | Nov-1992 | 1 | 1 | **1** | 22.0 |
| PA_01 | Feb-1992 | 0 | 0 | 0 | 23.1 |
| PA_01 | Nov-1992 | 0 | 1 | 0 | 21.0 |
| PA_02 | Feb-1992 | 0 | 0 | 0 | 23.6 |
| PA_02 | Nov-1992 | 0 | 1 | 0 | 21.3 |

Column definitions:
- **treated** (time-invariant): 1 if the unit is in the treatment group (NJ), 0 if control (PA). Same value in both rows for each unit.
- **post** (unit-invariant): 1 if this row is the post-treatment period, 0 if pre. Same for all units in a given period.
- **D = treated × post**: the treatment indicator. **Only = 1 for treated units in the post period.** This is the column whose OLS coefficient gives the DiD estimate.
- **Y**: the observed outcome (fast food FTE employment).

The 2×2 structure maps directly to the estimator:

| | pre (post=0) | post (post=1) | Change |
|---|---|---|---|
| **treated=1 (NJ)** | 20.0 | 21.8 | +1.8 |
| **treated=0 (PA)** | 23.3 | 21.2 | −2.1 |

DiD = 1.8 − (−2.1) = **+3.9 FTE**

The regression runs on every row in the full table. You do not collapse to group means first. OLS on the full panel with `Y ~ treated + post + D` recovers +3.9 exactly.

---

## The Classic 2×2 DiD

The simplest DiD setup has two groups and two time periods:

| | Pre-treatment | Post-treatment |
|---|---|---|
| **Treatment group** | $\bar{Y}_{T,\text{pre}}$ | $\bar{Y}_{T,\text{post}}$ |
| **Control group** | $\bar{Y}_{C,\text{pre}}$ | $\bar{Y}_{C,\text{post}}$ |

The **DiD estimator** is:

$$\hat{\tau}_{DiD} = (\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}) - (\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}})$$

**Why does this work?** Break it down:

- $\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}$ = the treated group's change = (true effect) + (background trend)
- $\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}}$ = the control group's change = (background trend only)
- Difference of differences = (effect + trend) - (trend) = **effect**

This is the counterfactual logic: we observe the treated group changed by some amount. We subtract the change that would have happened anyway (as measured by the control group). What remains is the treatment effect.

### The Card & Krueger (1994) Example

David Card and Alan Krueger studied the effect of a minimum wage increase in New Jersey in 1992. Pennsylvania (a neighboring state) did not raise its minimum wage. They compared fast food employment in NJ vs. PA before and after the NJ increase:

| | Pre (Feb 1992) | Post (Nov 1992) |
|---|---|---|
| **NJ (treated)** | 20.44 FTE | 21.03 FTE |
| **PA (control)** | 23.33 FTE | 21.17 FTE |

DiD = (21.03 - 20.44) - (21.17 - 23.33) = 0.59 - (-2.16) = **+2.75 FTE**

Despite the minimum wage increase, NJ employment actually *rose* relative to PA. This famous paper challenged the standard economic prediction that minimum wages reduce employment.

### Regression Form

The 2×2 DiD has a direct regression representation. Define:
- $\text{Treated}_i \in \{0,1\}$: whether unit $i$ is in the treatment group (time-invariant)
- $\text{Post}_t \in \{0,1\}$: whether period $t$ is after the treatment
- $\text{Treated}_i \times \text{Post}_t$: the interaction term — equals 1 only for treated units in the post period

The model:

$$Y_{it} = \alpha + \beta \, \text{Treated}_i + \gamma \, \text{Post}_t + \delta \, (\text{Treated}_i \times \text{Post}_t) + \varepsilon_{it}$$

Each coefficient has a clear interpretation:
- $\alpha$: control group baseline (pre-period mean)
- $\beta$: pre-existing level difference between treated and control
- $\gamma$: time trend shared by both groups (pre $\to$ post change in control)
- $\delta$: **the DiD estimate** — additional change in treated group relative to control

The coefficient $\delta$ equals $\hat{\tau}_{DiD}$ from the table formula exactly.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# ── Simulate 2x2 DiD panel data ──────────────────────────────────────────────
# 100 units, 2 periods, ~50% treated
n_units = 100
unit_ids = np.arange(n_units)
treated = (unit_ids >= 50).astype(int)   # units 50-99 are treated

# Stack into a panel: each unit appears in period 0 (pre) and period 1 (post)
df = pd.DataFrame({
    "unit":    np.tile(unit_ids, 2),
    "period":  np.repeat([0, 1], n_units),
    "treated": np.tile(treated, 2),
})
df["post"] = df["period"]  # post = period 1

# True data-generating process
# Unit-level fixed effect (pre-existing differences)
unit_fe = np.tile(np.random.normal(0, 2, n_units), 2)
# Common time trend (affects everyone)
time_trend = df["post"] * 1.5
# True treatment effect = 3.0 for treated units in post period
true_effect = 3.0
treatment_indicator = df["treated"] * df["post"]
noise = np.random.normal(0, 1, len(df))

df["Y"] = 10 + unit_fe + time_trend + true_effect * treatment_indicator + noise

# ── Manual DiD from group means ───────────────────────────────────────────────
means = df.groupby(["treated", "post"])["Y"].mean().unstack()
print("Group means:")
print(means.round(3))

diff_treated = means.loc[1, 1] - means.loc[1, 0]
diff_control = means.loc[0, 1] - means.loc[0, 0]
did_manual = diff_treated - diff_control
print(f"\nChange in treated:  {diff_treated:.3f}")
print(f"Change in control:  {diff_control:.3f}")
print(f"DiD (manual):       {did_manual:.3f}  (true = {true_effect})")

# ── OLS regression form ───────────────────────────────────────────────────────
model = smf.ols("Y ~ treated + post + treated:post", data=df).fit()
print("\nRegression results:")
print(model.summary().tables[1])
# The 'treated:post' coefficient should match did_manual
```

---

## The Parallel Trends Assumption

DiD's identifying assumption is **parallel trends**: in the absence of treatment, the treatment and control groups would have followed the same time trend.

Formally, let $Y_{it}(0)$ be unit $i$'s potential outcome under no treatment. Parallel trends states:

$$E[Y_{it}(0) - Y_{it'}(0) \mid \text{Treated}_i = 1] = E[Y_{it}(0) - Y_{it'}(0) \mid \text{Treated}_i = 0]$$

for any two periods $t, t'$. In plain English: the average change in the untreated potential outcome is the same for both groups.

**Important: this is about trends, not levels.** The treatment and control groups can have very different pre-treatment outcome levels — that's captured by $\beta$ in the regression. What matters is that they're changing at the same rate.

**This assumption is partially untestable.** In the post-treatment period, you never observe what the treated group would have done without treatment — that's the fundamental identification challenge. But you can test it in the pre-treatment periods (see Pre-Trends Test below).

**When is parallel trends plausible?**
- Treatment was assigned in a way that's unrelated to recent outcome trends
- The treated and control units are from the same broader population (e.g., neighboring states)
- Treatment was not triggered by a sudden outcome shock ("Ashenfelter's dip" problem)

**When is parallel trends suspect?**
- Self-selection into treatment based on recent poor performance (regression to the mean)
- Treatment and control groups are in fundamentally different economic sectors
- The treatment was implemented in response to an ongoing trend

**Visualizing parallel trends** — a good diagnostic is a simple line plot of average outcomes over time, with separate lines for treated and control groups. In pre-treatment periods, the lines should be roughly parallel (same slope). If they're converging or diverging before treatment, the parallel trends assumption is in doubt.

```python
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate multi-period data for visualization
n_units = 60
n_periods = 8          # periods 0-7; treatment starts at period 4
treated_ids = np.arange(30)

records = []
for unit in range(n_units):
    is_treated = int(unit < 30)
    unit_fe = np.random.normal(is_treated * 2, 1)  # treated group is higher baseline
    for t in range(n_periods):
        post = int(t >= 4)
        # Common time trend: +0.5 per period for everyone
        trend = 0.5 * t
        # Treatment effect: +3 for treated units in post periods
        effect = 3.0 * is_treated * post
        y = 10 + unit_fe + trend + effect + np.random.normal(0, 0.8)
        records.append({"unit": unit, "period": t, "treated": is_treated, "Y": y, "post": post})

panel = pd.DataFrame(records)

avg_by_group = panel.groupby(["period", "treated"])["Y"].mean().unstack()
avg_by_group.columns = ["Control", "Treated"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_by_group.index, avg_by_group["Control"], "o-", color="steelblue",
        label="Control group", linewidth=2)
ax.plot(avg_by_group.index, avg_by_group["Treated"], "s-", color="coral",
        label="Treated group", linewidth=2)
ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1.5, label="Treatment starts")
ax.fill_betweenx([ax.get_ylim()[0], 30], 3.5, 7.5, alpha=0.07, color="coral")

# Annotate the DiD gap
post_avg = avg_by_group.loc[7]
ax.annotate("", xy=(7, post_avg["Treated"]), xytext=(7, post_avg["Control"]),
            arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2))
ax.text(7.1, (post_avg["Treated"] + post_avg["Control"]) / 2, "DiD\neffect",
        color="darkgreen", fontsize=10)

ax.set_xlabel("Period")
ax.set_ylabel("Average outcome Y")
ax.set_title("Parallel Trends: Treatment vs. Control Over Time")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Pre-Trends Test

The pre-trends test checks whether the treatment and control groups were following parallel trends **before** the treatment. It can't prove parallel trends hold in the post period, but a failed pre-trend test is strong evidence that DiD is invalid.

### How to Run the Pre-Trends Test

Take only the pre-treatment periods. Interact time (as a set of period indicators) with the treated group indicator. If the groups were truly on parallel trends, none of these interactions should be statistically significant.

**Regression form** (for pre-treatment periods only):

$$Y_{it} = \alpha_i + \lambda_t + \sum_{k < 0} \beta_k \cdot (\text{Treated}_i \times \mathbf{1}[t = k]) + \varepsilon_{it}$$

All $\beta_k$ should be close to zero and statistically insignificant.

**Critical warning**: A failed pre-trends test doesn't mean you "adjust" and proceed. It means the DiD design is invalid for this setting. You need a different identification strategy.

```python
from scipy import stats

np.random.seed(42)

# Build a dataset with a clear parallel trends violation
# Treatment group is on a steeper upward trajectory in pre-period
n_units = 80
n_pre = 5   # periods 0..4 are pre-treatment

records = []
for unit in range(n_units):
    is_treated = int(unit < 40)
    unit_fe = np.random.normal(0, 1)
    for t in range(n_pre):
        # Violation: treated group has an extra +0.4 per period pre-treatment
        pre_trend_violation = 0.4 * t * is_treated
        y = 10 + unit_fe + 0.3 * t + pre_trend_violation + np.random.normal(0, 0.8)
        records.append({"unit": unit, "period": t, "treated": is_treated, "Y": y})

pre_df = pd.DataFrame(records)
# Add period dummies interacted with treated
for t in range(1, n_pre):
    pre_df[f"treated_x_t{t}"] = pre_df["treated"] * (pre_df["period"] == t).astype(int)

formula = "Y ~ treated + C(period) + " + " + ".join([f"treated_x_t{t}" for t in range(1, n_pre)])
pre_model = smf.ols(formula, data=pre_df).fit()

print("Pre-trends test (interaction coefficients for treated × period):")
interaction_coefs = pre_model.params[[c for c in pre_model.params.index if "treated_x" in c]]
interaction_pvals = pre_model.pvalues[[c for c in pre_model.pvalues.index if "treated_x" in c]]
for name, coef, pval in zip(interaction_coefs.index, interaction_coefs, interaction_pvals):
    sig = "*** SIGNIFICANT — PARALLEL TRENDS VIOLATED" if pval < 0.05 else ""
    print(f"  {name}: coef={coef:.3f}, p={pval:.3f}  {sig}")

# ── Visualize pre-trends ──────────────────────────────────────────────────────
avg_pre = pre_df.groupby(["period", "treated"])["Y"].mean().unstack()
avg_pre.columns = ["Control", "Treated"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(avg_pre.index, avg_pre["Control"], "o-", color="steelblue", label="Control", linewidth=2)
ax.plot(avg_pre.index, avg_pre["Treated"], "s-", color="coral", label="Treated", linewidth=2)
ax.set_xlabel("Pre-treatment period")
ax.set_ylabel("Average Y")
ax.set_title("Pre-Trends Test: Diverging Trends — DiD Invalid Here")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Event Study Design

Instead of collapsing the treatment effect into a single number, an **event study** estimates the treatment effect at each period relative to treatment. This serves two purposes:

1. **Pre-treatment periods**: tests the parallel trends assumption — coefficients should be near zero
2. **Post-treatment periods**: shows how the effect evolves — does it grow, shrink, or stay constant?

### Model

Let $T_i^*$ be the treatment date for unit $i$ (undefined for never-treated units). Define relative time $k = t - T_i^*$. The event study model:

$$Y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \delta_k \cdot \mathbf{1}[t - T_i^* = k] + \varepsilon_{it}$$

where:
- $\alpha_i$ = unit fixed effects (absorb time-invariant unit differences)
- $\lambda_t$ = time fixed effects (absorb common trends)
- $k = -1$ is the **reference period** (omitted to avoid multicollinearity)
- $\delta_k$ is the treatment effect at time $k$ relative to treatment

**Interpretation:**
- $\delta_k \approx 0$ for $k < 0$: parallel trends holds (no pre-treatment differences)
- $\delta_k \neq 0$ for $k \geq 0$: treatment has an effect post-treatment
- The pattern of $\delta_k$ post-treatment reveals the effect's dynamics

```python
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

np.random.seed(0)

# ── Simulate event study data ─────────────────────────────────────────────────
n_units = 60
treatment_period = 5   # all treated units treated at period 5
n_periods = 10         # periods 0..9

records = []
for unit in range(n_units):
    is_treated = int(unit < 30)
    unit_fe = np.random.normal(is_treated * 1.5, 1.5)
    for t in range(n_periods):
        rel_time = t - treatment_period if is_treated else None
        # Treatment effect grows post-treatment: 0 at k=0, then +1 per period
        if is_treated and t >= treatment_period:
            effect = (t - treatment_period) * 1.2
        else:
            effect = 0
        y = 8 + unit_fe + 0.4 * t + effect + np.random.normal(0, 1)
        records.append({
            "unit": unit, "period": t, "treated": is_treated,
            "Y": y, "rel_time": (t - treatment_period) if is_treated else np.nan
        })

es_df = pd.DataFrame(records)

# Create relative time dummies (exclude k=-1 as reference)
# For never-treated units, code them as a far-out value and exclude from sum
# Simple approach: restrict to treated units and use period FE + unit FE
treated_df = es_df[es_df["treated"] == 1].copy()
treated_df["rel_time_int"] = treated_df["rel_time"].astype(int)
treated_df["rel_time_str"] = "k_" + treated_df["rel_time_int"].astype(str).str.replace("-", "m")

# Use full panel but code relative time indicators for treated units only
es_df["rel_time_int"] = es_df["rel_time"].fillna(-999).astype(int)

# Dummies for all relative periods except -1
rel_periods = sorted(es_df.loc[es_df["treated"] == 1, "rel_time_int"].unique())
rel_periods_no_ref = [k for k in rel_periods if k != -1]

for k in rel_periods_no_ref:
    col = f"d_k{'m' if k < 0 else ''}{abs(k)}"
    es_df[col] = ((es_df["treated"] == 1) & (es_df["rel_time_int"] == k)).astype(int)

dummy_cols = [f"d_k{'m' if k < 0 else ''}{abs(k)}" for k in rel_periods_no_ref]

# OLS with unit and period FE (demeaning approach for simplicity)
# Absorb unit FE by demeaning within unit
def within_demean(df, cols, group_col):
    df = df.copy()
    for col in cols:
        df[col] = df[col] - df.groupby(group_col)[col].transform("mean")
    return df

es_dm = within_demean(es_df, ["Y"] + dummy_cols, "unit")
# Also absorb period FE by demeaning within period
es_dm = within_demean(es_dm, ["Y"] + dummy_cols, "period")

formula = "Y ~ " + " + ".join(dummy_cols) + " - 1"
es_model = smf.ols(formula, data=es_dm).fit(cov_type="HC3")

# Extract coefficients and confidence intervals
coefs = es_model.params[dummy_cols]
ci = es_model.conf_int().loc[dummy_cols]

# Map back to relative periods
rel_period_labels = rel_periods_no_ref  # excludes -1
# Insert the reference period (k=-1, effect=0) back for plotting
all_periods = sorted(rel_periods)
plot_periods = all_periods
plot_coefs = []
plot_lo = []
plot_hi = []
for k in plot_periods:
    if k == -1:
        plot_coefs.append(0)
        plot_lo.append(0)
        plot_hi.append(0)
    else:
        col = f"d_k{'m' if k < 0 else ''}{abs(k)}"
        plot_coefs.append(coefs[col])
        plot_lo.append(ci.loc[col, 0])
        plot_hi.append(ci.loc[col, 1])

fig, ax = plt.subplots(figsize=(11, 5))
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(-0.5, color="gray", linewidth=1.2, linestyle=":", label="Treatment date")

# Pre-treatment in blue, post in coral
colors = ["steelblue" if k < 0 else "coral" for k in plot_periods]
for i, (k, c, lo, hi) in enumerate(zip(plot_periods, plot_coefs, plot_lo, plot_hi)):
    ax.errorbar(k, c, yerr=[[c - lo], [hi - c]], fmt="o", color=colors[i],
                capsize=4, markersize=6, linewidth=1.5)

ax.set_xlabel("Periods relative to treatment (k)")
ax.set_ylabel("Estimated effect $\\delta_k$")
ax.set_title("Event Study: Treatment Effect by Relative Period\n"
             "(Pre-period coefficients ≈ 0 supports parallel trends)")
ax.legend()
plt.tight_layout()
plt.show()

print("Event study coefficients:")
for k, c, lo, hi in zip(plot_periods, plot_coefs, plot_lo, plot_hi):
    print(f"  k={k:+d}: {c:.3f}  [95% CI: {lo:.3f}, {hi:.3f}]")
```

---

## Two-Way Fixed Effects (TWFE)

When you have multiple units and multiple time periods, the standard implementation of DiD is **Two-Way Fixed Effects (TWFE)**:

$$Y_{it} = \alpha_i + \lambda_t + \delta \cdot D_{it} + \varepsilon_{it}$$

where:
- $\alpha_i$ = unit fixed effects — one intercept per unit, absorbing all time-invariant unit characteristics (geography, demographics, management quality, etc.)
- $\lambda_t$ = time fixed effects — one intercept per period, absorbing any shocks that affect all units equally (macroeconomic conditions, platform-wide trends, seasonality)
- $D_{it} = 1$ if unit $i$ is treated at time $t$ (0 otherwise)
- $\delta$ = the average treatment effect (the DiD coefficient)

**Why unit fixed effects?** They eliminate the baseline level difference between treated and control groups. You don't need treatment and control to look alike — you only need them to trend alike.

**Why time fixed effects?** They eliminate any common trend. If outcomes are rising for everyone, the time FE absorb that, so $\delta$ reflects only the differential change in treated units.

**Practical implementation** — use `statsmodels` with the `entity_effects` / `time_effects` capabilities, or the `linearmodels` library which has first-class panel support:

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

np.random.seed(7)

# ── Simulate balanced panel ───────────────────────────────────────────────────
n_units = 50
n_periods = 8
treatment_start = 4    # treated units treated from period 4 onward
treated_units = np.arange(25)   # first 25 units are treated

records = []
for unit in range(n_units):
    is_treated = int(unit in treated_units)
    unit_fe = np.random.normal(0, 3)
    for t in range(n_periods):
        time_fe = np.random.normal(0, 1)   # common time shock (absorbed by lambda_t)
        D = int(is_treated and t >= treatment_start)
        true_effect = 4.0
        y = 20 + unit_fe + 0.6 * t + true_effect * D + np.random.normal(0, 1.5)
        records.append({"unit": unit, "period": t, "treated": is_treated, "D": D, "Y": y})

panel = pd.DataFrame(records)

# ── Method 1: TWFE via dummy variables (small datasets) ──────────────────────
model_twfe = smf.ols("Y ~ D + C(unit) + C(period)", data=panel).fit(cov_type="HC3")
print(f"TWFE via dummies — D coefficient: {model_twfe.params['D']:.3f} "
      f"(SE={model_twfe.bse['D']:.3f})")

# ── Method 2: linearmodels PanelOLS (recommended for larger panels) ───────────
panel_indexed = panel.set_index(["unit", "period"])
mod = PanelOLS(panel_indexed["Y"], panel_indexed[["D"]],
               entity_effects=True, time_effects=True)
res = mod.fit(cov_type="clustered", cluster_entity=True)
print(f"\nTWFE via PanelOLS:")
print(res.summary.tables[1])
print(f"\nTrue effect: {4.0}")
```

### Clustered Standard Errors

In panel settings, observations within the same unit are correlated over time. Standard errors that ignore this are too small, leading to overconfidence. Always cluster standard errors at the unit level (or at the level of treatment assignment):

```python
# Using linearmodels: cov_type="clustered", cluster_entity=True
# Using statsmodels OLS with manual clustering:
from statsmodels.stats.sandwich_covariance import cov_cluster

# After fitting model_twfe:
# robust_se = np.sqrt(np.diag(cov_cluster(model_twfe, panel["unit"])))
```

---

## Staggered Adoption

In many real-world settings, different units adopt treatment at different times. This is called **staggered adoption** or **staggered DiD**.

**Example**: Different US states expanded Medicaid at different times under the ACA (2014–2017). Different markets get a new app feature at different weeks. Different stores receive a merchandising update on a rolling schedule.

### Why TWFE Fails with Staggered Adoption

When treatment effects are **heterogeneous** (different units have different effect sizes, or effects change over time), TWFE produces biased estimates. The Goodman-Bacon (2021) decomposition shows that TWFE implicitly uses **already-treated units as controls** for units treated later:

- Units treated early act as "controls" for units treated late
- But already-treated units' outcomes include the treatment effect — they're contaminated controls
- The resulting TWFE estimate is a weighted average of all possible 2×2 DiD comparisons, with potentially negative weights for some comparisons

**The problem in plain English**: TWFE is comparing late-adopters to early-adopters in the post period. If the treatment effect grows over time (dynamic effects), the early-adopters have "gained" more, making them look like they're declining relative to late-adopters — a spurious negative contribution to the estimate.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Simulate staggered adoption with heterogeneous effects ─────────────────────
# Three cohorts: treated at period 3, 5, 7. One never-treated group.
n_per_cohort = 25
n_periods = 10

cohort_treatment = {
    "cohort_3": 3,    # treated at period 3
    "cohort_5": 5,    # treated at period 5
    "cohort_7": 7,    # treated at period 7
    "never":    None  # never treated
}

records = []
unit_id = 0
for cohort, treat_period in cohort_treatment.items():
    for u in range(n_per_cohort):
        unit_fe = np.random.normal(0, 2)
        for t in range(n_periods):
            if treat_period is None:
                D = 0
                # Heterogeneous ATT: earlier cohorts get bigger effects
                effect = 0
            else:
                D = int(t >= treat_period)
                # Dynamic effect: grows over time; earlier cohorts have larger effects
                # Cohort 3: effect = 3 * (t - treat_period + 1) when treated
                # Cohort 5: effect = 2 * (t - treat_period + 1) when treated
                # Cohort 7: effect = 1 * (t - treat_period + 1) when treated
                multiplier = {"cohort_3": 3.0, "cohort_5": 2.0, "cohort_7": 1.0}[cohort]
                effect = D * multiplier * (t - treat_period + 1)
            y = 15 + unit_fe + 0.5 * t + effect + np.random.normal(0, 1)
            records.append({
                "unit": unit_id, "period": t, "cohort": cohort,
                "treat_period": treat_period if treat_period else 99,
                "D": D, "Y": y
            })
        unit_id += 1

stagger = pd.DataFrame(records)

# ── TWFE estimate (biased) ────────────────────────────────────────────────────
twfe_model = smf.ols("Y ~ D + C(unit) + C(period)", data=stagger).fit()
print(f"TWFE estimate: {twfe_model.params['D']:.3f}")

# True ATT: average of all treatment effects across cohorts and post-periods
true_att_records = stagger[stagger["D"] == 1].copy()
multipliers = {"cohort_3": 3.0, "cohort_5": 2.0, "cohort_7": 1.0, "never": 0.0}
true_att_records["true_effect"] = true_att_records.apply(
    lambda r: multipliers[r["cohort"]] * (r["period"] - r["treat_period"] + 1)
    if r["cohort"] != "never" else 0, axis=1
)
true_att = true_att_records["true_effect"].mean()
print(f"True ATT:      {true_att:.3f}")
print(f"\nBias in TWFE:  {twfe_model.params['D'] - true_att:.3f}")
print("(TWFE is biased because treated units serve as controls for later-treated units,")
print(" and the treatment effects are dynamic/heterogeneous.)")

# ── Goodman-Bacon decomposition (illustration) ────────────────────────────────
print("\nGoodman-Bacon insight: TWFE is a weighted average of all 2x2 DiD comparisons.")
print("Some comparisons use early-treated units as 'controls' -> contaminated.")
```

### Modern Solutions for Staggered DiD

When facing staggered adoption with potentially heterogeneous effects, use one of these estimators instead of TWFE:

| Method | Key Idea | Python Package |
|---|---|---|
| **Callaway-Sant'Anna (2021)** | Estimate ATT for each cohort×time cell separately, then aggregate | `csdid` (R), `did` package concepts |
| **Sun-Abraham (2021)** | Interaction-weighted estimator, clean cohort comparisons | Manual implementation |
| **Stacked DiD (Cengiz 2019)** | Stack separate 2×2 datasets per cohort, only using clean controls | Manual implementation |
| **TWFE + event study** | Check for pre-trends and dynamic effects; bias is worse with heterogeneity | `linearmodels` |

The core fix in all these methods: **compare each treatment cohort only to never-treated (or not-yet-treated) units**, never to already-treated units.

---

## DiD Validity Checks

Beyond the pre-trends test, several additional checks strengthen the credibility of a DiD analysis.

### 1. Placebo Outcome Test

Apply the DiD design to an outcome that should theoretically be unaffected by the treatment. If you find a large, significant effect on the placebo outcome, something is wrong — the groups are not comparable, or there is a confounding event.

**Example**: If studying the effect of a minimum wage increase on employment, also check whether it affects, say, the number of churches (which shouldn't be affected). If you find an effect on churches, parallel trends likely fails.

```python
# Placebo outcome test example
np.random.seed(42)
n_units, n_periods = 60, 8
treat_period = 4

records = []
for unit in range(n_units):
    is_treated = int(unit < 30)
    unit_fe = np.random.normal(0, 2)
    for t in range(n_periods):
        D = int(is_treated and t >= treat_period)
        # True outcome: treatment affects Y1 but NOT Y2
        y1 = 10 + unit_fe + 0.4 * t + 3.0 * D + np.random.normal(0, 1)
        y2 = 20 + unit_fe + 0.2 * t + 0.0 * D + np.random.normal(0, 1)  # placebo
        records.append({"unit": unit, "period": t, "treated": is_treated, "D": D,
                         "Y_main": y1, "Y_placebo": y2})

placebo_df = pd.DataFrame(records)

for outcome in ["Y_main", "Y_placebo"]:
    m = smf.ols(f"{outcome} ~ D + C(unit) + C(period)", data=placebo_df).fit()
    print(f"{outcome:12s}: coef={m.params['D']:.3f}, p={m.pvalues['D']:.3f}"
          + ("  <- significant treatment effect (expected)" if outcome == "Y_main"
             else "  <- should be ~0 for placebo"))
```

### 2. Falsification Test (Time Placebo)

Run the DiD as if the treatment had happened at an earlier date — before the actual treatment. If you find a significant effect in this fake treatment period, it suggests there are differential trends that DiD is picking up, not a genuine treatment effect.

```python
# Time placebo: pretend treatment happened at period 2 instead of period 4
# Use only pre-treatment data (periods 0..3)
pre_only = placebo_df[placebo_df["period"] < treat_period].copy()
# Define fake treatment: treated units "treated" starting at period 2
fake_treat_period = 2
pre_only["D_fake"] = ((pre_only["treated"] == 1) &
                       (pre_only["period"] >= fake_treat_period)).astype(int)

m_fake = smf.ols("Y_main ~ D_fake + C(unit) + C(period)", data=pre_only).fit()
print(f"\nFalsification test (fake treatment at period {fake_treat_period}):")
print(f"  D_fake coef = {m_fake.params['D_fake']:.3f}, p = {m_fake.pvalues['D_fake']:.3f}")
print("  Should be ~0 and insignificant if DiD design is valid")
```

### 3. Spillover / SUTVA Check

The Stable Unit Treatment Value Assumption (SUTVA) requires that one unit's treatment doesn't affect another unit's outcome. Spillovers contaminate the control group, making the DiD estimate a lower bound on the true effect.

**Example**: If you roll out a new recommendation algorithm in some markets, users in control markets might be told about the new feature by treated users (word of mouth), contaminating the control group.

**Checks**:
- Compare units that are geographically/socially distant from treated units vs. adjacent
- Look for sign of the DiD effect among "buffer zone" units that shouldn't be affected at all
- If possible, randomize treatment at a higher level (city-level instead of user-level)

### Summary of Validity Checks

| Check | What it tests | Action if fails |
|---|---|---|
| Pre-trends test | Parallel trends in pre-period | DiD is invalid; find better controls or use synthetic control |
| Placebo outcome | No effect on unrelated outcomes | Check for confounding events; DiD likely invalid |
| Time placebo | No effect in pre-treatment window | Differential pre-trends; design is suspect |
| Spillover check | Control group uncontaminated | Effect estimate is biased downward; expand buffer zones |

---

## Complete DiD Pipeline

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

np.random.seed(123)

# ════════════════════════════════════════════════════════════════════════════════
# SCENARIO: A company redesigns its mobile app UI and launches it in some markets
# first (treated markets). We want to measure the effect on daily active users (DAU).
# ════════════════════════════════════════════════════════════════════════════════

n_markets = 40
n_periods = 12    # weeks 0..11; UI launches in treated markets at week 6
true_effect = 5.0  # +5 DAU per treated market-week post-launch

records = []
for mkt in range(n_markets):
    is_treated = int(mkt < 20)
    market_fe = np.random.normal(is_treated * 10, 5)  # treated markets tend larger
    for t in range(n_periods):
        post = int(t >= 6)
        D = int(is_treated and post)
        # Common time trend: DAU grows by 0.8 / week
        y = 50 + market_fe + 0.8 * t + true_effect * D + np.random.normal(0, 2)
        records.append({"market": mkt, "week": t, "treated": is_treated,
                         "post": post, "D": D, "DAU": y})

df_app = pd.DataFrame(records)

# ── Step 1: Visualize trends ──────────────────────────────────────────────────
avg = df_app.groupby(["week", "treated"])["DAU"].mean().unstack()
avg.columns = ["Control markets", "Treated markets"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(avg.index, avg["Control markets"], "o-", color="steelblue", label="Control")
ax.plot(avg.index, avg["Treated markets"], "s-", color="coral", label="Treated")
ax.axvline(5.5, color="gray", linestyle="--", linewidth=1.5)
ax.set_title("DAU Trends: UI Redesign Launch\n(dashed = launch date)")
ax.set_xlabel("Week")
ax.set_ylabel("Average DAU")
ax.legend()

# ── Step 2: TWFE estimate ─────────────────────────────────────────────────────
model = smf.ols("DAU ~ D + C(market) + C(week)", data=df_app).fit(cov_type="HC3")
coef = model.params["D"]
se = model.bse["D"]
ci_lo, ci_hi = model.conf_int().loc["D"]

print(f"TWFE DiD Estimate: {coef:.3f} (SE={se:.3f})")
print(f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"True effect: {true_effect:.3f}")
print(f"p-value: {model.pvalues['D']:.4f}")

# ── Step 3: Pre-trends test ───────────────────────────────────────────────────
pre = df_app[df_app["week"] < 6].copy()
for t in range(1, 6):
    pre[f"t{t}x_treated"] = (pre["treated"] == 1) & (pre["week"] == t)
    pre[f"t{t}x_treated"] = pre[f"t{t}x_treated"].astype(int)

formula_pre = "DAU ~ treated + C(week) + " + " + ".join([f"t{t}x_treated" for t in range(1, 6)])
pre_model = smf.ols(formula_pre, data=pre).fit()
print("\nPre-trends test:")
for t in range(1, 6):
    col = f"t{t}x_treated"
    p = pre_model.pvalues[col]
    flag = " <- VIOLATION" if p < 0.05 else ""
    print(f"  Week {t} x treated: coef={pre_model.params[col]:.3f}, p={p:.3f}{flag}")

# ── Step 4: Event study ───────────────────────────────────────────────────────
df_app["rel_week"] = np.where(df_app["treated"] == 1, df_app["week"] - 6, np.nan)

treated_app = df_app[df_app["treated"] == 1].copy()
rel_weeks = sorted(treated_app["rel_week"].unique().astype(int))
rel_weeks_no_ref = [k for k in rel_weeks if k != -1]

for k in rel_weeks_no_ref:
    col = f"d_k{'m' if k < 0 else ''}{abs(k)}"
    df_app[col] = ((df_app["treated"] == 1) & (df_app["rel_week"] == k)).astype(int)

dummy_cols = [f"d_k{'m' if k < 0 else ''}{abs(k)}" for k in rel_weeks_no_ref]
formula_es = "DAU ~ " + " + ".join(dummy_cols) + " + C(market) + C(week)"
es_model = smf.ols(formula_es, data=df_app).fit(cov_type="HC3")

all_rel_weeks = sorted(rel_weeks)
coefs_es = []
ci_lo_es = []
ci_hi_es = []
for k in all_rel_weeks:
    if k == -1:
        coefs_es.append(0); ci_lo_es.append(0); ci_hi_es.append(0)
    else:
        col = f"d_k{'m' if k < 0 else ''}{abs(k)}"
        coefs_es.append(es_model.params[col])
        ci_lo_es.append(es_model.conf_int().loc[col, 0])
        ci_hi_es.append(es_model.conf_int().loc[col, 1])

ax2 = axes[1]
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.axvline(-0.5, color="gray", linewidth=1.2, linestyle=":", label="Launch week")
colors = ["steelblue" if k < 0 else "coral" for k in all_rel_weeks]
for k, c, lo, hi in zip(all_rel_weeks, coefs_es, ci_lo_es, ci_hi_es):
    ax2.errorbar(k, c, yerr=[[c - lo], [hi - c]], fmt="o", color=colors[all_rel_weeks.index(k)],
                 capsize=4, markersize=6)
ax2.set_xlabel("Week relative to UI launch")
ax2.set_ylabel("Estimated effect on DAU")
ax2.set_title("Event Study: Effect of UI Redesign on DAU")
ax2.legend()

plt.tight_layout()
plt.show()
```

---

## Methods in Practice

Step-by-step checklist for running a DiD analysis:

1. **Collect panel data**: unit × time table with outcome Y, unit ID, and time period.
2. **Define the indicators**: `treated` (1 = treatment group, 0 = control; time-invariant), `post` (1 = post-treatment period, 0 = pre; unit-invariant), `D = treated × post` (actual treatment indicator — the only column that = 1 for treated units in post period).
3. **Visualize trends**: plot average Y over time, separate lines for treated and control. Lines should be roughly parallel before the treatment date.
4. **Pre-trends test**: in pre-treatment periods only, regress Y on unit FE, time FE, and (period × treated) interaction dummies. All interaction coefficients should be near zero and insignificant. A significant pre-trend = DiD is invalid.
5. **Fit TWFE model**: `Y ~ D + C(unit) + C(period)` on the full panel. The coefficient on D is your DiD estimate.
6. **Cluster standard errors** at the unit level (or at the level of treatment assignment). Observations within the same unit are correlated over time — ignoring this makes SEs too small.
7. **Event study** (recommended): estimate a separate coefficient for each period relative to treatment. Pre-period coefficients ≈ 0 supports parallel trends; post-period coefficients reveal effect dynamics (immediate vs. building vs. fading).
8. **Validity checks**: placebo outcome test (D should have no effect on an unrelated outcome), time placebo (fake treatment date should show no effect in pre-period data), spillover check (control group should not be contaminated by treatment).

**What good output looks like:**
- `D coef = 3.9, SE = 0.8, 95% CI [2.4, 5.4], p < 0.001`
- Pre-trends: all period × treated interactions insignificant → parallel trends holds
- Event study: flat pre-period, clean jump at treatment, stable post-period → credible causal estimate

---

## Interview Questions

### Technical Questions

**Q1: What is the parallel trends assumption and why is it not fully testable?**

Parallel trends requires that, absent treatment, the treated and control groups would have followed the same time trend. It is not fully testable because the post-treatment counterfactual for the treated group is never observed — that's the fundamental problem DiD is trying to solve. We can only observe what the treated group *actually* did post-treatment, not what it would have done without treatment. The pre-trends test checks whether trends were parallel *before* treatment, which supports (but does not prove) that they would have been parallel afterward. A violation of pre-trends is sufficient evidence to invalidate DiD, but parallel pre-trends is necessary, not sufficient.

---

**Q2: Write out the DiD estimator in 2×2 form and explain each term.**

$$\hat{\tau}_{DiD} = \underbrace{(\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}})}_{\text{treated group change}} - \underbrace{(\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}})}_{\text{control group change (counterfactual trend)}}$$

The first difference is the raw before-after change in the treated group — it contains both the treatment effect and any background trend. The second difference is the before-after change in the control group — it contains only the background trend (assuming parallel trends). Subtracting the second from the first cancels the trend and leaves the treatment effect.

---

**Q3: Why does TWFE produce biased estimates in staggered adoption settings?**

In staggered adoption with heterogeneous effects, TWFE implicitly uses already-treated units as controls for later-treated units. This is problematic because already-treated units' post-treatment outcomes include the treatment effect — they are "contaminated" controls. The Goodman-Bacon (2021) decomposition shows that the TWFE coefficient is a weighted average of all possible 2×2 DiD comparisons (early vs. late adopters, early vs. never-treated, late vs. never-treated), and some of these weights can be negative when treatment effects are dynamic. The solution is to use estimators that only compare treated cohorts to clean (never-treated or not-yet-treated) controls, such as Callaway-Sant'Anna (2021).

---

**Q4: What is an event study design? What does it tell you beyond a single DiD coefficient?**

An event study estimates a separate treatment effect coefficient $\delta_k$ for each period $k$ relative to the treatment date, rather than a single pooled estimate. It serves two purposes: (1) in pre-treatment periods ($k < 0$), the $\delta_k$ should be near zero — this tests the parallel trends assumption; (2) in post-treatment periods ($k \geq 0$), the $\delta_k$ trace out the dynamics of the treatment effect — revealing whether effects are immediate or build up gradually, fade out, or are constant. A single DiD coefficient hides this heterogeneity.

---

**Q5: How do you interpret the coefficients in the DiD regression $Y_{it} = \alpha + \beta \, \text{Treated}_i + \gamma \, \text{Post}_t + \delta \, (\text{Treated}_i \times \text{Post}_t) + \varepsilon_{it}$?**

- $\alpha$: the baseline average for the control group in the pre-period
- $\beta$: the pre-existing level difference between treated and control groups (selection bias, absorbed and removed)
- $\gamma$: the common time trend — how much the control group changed from pre to post
- $\delta$: the DiD estimate — the additional change in the treated group beyond the common trend. This is the treatment effect under parallel trends.

---

### Case Study Questions

**Case 1: Your company launches a new onboarding flow in 10 US cities in January, leaving another 10 cities as controls. You want to measure the impact on 7-day retention. Walk through the full DiD analysis.**

Steps: (1) Collect pre-treatment data (at least 4–8 weeks before launch). (2) Visualize average retention over time by group — check that treated and control cities are moving in parallel before launch. (3) Run the pre-trends test: regress retention on city FE, week FE, and city-group × week interactions for pre-launch weeks — check for significance. (4) Fit the TWFE model with city and week fixed effects; the coefficient on the treatment indicator is your DiD estimate. (5) Run an event study to check if effects emerge exactly at launch (not before) and to see if effects persist. (6) Cluster standard errors at the city level, since that's the unit of treatment assignment. (7) Run a placebo outcome test on an outcome unrelated to onboarding (e.g., average session length for users who completed onboarding long before launch).

---

**Case 2: The pre-trends test fails — treated and control cities were diverging before the launch. What do you do?**

A failed pre-trends test means the DiD design is invalid as-is. Do not proceed with the DiD estimate and reframe it as "adjusted" — the bias is unknown and potentially large. Options: (1) Find a better control group — look for cities that more closely tracked the treated cities in the pre-period (or use synthetic control). (2) Check whether the divergence is explained by an observable covariate (e.g., treated cities were selected because they were growing faster) — if so, you may be able to use a conditional parallel trends assumption and control for that covariate. (3) Use a different identification strategy (e.g., instrumental variables, regression discontinuity if there's a threshold in treatment selection). (4) If the pre-trends violation is small, acknowledge it as a limitation and bound the estimate.

---

**Case 3: You are a PM reviewing an analyst's DiD study of the effect of Medicaid expansion on emergency room visits across states. The analyst reports a large, significant negative effect (expansion reduces ER visits). What questions do you ask?**

Key questions: (1) Which states are in the control group, and were they on parallel trends before expansion? Show me the pre-trends test and the visual. (2) Were there any other policy changes in the treated states around the same time (e.g., other ACA provisions)? This would violate the "only treatment changed" assumption. (3) When did treated states expand? If staggered, was TWFE used? If so, the estimate may be biased — did you use Callaway-Sant'Anna instead? (4) Are standard errors clustered at the state level? (5) Did you run a placebo test on an outcome that Medicaid expansion shouldn't affect (e.g., auto accident rates)? (6) How many states are in each group? With few states, power is low and cluster-robust SEs can be unreliable — consider wild cluster bootstrap.

---

**Case 4: You want to measure the effect of a price increase that was rolled out to all users simultaneously. Can you use DiD? What alternatives exist?**

With simultaneous treatment of all users, there is no control group — DiD does not apply. You need a different design: (1) **Pre/post OLS**: Simple before-after comparison, but this conflates the treatment effect with any time trend. Valid only if you believe the world was otherwise stable. (2) **Interrupted Time Series (ITS)**: Model the outcome as a function of time and add a structural break at the treatment date. Allows for a pre-existing trend, but still can't control for confounding events. (3) **Holdout group**: If you can withhold the price change from a random subset of users (even temporarily), you get a proper comparison group and can run DiD or an experiment. (4) **Synthetic control**: Build a synthetic "control" from historical data by matching to a combination of pre-treatment outcome paths, then compare post-treatment. Works well with a single treated unit and a long pre-treatment history.
