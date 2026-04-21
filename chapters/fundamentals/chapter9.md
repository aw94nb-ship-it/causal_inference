# Chapter 9: Instrumental Variables

Every method we have covered so far — matching, IPW, DML — requires the same core assumption: **no unmeasured confounders**. All variables that affect both treatment and outcome must be in your data. In practice, this assumption often fails. People self-select into treatments for reasons we cannot observe: motivation, ability, risk tolerance, social connections. **Instrumental Variables (IV)** is the solution for exactly this situation. The idea: find a variable $Z$ that shifts treatment assignment but has no direct effect on the outcome. By exploiting this external "push" on treatment, IV can recover causal effects even in the presence of unmeasured confounders — at the cost of estimating only a specific, local causal effect rather than the average treatment effect for everyone.

---

## The Problem: Unmeasured Confounding

Suppose you want to estimate the effect of education on earnings. People with more education earn more — but they also come from wealthier families, have higher ability, and are more motivated. These factors affect both the decision to get more education and future earnings. Even if you control for all observable characteristics, there will be unobservable factors (native ability, family connections, grit) that confound the relationship.

The DAG looks like this:

```
        U (unmeasured: ability, motivation)
       ↙ ↘
      T → Y
   (Education) (Earnings)
```

The unobserved variable $U$ creates a back-door path $T \leftarrow U \rightarrow Y$ that cannot be blocked — it is unmeasured, so you cannot condition on it. OLS will over-estimate the effect of education because it picks up the "ability" channel.

**No amount of observed-variable adjustment fixes this.** Once there is an unmeasured confounder, methods like matching, IPW, and DML all fail. You need a different identification strategy.

---

## What Is an Instrument?

A variable $Z$ is a **valid instrument** if it satisfies three conditions:

### Condition 1: Relevance

$$\text{Cov}(Z, T) \neq 0$$

The instrument must actually affect the treatment. In the education example: $Z$ must shift how much education people get. Without this, the instrument is **weak** — it provides no information about the causal effect of $T$ on $Y$.

### Condition 2: Exclusion Restriction

$$Z \perp\!\!\!\perp Y \mid T, U$$

The instrument affects $Y$ **only through** $T$. There is no direct arrow from $Z$ to $Y$, and no back-door path from $Z$ to $Y$ that bypasses $T$. This is the most important and most debated assumption in IV — it is not testable from data alone and must be justified by subject matter knowledge.

### Condition 3: Independence (Exogeneity / As-good-as-random)

$$Z \perp\!\!\!\perp U$$

The instrument is as good as randomly assigned — it is not correlated with the unmeasured confounder $U$. In other words, $Z$ is not itself confounded.

### The IV DAG

A valid instrument creates this DAG:

```
Z → T → Y
        ↑
U ─────┘
(Z has no arrow to Y, and no arrow to U)
```

$Z$ affects $T$, which affects $Y$. The unmeasured $U$ affects both $T$ and $Y$. But $Z$ is upstream of $T$ and has no other path to $Y$. Any correlation between $Z$ and $Y$ must therefore flow entirely through $T$ — and we can use this to isolate the causal effect of $T$ on $Y$.

---

## The Wald Estimator

For a **binary instrument** $Z \in \{0, 1\}$, the IV estimate is the **Wald estimator**:

$$\hat{\beta}_{IV} = \frac{E[Y \mid Z=1] - E[Y \mid Z=0]}{E[T \mid Z=1] - E[T \mid Z=0]} = \frac{\text{Reduced form (total effect of Z on Y)}}{\text{First stage (effect of Z on T)}}$$

**Intuition**: The instrument shifts $T$ by some amount (the first stage). It also shifts $Y$ (the reduced form). If the only way $Z$ affects $Y$ is through $T$, then the ratio of these shifts is the causal effect of $T$ on $Y$.

Think of it as: "the instrument gave $T$ a push of size $\Delta T$. $Y$ moved by $\Delta Y$. The effect of $T$ on $Y$ is $\Delta Y / \Delta T$."

### Numeric Example: Job Training Lottery

A job training program ran a lottery ($Z$): winning the lottery ($Z=1$) increased participation ($T$) by 60 percentage points. Winning the lottery also increased employment by 12 percentage points.

$$\hat{\beta}_{IV} = \frac{0.12}{0.60} = 0.20$$

The training program increased employment probability by 20 percentage points — for those induced to participate by the lottery.

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# -------------------------------------------------------
# Simulate: Unmeasured confounder + valid instrument
# Real-world analog: lottery assignment to job training
# -------------------------------------------------------
n = 2000

# Unmeasured confounder (motivation, ability)
U = np.random.randn(n)

# Instrument: random lottery assignment (truly random => independent of U)
Z = np.random.binomial(1, 0.5, n)

# True causal effect of training on employment
beta_true = 0.3

# Treatment (participation): depends on Z and U (self-selection)
# Z encourages participation; U also drives participation
T_latent = -0.5 + 1.2 * Z + 0.8 * U + np.random.randn(n) * 0.5
T = (T_latent > 0).astype(float)   # binary participation

# Outcome (employment): depends on T (causal) AND U (confounder)
Y = beta_true * T + 0.5 * U + np.random.randn(n) * 0.5

print("Data Summary:")
print(f"  Participation rate (Z=0): {T[Z==0].mean():.3f}")
print(f"  Participation rate (Z=1): {T[Z==1].mean():.3f}")
print(f"  First stage (Z effect on T): {T[Z==1].mean() - T[Z==0].mean():.3f}")
print(f"  Reduced form (Z effect on Y): {Y[Z==1].mean() - Y[Z==0].mean():.3f}")
print()

# -------------------------------------------------------
# Naive OLS: biased because U is unmeasured
# -------------------------------------------------------
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(T.reshape(-1, 1), Y)
beta_ols = ols.coef_[0]

# -------------------------------------------------------
# Wald Estimator: manual implementation
# -------------------------------------------------------
first_stage = T[Z == 1].mean() - T[Z == 0].mean()
reduced_form = Y[Z == 1].mean() - Y[Z == 0].mean()
beta_wald = reduced_form / first_stage

print(f"True effect:     {beta_true:.3f}")
print(f"OLS estimate:    {beta_ols:.3f}  ← biased upward (picks up U channel)")
print(f"Wald IV:         {beta_wald:.3f}  ← recovers true effect")
```

---

## Two-Stage Least Squares (2SLS)

The Wald estimator works for a binary instrument with no additional controls. **2SLS** is the general IV estimator that handles:
- Continuous instruments
- Multiple instruments
- Additional control variables $X$

### The Two Stages

**Stage 1**: Regress $T$ on $Z$ (and controls $X$):

$$T = \pi_0 + \pi_1 Z + \mathbf{X}\boldsymbol{\pi}_2 + \nu$$

Get fitted values $\hat{T} = \hat{\pi}_0 + \hat{\pi}_1 Z + \mathbf{X}\hat{\boldsymbol{\pi}}_2$.

**Stage 2**: Regress $Y$ on $\hat{T}$ (and controls $X$):

$$Y = \beta_0 + \beta_{2SLS} \hat{T} + \mathbf{X}\boldsymbol{\gamma} + \varepsilon$$

The coefficient $\beta_{2SLS}$ is the IV estimate.

**Why does this work?** $\hat{T}$ contains only the variation in $T$ that is driven by $Z$ — the part of $T$ that is "clean" (not confounded by $U$, since $Z \perp U$). By using $\hat{T}$ instead of $T$, the second stage regression uses only the exogenous variation in $T$.

The 2SLS formula (in matrix notation, simplified):

$$\hat{\beta}_{2SLS} = (\hat{T}' \hat{T})^{-1} \hat{T}' Y = (Z'X(X'X)^{-1}X'Z)^{-1} Z'X(X'X)^{-1}X'Y$$

**Important note**: The standard errors from running two separate OLS regressions are wrong — they do not account for the fact that $\hat{T}$ is estimated. Always use dedicated IV software (statsmodels, linearmodels) for correct standard errors.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# -------------------------------------------------------
# Simulate with controls and continuous instrument
# Real-world analog: distance to college as instrument
# for education → earnings
# -------------------------------------------------------
n = 1000

# Controls: family background (observed)
X_age = np.random.randint(25, 55, n).astype(float)
X_family_income = np.random.randn(n)   # observed family income

# Unmeasured confounder: ability
U_ability = np.random.randn(n)

# Instrument: distance to nearest college (miles, normalized)
# Relevant: living farther reduces years of education
# Exogenous: distance is determined by geography, not ability
# Exclusion: distance affects earnings only through education
Z_distance = np.random.exponential(1, n)  # distance (miles, normalized)

# True causal effect of education on log earnings
beta_true = 0.08  # 8% increase in earnings per extra year of education

# Education (years): decreases with distance, increases with ability
T_education = (12
               - 0.8 * Z_distance          # distance reduces education
               + 0.6 * U_ability            # ability increases education
               + 0.3 * X_family_income      # family income helps
               + np.random.randn(n))

# Log earnings: depends on education (causal), ability (confounder), controls
Y_log_earnings = (beta_true * T_education
                  + 0.3 * U_ability          # ability directly boosts earnings
                  - 0.01 * X_age
                  + 0.1 * X_family_income
                  + np.random.randn(n) * 0.5)

# Controls matrix
X_controls = np.column_stack([X_age, X_family_income])

# -------------------------------------------------------
# OLS: biased (ability affects both T and Y)
# -------------------------------------------------------
ols_design = np.column_stack([T_education, X_controls])
ols = LinearRegression().fit(ols_design, Y_log_earnings)
beta_ols = ols.coef_[0]

# -------------------------------------------------------
# 2SLS: manual implementation (for intuition only!)
# Standard errors will be wrong — don't use in production
# -------------------------------------------------------
# Stage 1: regress T on Z and X
stage1_design = np.column_stack([Z_distance, X_controls])
stage1_design_sm = sm.add_constant(stage1_design)
stage1 = sm.OLS(T_education, stage1_design_sm).fit()
T_hat = stage1.fittedvalues

# First-stage F-statistic (should be > 10)
# Test that Z's coefficient is significant
z_coef = stage1.params[1]
z_tstat = stage1.tvalues[1]
print(f"First stage: Z coefficient = {z_coef:.3f}, t = {z_tstat:.2f}")
print(f"First stage F-statistic (instrument): {z_tstat**2:.1f}  (rule of thumb: > 10)")

# Stage 2: regress Y on T_hat and X
stage2_design = np.column_stack([T_hat, X_controls])
stage2_design_sm = sm.add_constant(stage2_design)
stage2 = sm.OLS(Y_log_earnings, stage2_design_sm).fit()
beta_2sls_manual = stage2.params[1]

# -------------------------------------------------------
# 2SLS: using statsmodels IV2SLS (correct standard errors!)
# -------------------------------------------------------
from statsmodels.sandbox.regression.gmm import IV2SLS

# IV2SLS: endog = Y, exog = [const, T, controls], instrument = [const, Z, controls]
endog = Y_log_earnings
exog_with_const = sm.add_constant(np.column_stack([T_education, X_controls]))
instruments_with_const = sm.add_constant(np.column_stack([Z_distance, X_controls]))

iv_model = IV2SLS(endog, exog_with_const, instruments_with_const).fit()
beta_iv = iv_model.params[1]
se_iv = iv_model.bse[1]

print()
print("=" * 55)
print(f"True effect:          {beta_true:.4f}")
print(f"OLS estimate:         {beta_ols:.4f}  ← biased upward (ability)")
print(f"2SLS manual:          {beta_2sls_manual:.4f}  (SEs wrong — illustration only)")
print(f"IV2SLS (statsmodels): {beta_iv:.4f} ± {se_iv:.4f}")
print(f"IV2SLS 95% CI:        [{beta_iv - 1.96*se_iv:.4f}, {beta_iv + 1.96*se_iv:.4f}]")
print("=" * 55)
```

### Using linearmodels for 2SLS (Preferred)

```python
# pip install linearmodels
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS as LM_IV2SLS
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Reuse the data from above (copy for self-contained block)
n = 1000
X_age = np.random.randint(25, 55, n).astype(float)
X_family_income = np.random.randn(n)
U_ability = np.random.randn(n)
Z_distance = np.random.exponential(1, n)
beta_true = 0.08

T_education = (12 - 0.8 * Z_distance + 0.6 * U_ability
               + 0.3 * X_family_income + np.random.randn(n))
Y_log_earnings = (beta_true * T_education + 0.3 * U_ability
                  - 0.01 * X_age + 0.1 * X_family_income
                  + np.random.randn(n) * 0.5)

df = pd.DataFrame({
    'log_earnings': Y_log_earnings,
    'education': T_education,
    'distance': Z_distance,
    'age': X_age,
    'family_income': X_family_income,
    'const': 1.0
})

# linearmodels IV2SLS:
# dependent ~ 1 + exogenous_controls + [endogenous ~ instruments]
iv = LM_IV2SLS(
    dependent=df['log_earnings'],
    exog=df[['const', 'age', 'family_income']],
    endog=df[['education']],
    instruments=df[['distance']]
).fit(cov_type='robust')

print(iv.summary.tables[1])
print(f"\nTrue effect: {beta_true:.4f}")
print(f"IV estimate: {iv.params['education']:.4f}")
print(f"95% CI:      [{iv.conf_int().loc['education', 'lower']:.4f}, "
      f"{iv.conf_int().loc['education', 'upper']:.4f}]")
```

---

## LATE: What IV Actually Estimates

A critical and often misunderstood point: **IV does not estimate the ATE**. It estimates the **Local Average Treatment Effect (LATE)** — the average treatment effect for a specific subpopulation called **compliers**.

### Principal Strata

Given a binary instrument $Z$ and binary treatment $T$, we can classify each unit by their potential treatments $(T(Z=0), T(Z=1))$:

| Type | $T(Z=0)$ | $T(Z=1)$ | Description |
|---|---|---|---|
| **Compliers** | 0 | 1 | Take treatment when encouraged, don't when not |
| **Always-takers** | 1 | 1 | Take treatment regardless |
| **Never-takers** | 0 | 0 | Never take treatment |
| **Defiers** | 1 | 0 | Do the opposite of the instrument |

Under the **monotonicity** assumption (no defiers: $T(Z=1) \geq T(Z=0)$ for all units), the Wald/IV estimator identifies:

$$\hat{\beta}_{IV} = E[Y(1) - Y(0) \mid \text{complier}] = \text{LATE}$$

### Why IV Misses Non-Compliers

- **Always-takers**: always treated, regardless of $Z$. The instrument does not change their treatment, so they contribute nothing to the first stage (and nothing to IV identification).
- **Never-takers**: never treated, regardless of $Z$. Same — no contribution to identification.
- **Compliers**: the instrument flips their treatment. The first stage is nonzero because of them. The reduced form also works through them. IV is entirely identified by compliers.

### Implications

1. **LATE ≠ ATE**: If compliers are a non-representative subgroup, LATE can differ substantially from ATE.
2. **External validity**: Results from IV generalize to the complier population, not the full population.
3. **Different instruments, different LATEs**: If you use two different instruments for the same treatment, you may get different estimates — they are each identifying the LATE for *their* complier populations.
4. **Complier characteristics**: You can estimate average characteristics of compliers (even though you can't identify who they are) using instrumental variables methods.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
np.random.seed(42)

# -------------------------------------------------------
# Demonstrate LATE: IV estimates effect for compliers only
# -------------------------------------------------------
n = 5000

# Three types of people (true potential treatment status)
# 40% always-takers, 30% never-takers, 30% compliers
types = np.random.choice(['complier', 'always_taker', 'never_taker'],
                          size=n, p=[0.3, 0.4, 0.3])

# True treatment effects by type (heterogeneous effects)
true_effects = {'complier': 2.0, 'always_taker': 0.5, 'never_taker': 3.0}
true_ate = 0.3 * 2.0 + 0.4 * 0.5 + 0.3 * 3.0   # population ATE
true_late = 2.0                                    # complier ATE (LATE)

# Unmeasured confounder
U = np.where(types == 'always_taker', 1.0,
     np.where(types == 'never_taker', -0.5, 0.0)) + np.random.randn(n) * 0.3

# Binary instrument (random encouragement)
Z = np.random.binomial(1, 0.5, n)

# Treatment assignment by type and instrument
def assign_treatment(t_type, z):
    if t_type == 'always_taker':
        return 1
    elif t_type == 'never_taker':
        return 0
    else:  # complier
        return z

T = np.array([assign_treatment(types[i], Z[i]) for i in range(n)])

# True individual treatment effect
ind_effect = np.array([true_effects[t] for t in types])

# Outcome
Y0 = U + np.random.randn(n) * 0.5    # potential outcome without treatment
Y = Y0 + T * ind_effect

# -------------------------------------------------------
# OLS: biased (always-takers are high-U, skewing estimate)
# -------------------------------------------------------
ols = LinearRegression().fit(T.reshape(-1, 1), Y)
beta_ols = ols.coef_[0]

# -------------------------------------------------------
# Wald estimator
# -------------------------------------------------------
first_stage = T[Z == 1].mean() - T[Z == 0].mean()
reduced_form = Y[Z == 1].mean() - Y[Z == 0].mean()
beta_iv = reduced_form / first_stage

print("=" * 55)
print(f"True ATE  (all types):   {true_ate:.3f}")
print(f"True LATE (compliers):   {true_late:.3f}")
print(f"OLS estimate:            {beta_ols:.3f}  ← biased")
print(f"IV (Wald) estimate:      {beta_iv:.3f}  ← identifies LATE, not ATE")
print("=" * 55)
print(f"\nCompliers make up {(types == 'complier').mean():.0%} of the population.")
print(f"First stage (compliance rate): {first_stage:.3f}")
print(f"Note: IV estimates the effect for compliers ({true_late:.1f}),")
print(f"      not the population ATE ({true_ate:.2f}).")
```

---

## Weak Instruments

If the instrument only weakly affects the treatment, the first stage is small. A small denominator in the Wald estimator $\Delta Y / \Delta T$ amplifies any noise or slight violations of the exclusion restriction — the estimate becomes extremely sensitive to small errors.

### The Weak Instrument Problem

Small first-stage coefficient means:
- Small $\hat{T}$ variation → Stage 2 regression has very little signal
- Even tiny violations of the exclusion restriction get magnified: a tiny direct effect of $Z$ on $Y$ (which should be zero) becomes large when divided by a tiny first stage
- Confidence intervals blow up — very wide, often spanning implausible ranges
- IV estimate becomes severely biased toward OLS in finite samples

### The F-Statistic Rule of Thumb

The standard diagnostic is the **first-stage F-statistic**: test whether the instrument's coefficient in the first stage regression is statistically significant.

Rule of thumb (Staiger & Stock 1997): **F > 10** means the instrument is strong enough for reliable inference. Below 10, weak instrument bias becomes substantial.

More recent work (Lee et al. 2022) suggests higher thresholds (F > 104 for 5% size), but F > 10 remains the widely-used practical benchmark.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# -------------------------------------------------------
# Show how IV estimate degrades with weak instruments
# Vary the first-stage strength (pi_z)
# -------------------------------------------------------
n = 1000
beta_true = 0.5
U = np.random.randn(n)

# Range of first-stage coefficients (instrument strength)
pi_z_values = [0.02, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]

results = []
for pi_z in pi_z_values:
    estimates = []
    for _ in range(200):  # 200 simulations
        Z = np.random.randn(n)
        T = pi_z * Z + 0.8 * U + np.random.randn(n) * 0.3
        Y = beta_true * T + 0.5 * U + np.random.randn(n) * 0.5

        # First stage F-statistic
        Z_sm = sm.add_constant(Z)
        fs = sm.OLS(T, Z_sm).fit()
        f_stat = fs.fvalue

        # Wald IV estimate
        cov_ZY = np.cov(Z, Y)[0, 1]
        cov_ZT = np.cov(Z, T)[0, 1]
        if abs(cov_ZT) > 1e-10:
            beta_iv = cov_ZY / cov_ZT
        else:
            beta_iv = np.nan
        estimates.append(beta_iv)

    # Re-run once for F-stat summary
    Z = np.random.randn(n)
    T = pi_z * Z + 0.8 * U + np.random.randn(n) * 0.3
    Y = beta_true * T + 0.5 * U + np.random.randn(n) * 0.5
    Z_sm = sm.add_constant(Z)
    fs = sm.OLS(T, Z_sm).fit()

    results.append({
        'pi_z': pi_z,
        'mean_estimate': np.nanmean(estimates),
        'std_estimate': np.nanstd(estimates),
        'F_stat': fs.fvalue,
        'estimates': estimates
    })

# Summary table
print(f"{'pi_z':>6} | {'F-stat':>8} | {'Mean IV est.':>12} | {'Std Dev':>8} | {'Strength':>10}")
print("-" * 60)
for r in results:
    strength = "WEAK" if r['F_stat'] < 10 else "OK" if r['F_stat'] < 100 else "STRONG"
    print(f"{r['pi_z']:>6.2f} | {r['F_stat']:>8.1f} | {r['mean_estimate']:>12.3f} | "
          f"{r['std_estimate']:>8.3f} | {strength:>10}")
print(f"\nTrue beta = {beta_true}")

# Plot: distribution of IV estimates by instrument strength
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, r in enumerate(results):
    est = [e for e in r['estimates'] if abs(e) < 10]  # clip extreme values
    axes[i].hist(est, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    axes[i].axvline(beta_true, color='red', linewidth=2, label=f'True = {beta_true}')
    axes[i].axvline(np.nanmean(r['estimates']), color='orange', linewidth=2,
                    linestyle='--', label=f'Mean = {np.nanmean(r["estimates"]):.2f}')
    axes[i].set_title(f'F = {r["F_stat"]:.1f} (π_z = {r["pi_z"]})')
    axes[i].legend(fontsize=7)
    axes[i].set_xlabel('IV Estimate')
axes[-1].axis('off')
plt.suptitle('IV Estimate Distribution by Instrument Strength\n(Red = true value)',
             fontsize=13)
plt.tight_layout()
plt.savefig('iv_weak_instrument.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nConclusion: when F < 10, the IV estimate has huge variance and is unreliable.")
```

---

## Classic IV Examples

### Example 1: Distance to College (Card, 1995)

**Setting**: Estimate the return to education on earnings.

**Problem**: People who get more education differ in ability, motivation, and family background — unmeasured confounders.

**Instrument**: Distance from the individual's county of origin to the nearest 2-year or 4-year college.

- **Relevance**: Living farther from college reduces years of education attained (travel costs, information costs).
- **Exclusion restriction**: Distance to college affects earnings only through the education channel — not directly through wages or job opportunities. (Assumption: local labor markets are not meaningfully different based on distance to college *after controlling for region*.)
- **Exogeneity**: College locations were historically determined by land grants and state planning, not by the ability of local residents.

**Finding**: Card (1995) found IV estimates of returns to education of ~14%, higher than OLS (~7%), suggesting OLS under-estimates returns (ability bias goes the other way: ability ↑ wages directly, making it look like education is doing more, but also ability ↑ education — whether OLS is biased up or down depends on which force dominates).

**LATE interpretation**: This is the return to education for people who would have gotten more education had there been a college nearby — not the effect for everyone.

### Example 2: Quarter of Birth (Angrist & Krueger, 1991)

**Setting**: Estimate the return to compulsory schooling.

**Instrument**: Quarter of birth (season in which you were born).

**Mechanism**:
- US compulsory attendance laws require attendance until age 16.
- Children born in Q1 (January–March) reach the legal dropout age with less education than children born in Q4 (October–December) because school year cutoffs mean Q1 children enter school slightly older.
- This creates a small but measurable difference in years of schooling driven entirely by birth season.

- **Relevance**: Quarter of birth predicts years of schooling (first stage is meaningful over large samples).
- **Exclusion restriction**: Birth season affects earnings only through schooling — not directly. (Somewhat controversial: season of birth may affect health, cognitive development.)
- **Exogeneity**: Season of birth is essentially random (or driven by fertility patterns uncorrelated with ability).

**Why it's a weak-ish instrument**: The first stage is small (small differences in schooling by quarter), requiring very large samples ($n \approx 300{,}000$) to get precise estimates.

### Example 3: Lottery Assignment to Job Training (JTPA Study)

**Setting**: The Job Training Partnership Act (JTPA) evaluation wanted to estimate the effect of job training on employment and earnings.

**Instrument**: Random lottery assignment ($Z$) to be offered training.

- **Relevance**: Being offered training substantially increases participation.
- **Exclusion restriction**: Lottery assignment affects employment only through actual participation (winning the lottery itself doesn't get you a job).
- **Exogeneity**: The lottery was genuinely random — gold standard instrument.

**Complication**: Not all lottery winners participated, and some non-winners found other ways to get training. This is **non-compliance** — the lottery is an instrument for participation, not a direct assignment of participation. IV/2SLS handles this naturally.

### Example 4: Random Feature Promotion → Adoption → Retention

**Setting**: A product team wants to know if using Feature X improves 30-day retention. The feature was not A/B tested — heavier users adopt it more (confounding).

**Instrument**: A/B test of a push notification encouraging Feature X adoption ($Z$ = received nudge).

- **Relevance**: Users who received the nudge adopted Feature X at higher rates.
- **Exclusion restriction**: The nudge only affects retention through Feature X adoption — the notification itself doesn't improve retention through some other channel (e.g., re-engagement from the notification).
- **Exogeneity**: Randomized push notification → $Z \perp U$.

**LATE**: We estimate the retention effect for the complier population — users who adopted Feature X because they received the nudge.

```python
import numpy as np
import pandas as pd
from statsmodels.sandbox.regression.gmm import IV2SLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# -------------------------------------------------------
# Simulate: App feature nudge as instrument for adoption
# -------------------------------------------------------
n = 5000

# Unmeasured confounder: user engagement level
U_engagement = np.random.randn(n)

# Instrument: random push notification encouraging feature adoption
Z_nudge = np.random.binomial(1, 0.5, n)

# Observed controls
X_tenure = np.random.randint(1, 365, n).astype(float)  # days since signup
X_platform = np.random.binomial(1, 0.6, n).astype(float)  # iOS=1, Android=0

# True effect of feature adoption on retention
beta_true = 0.15  # 15 percentage points

# Feature adoption: depends on nudge + engagement (confounded)
adopt_latent = (-0.5
                + 1.5 * Z_nudge          # nudge encourages adoption
                + 0.8 * U_engagement     # engaged users adopt anyway
                + 0.002 * X_tenure       # longer-tenure users more likely
                + np.random.randn(n))
T_adoption = (adopt_latent > 0).astype(float)

# 30-day retention (binary outcome — use linear probability model for simplicity)
Y_retention = (beta_true * T_adoption
               + 0.2 * U_engagement        # engaged users retain anyway
               + 0.0003 * X_tenure
               + 0.05 * X_platform
               + np.random.randn(n) * 0.3)
# Clip to [0,1] range
Y_retention = np.clip(Y_retention, 0, 1)

controls = np.column_stack([X_tenure, X_platform])

# Naive OLS
ols_design = sm.add_constant(np.column_stack([T_adoption, controls]))
ols = sm.OLS(Y_retention, ols_design).fit()
beta_ols = ols.params[1]

# First stage check
fs_design = sm.add_constant(np.column_stack([Z_nudge, controls]))
first_stage = sm.OLS(T_adoption, fs_design).fit()
print(f"First stage F-statistic: {first_stage.fvalue:.1f}  (instrument strength)")
print(f"Nudge coefficient in first stage: {first_stage.params[1]:.3f} "
      f"(p = {first_stage.pvalues[1]:.4f})")

# 2SLS via statsmodels
endog = Y_retention
exog = sm.add_constant(np.column_stack([T_adoption, controls]))
instruments = sm.add_constant(np.column_stack([Z_nudge, controls]))
iv = IV2SLS(endog, exog, instruments).fit()

print(f"\n{'='*55}")
print(f"True effect (LATE):    {beta_true:.3f}")
print(f"OLS estimate:          {beta_ols:.3f}  ← biased upward")
print(f"IV/2SLS estimate:      {iv.params[1]:.3f}  ← LATE for compliers")
print(f"IV 95% CI:             [{iv.params[1] - 1.96*iv.bse[1]:.3f}, "
      f"{iv.params[1] + 1.96*iv.bse[1]:.3f}]")
print(f"\nCompliance rate (first stage): {T_adoption[Z_nudge==1].mean() - T_adoption[Z_nudge==0].mean():.3f}")
```

---

## Fuzzy Regression Discontinuity as IV (Preview)

In a **Sharp Regression Discontinuity Design (RDD)**, crossing a threshold assigns treatment deterministically: everyone above the cutoff is treated. In a **Fuzzy RDD**, crossing the threshold only changes the *probability* of treatment — not everyone above the cutoff is treated, and some below may be treated anyway.

Fuzzy RDD is formally an IV problem:
- **Instrument $Z$**: indicator for being above the cutoff ($Z = \mathbf{1}[X \geq c]$)
- **Treatment $T$**: actual treatment receipt
- **Endogenous variation**: $T$ does not perfectly follow $Z$

The 2SLS estimator applied to the fuzzy RDD:

$$\hat{\beta}_{Fuzzy RDD} = \frac{\lim_{x \to c^+} E[Y \mid X=x] - \lim_{x \to c^-} E[Y \mid X=x]}{\lim_{x \to c^+} E[T \mid X=x] - \lim_{x \to c^-} E[T \mid X=x]}$$

This is exactly the Wald estimator — reduced form divided by first stage — evaluated at the discontinuity. The LATE here is the effect for units near the cutoff who comply with the discontinuity. RDD is covered in depth in Chapter 10.

---

## IV Assumptions: Testability and Failure Modes

Not all IV assumptions are testable. Here's a summary:

| Assumption | Testable? | How to Test / Diagnose |
|---|---|---|
| **Relevance** | Yes | First-stage regression; F-statistic > 10 |
| **Exclusion restriction** | Not directly | Subject matter argument; over-identification tests (if >1 instrument) |
| **Exogeneity / Independence** | Partially | Balance tests: $Z$ vs. baseline covariates; placebo outcomes |
| **Monotonicity** | Not directly | Check for defiers via subject matter knowledge |

### Common Failure Modes

**Exclusion restriction violation**: The instrument has a direct path to $Y$ beyond $T$.
- Example: distance to college might also proxy for local economic development, which affects earnings directly.
- Fix: add controls for local economic conditions; argue that the direct effect is small.

**Weak instruments**: F < 10. The estimate becomes very noisy and biased toward OLS.
- Fix: find a stronger instrument; use LIML (Limited Information Maximum Likelihood) which is more robust to weak instruments than 2SLS.

**Defiers / monotonicity violation**: Some units do the opposite of the instrument.
- Generally rare in practice; address through subject matter reasoning.

---

## Comparison: IV vs. Other Methods

| | OLS | Matching / IPW | DML | IV |
|---|---|---|---|---|
| **Handles unmeasured confounders** | No | No | No | **Yes** |
| **Requires measured confounders** | Yes | Yes | Yes | Not for causal identification |
| **Requires an instrument** | No | No | No | **Yes** |
| **Estimates** | ATE (if unconfounded) | ATE / ATT | ATE / CATE | LATE (compliers only) |
| **External validity** | Full population | Full population | Full population | Complier subgroup |
| **Handles nonlinear confounding** | No | Partially | Yes | No (standard 2SLS) |
| **When to use** | No unmeasured confounding | Observed confounders, good overlap | Many observed confounders | Unmeasured confounding + valid instrument available |

---

## Summary

Instrumental variables is the workhorse method for causal inference when unmeasured confounders are unavoidable. The key ideas:

1. **Find a valid instrument**: must be relevant (first stage), exclusion-restricted (no direct effect on $Y$), and exogenous (as-good-as-random).
2. **The Wald estimator** divides the reduced form by the first stage — using only the exogenous variation in $T$ to identify the causal effect.
3. **2SLS generalizes** to continuous instruments and multiple controls; always use software with correct standard errors.
4. **IV estimates LATE**, not ATE — the effect for compliers, not the whole population.
5. **Weak instruments** (F < 10) produce unstable, biased estimates — always check the first-stage F-statistic.

The formula one more time:

$$\hat{\beta}_{2SLS} = \frac{\widehat{\text{Cov}}(Z, Y)}{\widehat{\text{Cov}}(Z, T)}$$

---

## Interview Questions

### Technical Q&A

**Q1: What are the three conditions for a valid instrument, and which is hardest to satisfy?**

The three conditions are: (1) **Relevance** — $Z$ must affect $T$ (testable via first-stage F-statistic); (2) **Exclusion restriction** — $Z$ affects $Y$ only through $T$ (not directly testable from data alone); (3) **Independence/Exogeneity** — $Z$ is as good as randomly assigned, not correlated with unmeasured confounders (partially testable via balance checks). The exclusion restriction is the hardest because it cannot be directly tested — it requires a subject-matter argument that there is truly no direct path from $Z$ to $Y$. A classic debate is whether distance to college satisfies exclusion: critics argue distance proxies for local economic development, which affects earnings directly.

**Q2: What does IV actually estimate, and how does it differ from ATE?**

IV estimates the **LATE** — the Local Average Treatment Effect — which is the average treatment effect for the **complier subpopulation**: units whose treatment status is changed by the instrument. Always-takers and never-takers contribute to the population but not to IV identification. LATE equals ATE only if there is no treatment effect heterogeneity, or if the instrument randomly selects units in a way that mirrors the full population. In most real applications, compliers are a specific subgroup (e.g., people near the margin of treatment decision), so LATE may differ from ATE substantially.

**Q3: What is the weak instrument problem and how do you detect and address it?**

A weak instrument has a small first-stage effect ($Z$ barely shifts $T$). This causes: (a) the Wald denominator to be small, amplifying any noise; (b) IV estimates to be very sensitive to tiny violations of the exclusion restriction; (c) severe finite-sample bias toward OLS. Detection: first-stage F-statistic < 10 (Staiger & Stock rule of thumb). Remedies: find a stronger instrument; use LIML instead of 2SLS (LIML is median-unbiased under weak instruments); use Anderson-Rubin confidence intervals which are valid even with weak instruments (based on reduced form, not the ratio).

**Q4: Why are the standard errors wrong when you run 2SLS as two separate OLS regressions?**

In manual 2SLS, Stage 2 regresses $Y$ on $\hat{T}$. But $\hat{T}$ is estimated — it has its own estimation error from Stage 1. Standard OLS treats $\hat{T}$ as if it were a fixed, known regressor and computes standard errors accordingly. This ignores the additional variance from Stage 1 estimation, making standard errors too small (over-confident). The correct approach uses the formula for 2SLS standard errors which accounts for the two-stage structure. Always use dedicated IV software (statsmodels IV2SLS, linearmodels IV2SLS) rather than manual two-step OLS.

**Q5: Can you have multiple instruments for one endogenous variable? What does that give you?**

Yes — multiple instruments give you **over-identification**, which allows for a test of instrument validity (the Sargan-Hansen J-test). Under exact identification (one instrument, one endogenous variable), there is no way to test the exclusion restriction. With over-identification, you have more equations than unknowns: if the instruments are both valid, they should give the same IV estimate. The J-test tests whether the estimates from different instruments are consistent with each other. Rejection suggests at least one instrument violates the exclusion restriction. Note: the J-test only detects inconsistency between instruments — it cannot tell you which one is invalid, and it has low power in small samples.

### Case Study Questions

**Case 1: Education and Earnings**

A researcher wants to use distance to the nearest college as an instrument to estimate the causal return to education. Walk through: (a) how you would test instrument relevance; (b) what threats to the exclusion restriction you would worry about; (c) how you would interpret the LATE in this context; and (d) whether the estimate applies to a policy of expanding college access.

*Key points*: (a) First-stage regression of education on distance, F-stat > 10. (b) Exclusion threats: local economic development (richer areas have more colleges and more high-paying jobs); urban/rural differences. Control for local labor market conditions. (c) LATE = return to education for people at the margin of going to college based on proximity — likely lower-SES students who are cost-sensitive. (d) The LATE is relevant for a policy of building more colleges precisely because it captures the effect for the marginal student who would be induced to attend.

**Case 2: Feature Rollout Without A/B Test**

Your company rolled out a recommendation feature to users, but uptake was non-random — power users adopted it more. A colleague suggests using a "recommendation nudge" (push notification sent to a random subset of users) as an instrument for feature adoption to measure the effect on retention. Evaluate this instrument. What would you check? What is the LATE in this context?

*Key points*: Relevance: does the nudge meaningfully increase adoption? Check adoption rates by nudge status; F-stat. Exclusion: does the nudge affect retention only through adoption, or could the notification itself re-engage users who then retain (even without adopting the feature)? This is a real threat — A/B test a "sham notification" as placebo. Exogeneity: the nudge was randomly sent — satisfied. LATE = retention effect for users who adopted the feature *because* of the nudge — these are users on the margin of adoption (not the most engaged users), so LATE may be lower than ATE.

**Case 3: Drug Prescription Study**

In health economics, physician prescribing preference (fraction of a doctor's other patients who received Drug A vs. Drug B) is used as an instrument for drug prescription to estimate drug effectiveness. A patient is assigned to the drug their doctor prefers to prescribe, even if the patient's characteristics would have led a different doctor to prescribe differently. Evaluate the three IV assumptions for this instrument.

*Key points*: Relevance: clearly satisfied — a doctor who prefers Drug A will prescribe it to most patients (strong first stage). Independence/Exogeneity: patients are not matched to doctors based on their potential drug response — satisfied if patient-doctor matching is quasi-random (same clinic, geography-based). Exclusion restriction (the hard one): does doctor preference affect outcomes only through drug choice? Potential violations: doctors who prefer Drug A may also provide better follow-up care, be more experienced, or work at better facilities — all of which directly affect health outcomes bypassing the drug channel. Partially addressable by controlling for physician and hospital characteristics. Also: physician preference may affect adherence behavior through the patient-doctor relationship itself.
