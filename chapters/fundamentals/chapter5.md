# Chapter 5: Propensity Score Methods

Chapter 4 showed that matching can construct a comparable control group — but it hits a wall when covariates are many or continuous: the curse of dimensionality makes it impossible to find units that match on all dimensions simultaneously. Propensity score methods solve this by compressing all covariate information into a single scalar — the probability of receiving treatment given observed covariates. Rosenbaum and Rubin (1983) proved that conditioning on this scalar is sufficient for confounding adjustment: if you've removed confounding by conditioning on $X$, you've also removed it by conditioning on $e(X)$. This chapter covers how to estimate propensity scores, how to use them for matching and reweighting, and the doubly robust estimator that combines both approaches for extra protection against model misspecification.

---

## The Dimensionality Problem

Suppose you want to evaluate a customer loyalty program. Customers who joined the program tend to differ from those who didn't on dozens of variables: purchase frequency, tenure, average order value, product category preferences, geographic region, engagement with emails, app usage, and more.

Matching directly on all these covariates fails:
- With 20 continuous covariates, the space has $20$ dimensions and distances between points lose meaning
- Most treated units will have no nearby control units — you'd have to drop most of the sample
- The curse of dimensionality means that even with a large dataset, the nearest neighbor might be quite far away

The insight of Rosenbaum and Rubin: instead of matching on all covariates simultaneously, collapse them into a single score — the **propensity score** — and match on that.

---

## What Is a Propensity Score?

The **propensity score** is the conditional probability of receiving treatment given observed covariates:

$$e(X) = P(T = 1 \mid X)$$

It's a number between 0 and 1 for each unit. A high propensity score means the unit was likely to be treated (based on their covariates); a low score means they were likely to be a control.

### The Rosenbaum-Rubin Theorem

The key result (Rosenbaum & Rubin, 1983): if unconfoundedness holds conditional on $X$:

$$(Y(0), Y(1)) \perp T \mid X$$

then it also holds conditional on the propensity score alone:

$$(Y(0), Y(1)) \perp T \mid e(X)$$

**Why?** The propensity score is a **balancing score** — within any stratum of $e(X)$, the distribution of $X$ is the same for treated and control units. So conditioning on $e(X)$ is equivalent to conditioning on all of $X$ for the purposes of removing confounding.

This is a remarkable dimensionality reduction: instead of adjusting for a $p$-dimensional covariate vector, you adjust for a single scalar.

### Intuition with a Business Example

In the loyalty program example:
- Customer A has propensity 0.8 (highly likely to enroll based on their profile)
- Customer B has propensity 0.81 (nearly identical enrollment probability)

Even though A and B might differ on individual covariates, they had essentially the same overall "propensity to be treated." Within this propensity stratum, treatment assignment is approximately random — so comparing their outcomes gives a valid estimate of the treatment effect.

---

## Estimating the Propensity Score

### Logistic Regression (Standard Approach)

The most common method: fit a logistic regression of treatment on all covariates.

$$\log\frac{e(X)}{1 - e(X)} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p$$

The fitted probabilities $\hat{e}(X_i) = \hat{P}(T_i = 1 \mid X_i)$ are the estimated propensity scores.

**Important distinction from predictive modeling**: the goal here is **not** to maximize predictive accuracy. The goal is to build a model that captures all the confounding relationships. A model that perfectly separates treated and control units will produce extreme propensity scores (near 0 or 1) for many units, blowing up the IPW weights. What you want is a model that explains selection without overfitting.

### Machine Learning Alternatives

Random forests, gradient boosting (XGBoost, LightGBM), and other ML methods can be used to estimate propensity scores, especially when relationships between covariates and treatment are non-linear. However:
- Regularize appropriately to avoid extreme scores
- Cross-validation helps, but be careful — you're not optimizing for AUC, you're optimizing for balance
- After estimation, always check covariate balance (SMD) — that's the true diagnostic

### What Makes a Good Propensity Score?

A propensity score model is "good" if, after matching or weighting on it, **covariate balance improves**. A model that achieves low classification error but leaves covariates unbalanced is failing at its actual purpose.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 3000

# Simulate loyalty program dataset
# Confounders: purchase_freq, tenure, avg_order_value, app_usage
# True ATE = 50 (program increases monthly purchases by $50)

purchase_freq   = np.random.poisson(5, n)
tenure          = np.random.exponential(2, n)           # years
avg_order_value = np.random.lognormal(4, 0.5, n)        # dollars (~55)
app_usage       = np.random.exponential(3, n)           # sessions/week

# Treatment: high engagement -> more likely to enroll
log_odds = (
    -2
    + 0.2 * purchase_freq
    + 0.3 * tenure
    + 0.01 * (avg_order_value - 55)
    + 0.15 * app_usage
)
p_enroll = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_enroll)

# Outcome: monthly purchases
purchases = (
    50
    + 8 * purchase_freq
    + 10 * tenure
    + 0.3 * avg_order_value
    + 5 * app_usage
    + 50 * treatment          # true ATE = ATT = 50
    + np.random.normal(0, 20, n)
)

df = pd.DataFrame({
    "purchase_freq":   purchase_freq,
    "tenure":          tenure,
    "avg_order_value": avg_order_value,
    "app_usage":       app_usage,
    "treatment":       treatment,
    "purchases":       purchases
})

covariates = ["purchase_freq", "tenure", "avg_order_value", "app_usage"]

print(f"Treated: {treatment.sum()}, Control: {(1-treatment).sum()}")
naive_ate = df[df.treatment==1]["purchases"].mean() - df[df.treatment==0]["purchases"].mean()
print(f"Naive ATE: ${naive_ate:.1f}  (biased — high-engagement users buy more)")

# ---- Estimate propensity score with logistic regression ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[covariates])

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_scaled, treatment)
ps = ps_model.predict_proba(X_scaled)[:, 1]   # P(T=1 | X)

df["ps"] = ps

print(f"\nPropensity score summary:")
print(df.groupby("treatment")["ps"].describe().round(3))
```

---

## Propensity Score Matching (PSM)

### The Procedure

1. Estimate the propensity score $\hat{e}(X)$ for all units
2. Match each treated unit to the control unit(s) with the most similar propensity score
3. Check balance on the original covariates (not just on the PS)
4. Estimate ATT on the matched sample

### Caliper Rule of Thumb

The standard caliper for PSM is:

$$\delta = 0.2 \times \text{std}(\text{logit}(\hat{e}(X)))$$

where $\text{logit}(p) = \log(p / (1-p))$. This was proposed by Austin (2011) and has become the default recommendation. It trims poor matches without being too aggressive.

Matching on the logit of the propensity score (rather than the score itself) is also standard — distances on the logit scale are more meaningful near the extremes.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
n = 3000

purchase_freq   = np.random.poisson(5, n)
tenure          = np.random.exponential(2, n)
avg_order_value = np.random.lognormal(4, 0.5, n)
app_usage       = np.random.exponential(3, n)

log_odds = -2 + 0.2*purchase_freq + 0.3*tenure + 0.01*(avg_order_value-55) + 0.15*app_usage
p_enroll = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_enroll)

purchases = 50 + 8*purchase_freq + 10*tenure + 0.3*avg_order_value + 5*app_usage + 50*treatment + np.random.normal(0, 20, n)

df = pd.DataFrame({
    "purchase_freq": purchase_freq, "tenure": tenure,
    "avg_order_value": avg_order_value, "app_usage": app_usage,
    "treatment": treatment, "purchases": purchases
})
covariates = ["purchase_freq", "tenure", "avg_order_value", "app_usage"]

# ---- Step 1: Estimate PS ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[covariates])
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_scaled, treatment)
ps = ps_model.predict_proba(X_scaled)[:, 1]
df["ps"] = ps

# ---- Step 2: Compute caliper ----
logit_ps = np.log(ps / (1 - ps + 1e-9))
df["logit_ps"] = logit_ps
caliper = 0.2 * np.std(logit_ps)
print(f"Caliper (0.2 * std(logit PS)): {caliper:.4f}")

# ---- Step 3: 1-NN matching on logit(PS) ----
treated  = df[df.treatment == 1].reset_index(drop=True)
controls = df[df.treatment == 0].reset_index(drop=True)

nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(controls[["logit_ps"]].values)
distances, indices = nn.kneighbors(treated[["logit_ps"]].values)

within_caliper = distances.flatten() < caliper
n_dropped = (~within_caliper).sum()
print(f"Treated units dropped (caliper): {n_dropped}")

matched_ctrl = controls.iloc[indices.flatten()].reset_index(drop=True)

# ---- Step 4: Balance check ----
def compute_smd(df, covariates, treat_col="treatment"):
    t = df[df[treat_col] == 1]
    c = df[df[treat_col] == 0]
    return {
        col: (t[col].mean() - c[col].mean()) / np.sqrt((t[col].var() + c[col].var()) / 2)
        for col in covariates
    }

smds_before = compute_smd(df, covariates)

matched_sample = pd.concat([
    treated[within_caliper].assign(treatment=1),
    matched_ctrl[within_caliper].assign(treatment=0)
], ignore_index=True)

smds_after = compute_smd(matched_sample, covariates)

print(f"\n{'Covariate':<20} {'Before':>8} {'After':>8}")
for col in covariates:
    flag = "  OK" if abs(smds_after[col]) < 0.1 else "  CHECK"
    print(f"{col:<20} {smds_before[col]:>8.3f} {smds_after[col]:>8.3f}{flag}")

# ---- Step 5: ATT estimate ----
t_matched = treated[within_caliper].reset_index(drop=True)
c_matched = matched_ctrl[within_caliper].reset_index(drop=True)
att_psm = (t_matched["purchases"] - c_matched["purchases"]).mean()

print(f"\nPSM ATT estimate:  ${att_psm:.1f}")
print(f"True ATT:          $50.0")
```

---

## Inverse Probability Weighting (IPW)

Instead of discarding unmatched units, IPW **reweights** every unit so that the weighted sample mimics a randomized experiment.

### The Intuition

In the weighted sample, controls with high propensity scores get upweighted (they're rare in the control group given their profile, so they represent many similar units that *could* have been treated). Controls with low propensity scores get downweighted. The result: the weighted distribution of covariates among controls looks like the distribution among treated units.

### IPW Weights

For ATE estimation, the standard IPW weights are:

$$w_i = \frac{T_i}{e(X_i)} + \frac{1 - T_i}{1 - e(X_i)}$$

Each treated unit is weighted by $1/e(X_i)$ (inverse of the probability of receiving the treatment they got). Each control unit is weighted by $1/(1-e(X_i))$. This makes the weighted treated group represent the full population, and likewise for the weighted control group.

### IPW ATE Estimator

$$\widehat{\text{ATE}}_{\text{IPW}} = \frac{1}{n} \sum_{i=1}^n \frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1 - e(X_i)}$$

This is the **Horvitz-Thompson estimator**. The treated term $T_i Y_i / e(X_i)$ estimates $E[Y(1)]$; the control term estimates $E[Y(0)]$.

### IPW ATT Weights

For the ATT (not ATE), use:

$$w_i = T_i + \frac{(1-T_i) \cdot e(X_i)}{1 - e(X_i)}$$

Treated units get weight 1; controls get weight $e(X_i) / (1-e(X_i))$ — upweighting controls who look similar to the treated.

### Stabilized Weights

Standard IPW weights can be very large when $e(X_i)$ is near 0 or 1, inflating variance. **Stabilized weights** multiply by the marginal treatment probability to bound their range:

$$w_i^{\text{stab}} = \frac{T_i \cdot P(T=1)}{e(X_i)} + \frac{(1-T_i) \cdot P(T=0)}{1-e(X_i)}$$

Stabilized weights have the same expectation as standard weights but lower variance.

### Trimming Extreme Weights

Units with very high weights are influential — a few extreme observations can dominate the estimator. Standard practice: trim weights above the 99th (or 95th) percentile. This introduces a small bias in exchange for substantially lower variance.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 3000

purchase_freq   = np.random.poisson(5, n)
tenure          = np.random.exponential(2, n)
avg_order_value = np.random.lognormal(4, 0.5, n)
app_usage       = np.random.exponential(3, n)

log_odds = -2 + 0.2*purchase_freq + 0.3*tenure + 0.01*(avg_order_value-55) + 0.15*app_usage
p_enroll = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_enroll)

purchases = 50 + 8*purchase_freq + 10*tenure + 0.3*avg_order_value + 5*app_usage + 50*treatment + np.random.normal(0, 20, n)

df = pd.DataFrame({
    "purchase_freq": purchase_freq, "tenure": tenure,
    "avg_order_value": avg_order_value, "app_usage": app_usage,
    "treatment": treatment, "purchases": purchases
})
covariates = ["purchase_freq", "tenure", "avg_order_value", "app_usage"]

# ---- Estimate PS ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[covariates])
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_scaled, treatment)
ps = ps_model.predict_proba(X_scaled)[:, 1]

# Clip PS away from 0/1 to avoid division by zero
ps = np.clip(ps, 1e-6, 1 - 1e-6)
df["ps"] = ps

# ---- IPW ATE weights ----
df["ipw_weight"] = np.where(
    df["treatment"] == 1,
    1.0 / df["ps"],
    1.0 / (1 - df["ps"])
)

# ---- Stabilized weights ----
p_treat = treatment.mean()
df["stable_weight"] = np.where(
    df["treatment"] == 1,
    p_treat / df["ps"],
    (1 - p_treat) / (1 - df["ps"])
)

# ---- Trim extreme weights at 99th percentile ----
w99 = df["stable_weight"].quantile(0.99)
df["trimmed_weight"] = df["stable_weight"].clip(upper=w99)

print("Weight distribution (stabilized):")
print(df["stable_weight"].describe().round(3))
print(f"\nMax trimmed weight (99th pctile cap): {w99:.2f}")

# ---- IPW ATE estimator (Horvitz-Thompson) ----
T = df["treatment"].values
Y = df["purchases"].values
ps_vals = df["ps"].values

ipw_ate = np.mean(T * Y / ps_vals) - np.mean((1 - T) * Y / (1 - ps_vals))
print(f"\nIPW ATE (Horvitz-Thompson):  ${ipw_ate:.1f}")

# ---- Stabilized / trimmed Hajek estimator ----
# Hajek: normalize weights to sum to 1 within each group (more stable)
treated_mask  = df["treatment"] == 1
control_mask  = df["treatment"] == 0

w_t = df.loc[treated_mask,  "trimmed_weight"]
w_c = df.loc[control_mask,  "trimmed_weight"]
y_t = df.loc[treated_mask,  "purchases"]
y_c = df.loc[control_mask,  "purchases"]

hajek_ate = (
    np.average(y_t, weights=w_t)
    - np.average(y_c, weights=w_c)
)
print(f"Stabilized / trimmed Hajek ATE: ${hajek_ate:.1f}")
print(f"True ATE:                       $50.0")

# ---- Balance check after IPW ----
def smd_weighted(col_t, col_c, w_t, w_c):
    mean_t = np.average(col_t, weights=w_t)
    mean_c = np.average(col_c, weights=w_c)
    var_t  = np.average((col_t - mean_t)**2, weights=w_t)
    var_c  = np.average((col_c - mean_c)**2, weights=w_c)
    return (mean_t - mean_c) / np.sqrt((var_t + var_c) / 2)

print(f"\n{'Covariate':<20} {'Unweighted SMD':>16} {'Weighted SMD':>14}")
for col in covariates:
    t_vals = df.loc[treated_mask, col].values
    c_vals = df.loc[control_mask, col].values
    unw    = (t_vals.mean() - c_vals.mean()) / np.sqrt((t_vals.var() + c_vals.var())/2)
    wgt    = smd_weighted(t_vals, c_vals, w_t.values, w_c.values)
    print(f"{col:<20} {unw:>16.3f} {wgt:>14.3f}")
```

---

## Doubly Robust Estimation (AIPW)

The **Augmented Inverse Probability Weighted (AIPW)** estimator, also called the doubly robust estimator, combines IPW with outcome regression. Its defining property: it is consistent if **either** the propensity score model **or** the outcome regression model is correctly specified — not necessarily both.

### The Estimator

$$\widehat{\text{ATE}}_{\text{DR}} = \frac{1}{n} \sum_{i=1}^n \left[
    \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)}
    - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-e(X_i)}
    + \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)
\right]$$

where:
- $\hat{\mu}_1(X_i) = \hat{E}[Y \mid T=1, X_i]$: predicted outcome if treated
- $\hat{\mu}_0(X_i) = \hat{E}[Y \mid T=0, X_i]$: predicted outcome if untreated
- $e(X_i)$: estimated propensity score

### Why "Doubly Robust"?

Think of it as a bias-correction mechanism. The last term $\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)$ is the direct outcome-regression estimate (like the adjustment formula from Chapter 2). The first two terms are IPW-style corrections that kick in when the outcome model is misspecified. Specifically:

- If the **outcome model** is correct, the residuals $Y_i - \hat{\mu}_T(X_i)$ are approximately zero, so the IPW correction terms are small — the estimator behaves like pure outcome regression.
- If the **propensity model** is correct, the IPW reweighting correctly removes confounding — the estimator behaves like pure IPW.
- Both need to fail simultaneously for the estimator to be inconsistent.

This double protection is extremely valuable in practice, where we're never certain either model is exactly right.

### Cross-fitting

In practice, AIPW is best implemented with **cross-fitting** (also called sample splitting): use one half of the data to fit $\hat{\mu}$ and $\hat{e}$, then evaluate the AIPW formula on the other half. This prevents overfitting from inflating the estimate and is required for valid inference when using flexible ML models for $\hat{\mu}$ and $\hat{e}$.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

np.random.seed(42)
n = 3000

purchase_freq   = np.random.poisson(5, n)
tenure          = np.random.exponential(2, n)
avg_order_value = np.random.lognormal(4, 0.5, n)
app_usage       = np.random.exponential(3, n)

log_odds = -2 + 0.2*purchase_freq + 0.3*tenure + 0.01*(avg_order_value-55) + 0.15*app_usage
p_enroll = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_enroll)

purchases = 50 + 8*purchase_freq + 10*tenure + 0.3*avg_order_value + 5*app_usage + 50*treatment + np.random.normal(0, 20, n)

df = pd.DataFrame({
    "purchase_freq": purchase_freq, "tenure": tenure,
    "avg_order_value": avg_order_value, "app_usage": app_usage,
    "treatment": treatment, "purchases": purchases
})
covariates = ["purchase_freq", "tenure", "avg_order_value", "app_usage"]

X = df[covariates].values
T = df["treatment"].values
Y = df["purchases"].values

# ---- Doubly Robust (AIPW) with 5-fold cross-fitting ----
kf = KFold(n_splits=5, shuffle=True, random_state=0)
scaler = StandardScaler()

psi = np.zeros(n)   # influence function values

for train_idx, eval_idx in kf.split(X):
    X_train, X_eval = X[train_idx], X[eval_idx]
    T_train, T_eval = T[train_idx], T[eval_idx]
    Y_train, Y_eval = Y[train_idx], Y[eval_idx]

    # Scale only on training fold
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s  = scaler.transform(X_eval)

    # Propensity model
    ps_model = LogisticRegression(max_iter=1000, C=1.0)
    ps_model.fit(X_train_s, T_train)
    ps_eval = np.clip(ps_model.predict_proba(X_eval_s)[:, 1], 1e-6, 1 - 1e-6)

    # Outcome models (separate for treated and control)
    mu1_model = LinearRegression()
    mu1_model.fit(X_train_s[T_train == 1], Y_train[T_train == 1])
    mu1_eval = mu1_model.predict(X_eval_s)

    mu0_model = LinearRegression()
    mu0_model.fit(X_train_s[T_train == 0], Y_train[T_train == 0])
    mu0_eval = mu0_model.predict(X_eval_s)

    # AIPW influence function
    psi[eval_idx] = (
        T_eval * (Y_eval - mu1_eval) / ps_eval
        - (1 - T_eval) * (Y_eval - mu0_eval) / (1 - ps_eval)
        + mu1_eval - mu0_eval
    )

aipw_ate = psi.mean()
aipw_se  = psi.std() / np.sqrt(n)

print(f"AIPW (DR) ATE estimate: ${aipw_ate:.2f}")
print(f"Standard error:          ${aipw_se:.2f}")
print(f"95% CI:                  (${aipw_ate - 1.96*aipw_se:.2f}, ${aipw_ate + 1.96*aipw_se:.2f})")
print(f"True ATE:                $50.00")
```

### Using EconML for Doubly Robust Estimation

For production use, `econml` provides a well-tested AIPW implementation with cross-fitting built in:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 3000

purchase_freq   = np.random.poisson(5, n)
tenure          = np.random.exponential(2, n)
avg_order_value = np.random.lognormal(4, 0.5, n)
app_usage       = np.random.exponential(3, n)

log_odds = -2 + 0.2*purchase_freq + 0.3*tenure + 0.01*(avg_order_value-55) + 0.15*app_usage
p_enroll = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_enroll)
purchases = 50 + 8*purchase_freq + 10*tenure + 0.3*avg_order_value + 5*app_usage + 50*treatment + np.random.normal(0, 20, n)

X = np.column_stack([purchase_freq, tenure, avg_order_value, app_usage])

try:
    from econml.dr import LinearDRLearner

    # LinearDRLearner is a doubly-robust CATE estimator (also gives ATE)
    dr = LinearDRLearner(
        model_propensity=LogisticRegression(max_iter=1000),
        model_regression=LinearRegression(),
        cv=5
    )
    dr.fit(purchases, treatment, X=X)
    ate_econml = dr.ate(X)
    ate_interval = dr.ate_interval(X, alpha=0.05)
    print(f"EconML DR ATE: ${ate_econml:.2f}")
    print(f"95% CI:        (${ate_interval[0]:.2f}, ${ate_interval[1]:.2f})")
    print(f"True ATE:      $50.00")

except ImportError:
    print("econml not installed — run: pip install econml")
    print("Falling back to manual AIPW (see cross-fitting example above)")
```

---

## Checking Balance with Propensity Scores

After IPW or PSM, verify that the reweighted/matched sample is actually balanced. The mechanics are the same as in Chapter 4 (SMD), but now applied to the weighted sample.

### Overlap Plot

Before any adjustment, check whether the propensity score distributions for treated and control units overlap sufficiently. Poor overlap = poor common support = extrapolation.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)
n = 3000

purchase_freq   = np.random.poisson(5, n)
tenure          = np.random.exponential(2, n)
avg_order_value = np.random.lognormal(4, 0.5, n)
app_usage       = np.random.exponential(3, n)

log_odds = -2 + 0.2*purchase_freq + 0.3*tenure + 0.01*(avg_order_value-55) + 0.15*app_usage
p_enroll = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_enroll)

purchases = (50 + 8*purchase_freq + 10*tenure + 0.3*avg_order_value
             + 5*app_usage + 50*treatment + np.random.normal(0, 20, n))

df = pd.DataFrame({
    "purchase_freq": purchase_freq, "tenure": tenure,
    "avg_order_value": avg_order_value, "app_usage": app_usage,
    "treatment": treatment, "purchases": purchases
})
covariates = ["purchase_freq", "tenure", "avg_order_value", "app_usage"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[covariates])
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_scaled, treatment)
ps = ps_model.predict_proba(X_scaled)[:, 1]
df["ps"] = ps

# ---- Overlap plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: PS distributions by group
for ax, metric, title in zip(
    axes,
    ["ps", "ps"],
    ["Propensity Score Distribution", "Logit PS Distribution"]
):
    if "Logit" in title:
        vals = np.log(df["ps"] / (1 - df["ps"] + 1e-9))
    else:
        vals = df["ps"]

    df_t = vals[df.treatment == 1]
    df_c = vals[df.treatment == 0]

    ax.hist(df_c, bins=40, alpha=0.5, color="steelblue", label="Control", density=True)
    ax.hist(df_t, bins=40, alpha=0.5, color="coral",     label="Treated", density=True)
    ax.set_xlabel("Logit(PS)" if "Logit" in title else "Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# ---- Diagnose overlap ----
ps_treated = df.loc[df.treatment == 1, "ps"]
ps_control = df.loc[df.treatment == 0, "ps"]

print(f"Treated PS range:  [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
print(f"Control PS range:  [{ps_control.min():.3f}, {ps_control.max():.3f}]")
print(f"Overlap region:    [{max(ps_treated.min(), ps_control.min()):.3f}, "
      f"{min(ps_treated.max(), ps_control.max()):.3f}]")

# Units outside common support
no_overlap = ((ps < ps_control.min()) | (ps > ps_control.max())) & (treatment == 1)
print(f"Treated units outside control PS range: {no_overlap.sum()}")
```

### Trimming Non-Overlapping Units

Units whose propensity score falls outside the range of the opposite group are in regions with no common support. Trimming them:

1. Removes units where counterfactual comparison is pure extrapolation
2. Changes the estimand from "ATT for all treated" to "ATT for treated with overlap" — be explicit about this

```python
# Trim treated units with PS above max control PS
# Trim control units with PS below min treated PS
ps_min_treated = df.loc[df.treatment == 1, "ps"].min()
ps_max_control = df.loc[df.treatment == 0, "ps"].max()

df_trimmed = df[
    (df["ps"] >= ps_min_treated) &
    (df["ps"] <= ps_max_control)
].copy()

print(f"Original n: {len(df)}, After trimming: {len(df_trimmed)}")
print(f"Dropped: {len(df) - len(df_trimmed)} units outside common support")
```

---

## Stratification on Propensity Score

A simpler alternative to IPW: divide units into $K$ strata (usually quintiles) based on their propensity score, estimate the treatment effect within each stratum, and average across strata.

$$\widehat{\text{ATE}}_{\text{strat}} = \sum_{k=1}^{K} \frac{n_k}{n} \cdot \widehat{\tau}_k$$

where $\widehat{\tau}_k$ is the mean outcome difference within stratum $k$ and $n_k / n$ is the weight of stratum $k$.

This is simpler than IPW and less sensitive to extreme weights, but less efficient. Rosenbaum and Rubin showed that 5 strata removes roughly 90% of the bias from a single confounder.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 3000

purchase_freq   = np.random.poisson(5, n)
tenure          = np.random.exponential(2, n)
avg_order_value = np.random.lognormal(4, 0.5, n)
app_usage       = np.random.exponential(3, n)

log_odds = -2 + 0.2*purchase_freq + 0.3*tenure + 0.01*(avg_order_value-55) + 0.15*app_usage
p_enroll = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_enroll)
purchases = 50 + 8*purchase_freq + 10*tenure + 0.3*avg_order_value + 5*app_usage + 50*treatment + np.random.normal(0, 20, n)

df = pd.DataFrame({
    "purchase_freq": purchase_freq, "tenure": tenure,
    "avg_order_value": avg_order_value, "app_usage": app_usage,
    "treatment": treatment, "purchases": purchases
})
covariates = ["purchase_freq", "tenure", "avg_order_value", "app_usage"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[covariates])
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_scaled, treatment)
df["ps"] = ps_model.predict_proba(X_scaled)[:, 1]

# ---- Stratification on PS quintiles ----
K = 5
df["ps_stratum"] = pd.qcut(df["ps"], q=K, labels=False)

stratum_effects = []
stratum_ns      = []

print(f"{'Stratum':>8} {'n_T':>6} {'n_C':>6} {'Effect':>10}")
for k in range(K):
    stratum = df[df.ps_stratum == k]
    n_t = (stratum.treatment == 1).sum()
    n_c = (stratum.treatment == 0).sum()
    if n_t == 0 or n_c == 0:
        continue
    tau_k = (stratum[stratum.treatment==1]["purchases"].mean()
             - stratum[stratum.treatment==0]["purchases"].mean())
    stratum_effects.append(tau_k)
    stratum_ns.append(len(stratum))
    print(f"{k:>8} {n_t:>6} {n_c:>6} {tau_k:>10.2f}")

# Weighted average
weights = np.array(stratum_ns) / sum(stratum_ns)
ate_stratified = np.dot(weights, stratum_effects)
print(f"\nStratified ATE estimate:  ${ate_stratified:.1f}")
print(f"True ATE:                 $50.0")
```

---

## Method Comparison

| Method | Estimand | Pros | Cons |
|---|---|---|---|
| **PSM** | ATT | Intuitive, inspectable matched pairs | Discards data; sensitive to caliper choice |
| **IPW** | ATE (or ATT with different weights) | Uses all data; unbiased in large samples | Extreme weights inflate variance |
| **Stabilized IPW** | ATE | Lower variance than IPW | Still sensitive to PS misspecification |
| **AIPW (DR)** | ATE | Doubly robust; efficient | More complex; requires two models |
| **Stratification** | ATE | Simple; robust to extreme weights | Less efficient; coarse |
| **Regression adjustment** | ATE | Simple; efficient | Sensitive to outcome model specification |

In practice:
- Use **PSM** when you want interpretable matched pairs and your sample is large
- Use **IPW** when you want ATE and have good PS overlap
- Use **AIPW** when you want the most robust estimate and are willing to fit two models
- Use **stratification** as a quick sanity check or when overlap is good

---

## Real-World Examples

### Customer Loyalty Program Evaluation

An e-commerce company launches a loyalty program (free shipping, early access). Customers self-select into the program — highly engaged customers join more. Naive comparison of purchases shows program members buy 30% more. Is that the program's effect?

**Propensity score approach**: model enrollment probability using pre-program engagement metrics (purchase frequency, tenure, app usage, email open rate). Use IPW or PSM to compare program members to similar non-members. The adjusted estimate is typically much smaller than the naive 30% — the program works, but most of the difference was pre-existing engagement.

### Healthcare: Treatment Selection Bias

Cardiologists recommend bypass surgery for high-risk patients and medication for lower-risk ones. Comparing surgery vs. medication outcomes naively would make surgery look terrible (sicker patients got surgery). Propensity score methods estimate $P(\text{surgery} \mid X)$ from patient covariates (disease severity, age, ejection fraction, comorbidities) and construct a reweighted or matched comparison group with similar risk profiles.

### Economics: Job Training Self-Selection

Workers in economic distress are more likely to seek training programs. They have lower pre-program earnings, are more likely to be unemployed, and may be in declining industries. Propensity score matching controls for these selection factors, providing a fairer estimate of the program's effect — which was the key methodological innovation in Dehejia and Wahba (1999)'s reanalysis of the LaLonde dataset.

---

## Interview Questions

### Technical Questions

**Q1: What is a propensity score and why is it useful?**

The propensity score is $e(X) = P(T=1 \mid X)$ — the conditional probability of receiving treatment given observed covariates. Its utility comes from the Rosenbaum-Rubin theorem: if unconfoundedness holds given $X$, it also holds given $e(X)$. This collapses a potentially high-dimensional covariate vector into a single scalar for matching or weighting, solving the curse of dimensionality.

---

**Q2: Explain the difference between IPW and propensity score matching.**

Both use the propensity score but in different ways:

- **PSM** discards units by finding matched pairs. It retains only units with similar propensity scores, which can reduce sample size substantially. It targets the ATT (counterfactuals for treated units only).
- **IPW** keeps all units and reweights them. Treated units with high propensity scores get lower weight ($1/e(X)$ — not so unusual to be treated); control units with high propensity scores get high weight ($1/(1-e(X))$ — unusual to be untreated, so they represent many treated-like units). IPW typically targets the ATE.

PSM is more interpretable (you can inspect matched pairs). IPW is more efficient (uses all data) but sensitive to extreme propensity scores.

---

**Q3: What is the doubly robust estimator and why is it valuable?**

The AIPW estimator combines IPW with outcome regression:

$$\widehat{\text{ATE}}_{\text{DR}} = \frac{1}{n}\sum_i \left[\frac{T_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-e(X_i)} + \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)\right]$$

It is consistent if **either** the propensity model **or** the outcome model is correctly specified. In practice, we're never certain either model is exactly right. Doubly robust provides insurance: even if one model is somewhat misspecified, the other corrects for it. This is especially valuable in high-dimensional settings where both models are flexible ML methods.

---

**Q4: What is the overlap assumption and what happens when it fails?**

The overlap (positivity) assumption requires $0 < e(X) < 1$ for all $X$ in the support of the data. This means every unit has a nonzero probability of receiving each treatment level.

When overlap fails:
- In PSM: treated units with no comparable controls are dropped (changing the estimand)
- In IPW: propensity scores near 0 or 1 produce extreme weights that inflate variance and can cause numerical instability
- Structurally: you're trying to estimate counterfactuals for regions where one treatment was never observed — any estimate relies purely on model extrapolation

Diagnosis: plot PS distributions for treated and control. If they don't overlap, trim or restrict to the common support region and be explicit that the estimand has changed.

---

**Q5: How do you assess whether a propensity score model is "good"?**

Not by predictive accuracy (AUC). The propensity score model is good if, after matching or weighting on it, **covariate balance improves**. The diagnostic is the SMD for each covariate before and after adjustment. Target: $|\text{SMD}| < 0.1$ for all covariates after adjustment.

A model with high AUC but poor balance is failing at its job. Conversely, a model with mediocre AUC that achieves excellent balance is doing exactly what's needed. You may need to add interactions or polynomial terms to the PS model to achieve balance on non-linear covariate relationships.

---

### Case Study Questions

**Case 1: A loyalty program was launched for high-value customers. You want to estimate whether the program increased customer purchases. How would you use propensity score methods?**

First, define the causal question precisely: what is the ATT of program enrollment on purchases over the next 6 months, among customers who enrolled?

Draw the DAG: what drives enrollment? Purchase frequency, tenure, average order value, app engagement all likely predict enrollment and also independently predict future purchases — these are confounders.

Estimate the propensity score using logistic regression on all pre-enrollment metrics. Check the overlap plot — if high-value customers all enrolled (PS near 1 for treated), overlap will be poor. In that case, PSM might drop most treated units; IPW with trimming is a better choice.

After IPW or PSM, check SMD for all covariates. Target < 0.1. Report the ATT or ATE with confidence intervals from bootstrapping or the AIPW influence function. Note the key assumption: all confounders are measured (unconfoundedness).

---

**Case 2: You use logistic regression to estimate the propensity score, but after IPW the balance table shows several covariates with |SMD| > 0.2. What do you do?**

The propensity score model is not capturing the confounding well enough. The logistic regression is likely missing non-linear relationships or interactions. Steps:

1. Add polynomial terms and interactions to the logistic regression (e.g., `purchase_freq^2`, `tenure * purchase_freq`)
2. Switch to a more flexible model: random forest or gradient boosting for propensity estimation
3. Check whether trimming extreme weights changes the balance (sometimes a few outlier units drive the imbalance)
4. Consider propensity score matching instead of IPW — matching sometimes achieves better balance because it directly discards poor matches
5. Re-check SMD after each modification

The criterion for "good enough" is empirical balance (SMD < 0.1), not statistical fit of the PS model.

---

**Case 3: Your IPW estimate has very high variance — the confidence interval is [$-50, $150] when the point estimate is $50. What causes this and how do you fix it?**

High variance in IPW is almost always caused by extreme propensity scores (near 0 or 1), which produce very large weights ($1/e(X_i)$ blows up when $e(X_i) \approx 0$). A handful of control units with high PS that ended up untreated get enormous weights and dominate the estimator.

Fixes, in order of aggressiveness:
1. **Stabilized weights**: multiply by the marginal treatment probability, bounding the expected weight to 1
2. **Trim at 99th percentile**: cap weights above the 99th percentile — small bias, large variance reduction
3. **Trim at 95th percentile**: more aggressive trimming for very bad overlap
4. **Switch to PSM**: matching discards extreme-PS units rather than inflating their weight
5. **Switch to AIPW**: the outcome regression component reduces reliance on IPW weights in regions of poor overlap

Always report what trimming was applied — it changes the estimand slightly by effectively excluding units with near-zero or near-one propensity scores.

---

**Case 4: A colleague says "I'll just use logistic regression to control for confounders — why bother with propensity scores?" How do you respond?**

Regression adjustment and propensity scores are both valid approaches to confounding, with different strengths:

- **Regression** is more efficient when the outcome model is correctly specified. It uses all observations and directly estimates the treatment effect as a coefficient.
- **Propensity scores** are better when: (1) you want to separate the design stage from the analysis stage — PSM lets you check balance before looking at outcomes; (2) you have poor overlap — PS methods make this explicit rather than extrapolating silently; (3) you care about transparency — stakeholders can understand "we compared similar customers" more easily than a regression coefficient.

The real answer: use **AIPW**, which does both simultaneously and is doubly robust to misspecification of either model. If only one method is available, regression is fine for simple settings; propensity scores become essential when overlap is poor or when you need to establish credibility by demonstrating balance.
