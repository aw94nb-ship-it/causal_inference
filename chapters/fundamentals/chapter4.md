# Chapter 4: Matching Methods

Chapters 2 and 3 established that the core challenge in observational causal inference is confounding: treated and untreated units differ systematically in ways that also affect the outcome. The adjustment formula handles this by standardizing over confounders analytically. **Matching** takes a different route — it physically constructs a comparison group that looks like the treated group, discarding the parts of the data where comparisons would be apples-to-oranges. The result is a matched dataset where treated and control units are similar on observed covariates, and a simple difference in outcomes approximates the causal effect. This chapter covers the mechanics, the estimand, diagnostic tools, and the practical workflow for matching in Python.

---

## The Matching Intuition

The fundamental problem of causal inference is that we never observe the same unit under both treatment and control. For a user who adopted a feature, we don't know what their retention would have been without it. Matching sidesteps this by finding a **near-identical unit** from the control group — a "statistical twin" — and using that unit's outcome as the counterfactual.

### What Matching Does

Imagine you're evaluating a job training program. Participants tend to be younger and less educated than non-participants — these are confounders. A naive comparison of earnings would conflate the training effect with the age and education differences.

Matching fixes this by:
1. Taking each treated unit (program participant)
2. Finding a control unit (non-participant) with similar age and education
3. Discarding all unmatched control units
4. Estimating the effect as the average outcome difference in matched pairs

The core insight: **matching constructs a comparable control group from observational data**, mimicking what randomization does by design in an RCT.

### Why Matching Instead of Regression?

Both matching and regression adjust for confounders. The difference is transparency and assumptions:

| | Matching | Regression |
|---|---|---|
| **How it adjusts** | Physically pairs similar units | Parametric extrapolation |
| **Extrapolation** | Minimal — stays within common support | Can extrapolate outside data range |
| **Assumptions** | Less model-specific | Requires correct functional form |
| **Interpretability** | Can inspect matched pairs | Effect is a coefficient |
| **Handling non-overlap** | Discards non-overlapping units | Extrapolates (silently) |

Matching makes the assumption of **common support** explicit. If you can't find a good match for a treated unit, that unit is dropped — rather than extrapolated over via regression. This is often more honest.

---

## The Estimand: ATT

Matching typically targets the **Average Treatment Effect on the Treated (ATT)**, not the overall ATE.

$$\text{ATT} = E[Y(1) - Y(0) \mid T = 1]$$

In plain language: **among units that actually received the treatment, what would their outcomes have been if they hadn't?**

### Why ATT, Not ATE?

Because matching works by finding controls for each treated unit. We're estimating the counterfactual for treated units specifically. We don't need good matches for every control unit — only enough controls to pair with the treated units.

Contrast with ATE, which requires counterfactuals for both treated ($Y(0)$ for treated) and control units ($Y(1)$ for controls):

$$\text{ATE} = E[Y(1) - Y(0)]$$

If you want ATT, you match treated units to controls.
If you want ATE, you also need to match control units to treated units (or use IPW — Chapter 5).

### Business Interpretation

In the job training example:
- **ATT**: "Among workers who enrolled in training, how much did their earnings increase because of training?"
- **ATE**: "If we enrolled a randomly selected worker in training, how much would their earnings change?"

ATT answers the policy-relevant question when the program already exists and you want to evaluate it for the people who actually participated.

---

## Exact Matching

The simplest form: match treated and control units that have **exactly the same values** on all covariates.

### How It Works

For each treated unit $i$, find a control unit $j$ such that $X_j = X_i$ (covariate values are identical). Then estimate ATT as:

$$\widehat{\text{ATT}}_{\text{exact}} = \frac{1}{n_T} \sum_{i: T_i=1} \left( Y_i - \frac{1}{|M_i|} \sum_{j \in M_i} Y_j \right)$$

where $M_i$ is the set of matched controls for treated unit $i$, and $n_T$ is the number of treated units.

In words: for each treated unit, average the outcomes of all exact matches from the control group, then take the difference. Average these differences over all treated units.

### Coded Example

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

# Simulate a job training dataset with discrete covariates
# Confounders: age_group (young/old), education (HS/college)
# True ATT = 2000 (program raises earnings by $2000)

age_group = np.random.choice([0, 1], n, p=[0.5, 0.5])       # 0=young, 1=old
education = np.random.choice([0, 1], n, p=[0.6, 0.4])       # 0=HS, 1=college

# Treatment: young and college-educated more likely to participate
p_treat = 0.1 + 0.3 * (1 - age_group) + 0.2 * education
p_treat = np.clip(p_treat, 0, 1)
treatment = np.random.binomial(1, p_treat)

# Outcome: earnings affected by age, education, and treatment
earnings = (
    20000
    + 3000 * education
    - 1500 * age_group
    + 2000 * treatment             # true ATT = ATE = 2000 (homogeneous)
    + np.random.normal(0, 1000, n)
)

df = pd.DataFrame({
    "age_group": age_group,
    "education": education,
    "treatment": treatment,
    "earnings": earnings
})

print(f"Treated units:   {treatment.sum()}")
print(f"Control units:   {(1 - treatment).sum()}")

# Naive comparison (biased — confounded)
naive_att = (
    df[df.treatment == 1]["earnings"].mean()
    - df[df.treatment == 0]["earnings"].mean()
)
print(f"\nNaive ATT estimate: ${naive_att:.0f}")

# --- Exact Matching ---
treated = df[df.treatment == 1].reset_index(drop=True)
controls = df[df.treatment == 0].reset_index(drop=True)

match_keys = ["age_group", "education"]   # covariates to match on exactly

diffs = []
matched_treated_idx = []

for idx, row in treated.iterrows():
    # Find all controls with identical covariate values
    mask = (controls[match_keys] == row[match_keys]).all(axis=1)
    matched_controls = controls[mask]

    if len(matched_controls) == 0:
        continue   # no exact match — drop this treated unit

    # Counterfactual = average earnings of matched controls
    y_counterfactual = matched_controls["earnings"].mean()
    diffs.append(row["earnings"] - y_counterfactual)
    matched_treated_idx.append(idx)

att_exact = np.mean(diffs)
print(f"Exact matching ATT: ${att_exact:.0f}  (based on {len(diffs)} matched units)")
print(f"True ATT:           $2000")
```

### Limitation: Curse of Dimensionality

Exact matching works only when covariates are **discrete and low-dimensional**. With 10 binary covariates, there are $2^{10} = 1024$ possible covariate profiles. Most cells will be sparsely populated or empty — many treated units will find no exact match. This is the **curse of dimensionality**.

The solution: use approximate matching on a distance metric, or collapse the covariates into a scalar (propensity score — Chapter 5).

---

## Distance-Based Matching

When exact matching isn't feasible, match on a distance measure between covariate vectors.

### Euclidean Distance

The simplest option: straight-line distance in covariate space.

$$d_E(x_i, x_j) = \sqrt{\sum_{k=1}^{p} (x_{ik} - x_{jk})^2}$$

**Problem**: highly sensitive to scale. If age is measured in years (range 20–60) and income in dollars (range 20k–100k), income will dominate the distance metric purely because of units.

Always standardize covariates before using Euclidean distance.

### Mahalanobis Distance

A better choice: Mahalanobis distance accounts for the scale and correlation of covariates:

$$d_M(x_i, x_j) = \sqrt{(x_i - x_j)^T \Sigma^{-1} (x_i - x_j)}$$

where $\Sigma$ is the covariance matrix of the covariates.

This is equivalent to Euclidean distance after transforming the data to have identity covariance — it's **scale-invariant** and **decorrelates** the covariates. Two units that differ on a high-variance covariate are not penalized as heavily as two units that differ on a low-variance one (where the difference is more unusual).

### Nearest-Neighbor Matching

The most common algorithm: for each treated unit, find the control unit(s) with the smallest distance.

**Without replacement**: each control unit can only be matched to one treated unit. Less bias from reuse, but fewer matches available.

**With replacement**: a control unit can be matched to multiple treated units. Typically yields better matches (lower bias) but increases variance in the ATT estimate because the same controls are used repeatedly.

**$k$-nearest-neighbor matching**: match each treated unit to $k$ closest controls and average their outcomes. More controls per treated unit reduces variance at the cost of some additional bias (farther-away matches).

### Caliper Matching

A crucial safeguard: only accept a match if the distance is below a threshold $\delta$:

$$\text{match } i \text{ to } j \text{ only if } d(x_i, x_j) < \delta$$

If no control unit is within the caliper, the treated unit is **dropped** from the analysis. This enforces common support — you only estimate the effect where you have genuinely comparable controls.

The caliper introduces a bias-variance tradeoff: tighter calipers give better matches (lower bias) but drop more treated units (higher variance and changes the effective estimand).

### Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 1000

# Simulate job training dataset with continuous covariates
# Confounders: age, years_education
# True ATT = 2000

age = np.random.normal(35, 8, n)
education = np.random.normal(13, 2, n)

# Treatment: younger + more educated -> more likely to participate
log_odds = -1 + 0.05 * (30 - age) + 0.2 * (education - 12)
p_treat = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_treat)

# Outcome: earnings
earnings = (
    20000
    + 500 * (age - 35)
    + 1500 * (education - 12)
    + 2000 * treatment
    + np.random.normal(0, 2000, n)
)

df = pd.DataFrame({
    "age": age,
    "education": education,
    "treatment": treatment,
    "earnings": earnings
})

print(f"Treated: {treatment.sum()}, Control: {(1-treatment).sum()}")

naive_att = (
    df[df.treatment == 1]["earnings"].mean()
    - df[df.treatment == 0]["earnings"].mean()
)
print(f"Naive ATT: ${naive_att:.0f}  (biased — younger workers earn less)")

# ---- 1-Nearest-Neighbor Matching with Mahalanobis Distance ----

treated_df = df[df.treatment == 1].copy()
control_df = df[df.treatment == 0].copy()

X_treated = treated_df[["age", "education"]].values
X_control = control_df[["age", "education"]].values

# Compute covariance matrix on the full sample for Mahalanobis
cov = np.cov(df[["age", "education"]].values.T)
cov_inv = np.linalg.inv(cov)

# sklearn NearestNeighbors with Mahalanobis metric
nn = NearestNeighbors(
    n_neighbors=1,
    metric="mahalanobis",
    metric_params={"VI": cov_inv}
)
nn.fit(X_control)
distances, indices = nn.kneighbors(X_treated)

# Build matched dataset
matched_control_idx = indices.flatten()
matched_controls = control_df.iloc[matched_control_idx].reset_index(drop=True)
treated_reset = treated_df.reset_index(drop=True)

# Caliper: discard matches with distance > 1.0 (in Mahalanobis units)
caliper = 1.0
within_caliper = distances.flatten() < caliper

n_dropped = (~within_caliper).sum()
print(f"\nCaliper = {caliper}: {n_dropped} treated units dropped (no close match)")

# ATT estimate on matched sample within caliper
diffs = (
    treated_reset["earnings"].values[within_caliper]
    - matched_controls["earnings"].values[within_caliper]
)
att_matched = diffs.mean()

print(f"Matched ATT (1-NN, Mahalanobis, caliper={caliper}): ${att_matched:.0f}")
print(f"True ATT: $2000")
```

---

## Assessing Balance

After matching, the most important diagnostic is whether the matched sample is actually balanced — do the treated and matched control units look similar on observed covariates?

### Standardized Mean Difference (SMD)

The standard metric for covariate balance is the **Standardized Mean Difference**:

$$\text{SMD} = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{(s_T^2 + s_C^2)/2}}$$

where $\bar{X}_T$ and $\bar{X}_C$ are the covariate means in the treated and control groups, and $s_T^2$, $s_C^2$ are their variances.

The SMD expresses the difference in covariate means in **standard deviation units**, making it comparable across covariates measured on different scales (unlike a raw difference or a p-value).

**Rule of thumb**: $|\text{SMD}| < 0.1$ indicates good balance. Values above 0.25 suggest meaningful imbalance.

Note: SMD is preferred over t-test p-values for balance checking. A t-test's p-value depends on sample size — in a large sample, even a trivially small imbalance will be "statistically significant." SMD measures the magnitude of imbalance, which is what matters.

### Love Plot

A **Love plot** is a dot plot of the SMD for each covariate, before and after matching. It gives a quick visual summary of how much matching improved balance.

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

# Simulate dataset with more covariates for a richer balance table
age = np.random.normal(35, 8, n)
education = np.random.normal(13, 2, n)
tenure = np.random.exponential(3, n)            # years at current job
prior_earnings = 18000 + 400 * (age - 35) + 800 * (education - 12) + np.random.normal(0, 2000, n)

log_odds = -1 + 0.05 * (30 - age) + 0.2 * (education - 12) - 0.1 * tenure
p_treat = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_treat)

earnings = 20000 + 500*(age-35) + 1500*(education-12) + 0.1*prior_earnings + 2000*treatment + np.random.normal(0,2000,n)

df = pd.DataFrame({
    "age": age,
    "education": education,
    "tenure": tenure,
    "prior_earnings": prior_earnings,
    "treatment": treatment,
    "earnings": earnings
})

covariates = ["age", "education", "tenure", "prior_earnings"]

# ----- Compute SMD -----
def compute_smd(df, covariates, treat_col="treatment"):
    treated = df[df[treat_col] == 1]
    control = df[df[treat_col] == 0]
    smds = {}
    for col in covariates:
        mean_t = treated[col].mean()
        mean_c = control[col].mean()
        var_t  = treated[col].var()
        var_c  = control[col].var()
        smd = (mean_t - mean_c) / np.sqrt((var_t + var_c) / 2)
        smds[col] = smd
    return smds

smds_before = compute_smd(df, covariates)

# ----- Run 1-NN Mahalanobis matching -----
treated_df = df[df.treatment == 1].copy()
control_df = df[df.treatment == 0].copy()

X_treated = treated_df[covariates].values
X_control = control_df[covariates].values

cov_mat = np.cov(df[covariates].values.T)
cov_inv = np.linalg.inv(cov_mat)

nn = NearestNeighbors(n_neighbors=1, metric="mahalanobis", metric_params={"VI": cov_inv})
nn.fit(X_control)
_, indices = nn.kneighbors(X_treated)

matched_controls = control_df.iloc[indices.flatten()].reset_index(drop=True)
treated_reset = treated_df.reset_index(drop=True)

# Combine matched sample
matched_df = pd.concat([
    treated_reset.assign(matched_group="treated"),
    matched_controls.assign(matched_group="control")
], ignore_index=True)

smds_after = compute_smd(matched_df, covariates)

# ----- Balance table -----
balance = pd.DataFrame({
    "Before Matching": smds_before,
    "After Matching":  smds_after
}).round(3)
print("=== Balance Table (SMD) ===")
print(balance)
print(f"\nRule of thumb: |SMD| < 0.10 is well-balanced")

# ----- Love Plot -----
fig, ax = plt.subplots(figsize=(7, 4))
y_pos = range(len(covariates))

ax.scatter(list(smds_before.values()), y_pos, label="Before matching", marker="o", s=60, color="coral")
ax.scatter(list(smds_after.values()),  y_pos, label="After matching",  marker="D", s=60, color="steelblue")
ax.axvline(0,    color="black", linewidth=0.8)
ax.axvline(0.1,  color="gray",  linewidth=0.8, linestyle="--", alpha=0.6)
ax.axvline(-0.1, color="gray",  linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(covariates)
ax.set_xlabel("Standardized Mean Difference")
ax.set_title("Love Plot: Covariate Balance Before and After Matching")
ax.legend()
plt.tight_layout()
plt.show()

# ----- ATT estimate -----
att = (treated_reset["earnings"] - matched_controls["earnings"]).mean()
print(f"\nMatched ATT estimate: ${att:.0f}")
print(f"True ATT:             $2000")
```

---

## Full Matching Pipeline

Here is the complete workflow — simulate, match, check balance, and compute ATT — in one self-contained block:

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

np.random.seed(0)
n = 2000

# ---- Data Generating Process ----
# Job training program (LaLonde-style)
# Confounders: age, education both affect participation AND earnings
# True ATT = $2000

age       = np.random.normal(34, 7, n)
education = np.random.normal(12, 2, n)

log_odds = -0.5 + 0.06 * (30 - age) + 0.15 * (education - 12)
p_treat  = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, p_treat)

earnings = (
    20000
    + 400 * (age - 34)
    + 1200 * (education - 12)
    + 2000 * treatment
    + np.random.normal(0, 1500, n)
)

df = pd.DataFrame({
    "age": age,
    "education": education,
    "treatment": treatment,
    "earnings": earnings
})

# ---- Naive estimate ----
naive = df.groupby("treatment")["earnings"].mean().diff().iloc[-1]
print(f"Naive ATT:   ${naive:.0f}  (biased)")

# ---- Match: 1-NN with replacement, Mahalanobis ----
covariates = ["age", "education"]
treated = df[df.treatment == 1].reset_index(drop=True)
controls = df[df.treatment == 0].reset_index(drop=True)

cov_mat = np.cov(df[covariates].values.T)
nn = NearestNeighbors(
    n_neighbors=1,
    metric="mahalanobis",
    metric_params={"VI": np.linalg.inv(cov_mat)}
)
nn.fit(controls[covariates].values)
_, idx = nn.kneighbors(treated[covariates].values)

matched_ctrl = controls.iloc[idx.flatten()].reset_index(drop=True)

# ---- Compute ATT ----
att_matched = (treated["earnings"] - matched_ctrl["earnings"]).mean()
print(f"Matched ATT: ${att_matched:.0f}")
print(f"True ATT:    $2000")

# ---- Balance check ----
def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)

print("\n=== SMD before and after matching ===")
print(f"{'Covariate':<15} {'Before':>8} {'After':>8}")
for col in covariates:
    before = smd(df[df.treatment==1][col], df[df.treatment==0][col])
    after  = smd(treated[col], matched_ctrl[col])
    flag   = "  OK" if abs(after) < 0.1 else "  CHECK"
    print(f"{col:<15} {before:>8.3f} {after:>8.3f}{flag}")
```

---

## Regression Adjustment After Matching

Even after matching, some residual imbalance may remain — especially if the caliper was loose or if $k > 1$ neighbors were used. A common remedy: run a **regression on the matched sample** to mop up any leftover covariate differences.

This is sometimes called the **bias-corrected matching estimator** or a preview of the **Augmented IPW (AIPW)** estimator covered in Chapter 5. The idea:

1. Match to balance covariates (reduce confounding)
2. Regress outcome on treatment + covariates within matched sample (reduce residual bias)

The resulting estimator is more robust: matching handles the bulk of confounding; regression cleans up what matching leaves behind.

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

np.random.seed(1)
n = 2000

age       = np.random.normal(34, 7, n)
education = np.random.normal(12, 2, n)
log_odds  = -0.5 + 0.06*(30-age) + 0.15*(education-12)
treatment = np.random.binomial(1, 1/(1+np.exp(-log_odds)))
earnings  = 20000 + 400*(age-34) + 1200*(education-12) + 2000*treatment + np.random.normal(0, 1500, n)

df = pd.DataFrame({"age": age, "education": education, "treatment": treatment, "earnings": earnings})

covariates = ["age", "education"]
treated  = df[df.treatment == 1].reset_index(drop=True)
controls = df[df.treatment == 0].reset_index(drop=True)

cov_mat = np.cov(df[covariates].values.T)
nn = NearestNeighbors(n_neighbors=1, metric="mahalanobis",
                      metric_params={"VI": np.linalg.inv(cov_mat)})
nn.fit(controls[covariates].values)
_, idx = nn.kneighbors(treated[covariates].values)

matched_ctrl = controls.iloc[idx.flatten()].reset_index(drop=True)

# Combine matched sample
matched_sample = pd.concat([
    treated.assign(treatment=1),
    matched_ctrl.assign(treatment=0)
], ignore_index=True)

# Step 1: matching-only ATT
att_matching_only = (treated["earnings"] - matched_ctrl["earnings"]).mean()

# Step 2: regression on matched sample
reg = LinearRegression().fit(
    matched_sample[covariates + ["treatment"]],
    matched_sample["earnings"]
)
att_regression_adjusted = reg.coef_[covariates.__len__()]   # coefficient on treatment

print(f"Matching-only ATT:             ${att_matching_only:.0f}")
print(f"Matching + regression ATT:     ${att_regression_adjusted:.0f}")
print(f"True ATT:                      $2000")
```

The regression step is especially valuable when the matched sample is large and residual imbalance is non-trivial. This doubly-robust flavor means: if the matching is slightly off, the regression corrects it, and vice versa.

---

## Limitations of Matching

Matching is a powerful tool with real constraints. Know these before applying it.

### 1. Only Handles Observed Confounding

Matching controls only for confounders you've measured and included. If there's an **unmeasured confounder** — a variable that affects both treatment and outcome and isn't in your data — matching cannot help. This is the fundamental limitation of all observational methods that rely on the **unconfoundedness assumption**:

$$(Y(0), Y(1)) \perp T \mid X$$

If this assumption fails (hidden confounders), the matched estimate is still biased.

### 2. Requires Overlap (Common Support)

Matching requires that for every treated unit, there exist control units with similar covariate values. If overlap is poor — e.g., all young educated workers participated in the training program — many treated units will have no good match and will be dropped. The effective estimand shifts from "ATT for all treated" to "ATT for matched treated," which may be a specific subgroup.

### 3. Variance Can Be High

With 1-NN matching without replacement, the estimator can be noisy — especially when the treated group is large relative to the control group. Using more neighbors ($k > 1$) reduces variance but adds bias. Using matching with replacement introduces correlation between matched pairs, requiring variance estimators that account for the reuse of controls.

### 4. Doesn't Scale Well to Many Covariates

Matching directly on high-dimensional covariate vectors suffers from the curse of dimensionality — distances become uninformative in high-dimensional spaces. The standard solution is to match on the **propensity score** (Chapter 5), which collapses all covariates into a single number.

### 5. Doesn't Estimate ATE Directly

Standard matching gives the ATT — the effect for the treated units. If you want the ATE (population-average effect), you'd also need to match control units to treated units, or switch to inverse probability weighting (Chapter 5).

### Summary Table

| Issue | Impact | Remedy |
|---|---|---|
| Unmeasured confounders | Bias | Sensitivity analysis (Chapter 7) |
| Poor overlap | Changes estimand, high variance | Check overlap, caliper matching |
| High-dimensional covariates | Poor matches | Propensity score matching (Ch. 5) |
| High variance | Wide confidence intervals | More neighbors, regression adjustment |
| ATT only | Can't estimate ATE | IPW (Ch. 5) |

---

## Real-World Examples

### Job Training Programs (LaLonde Dataset)

The classic benchmark for matching methods. Robert LaLonde (1986) studied the National Supported Work Demonstration — a randomized job training program. He then compared the experimental estimates to estimates from observational methods using the same treated group but non-experimental controls (CPS and PSID survey data). Most regression methods failed to reproduce the experimental benchmark. Dehejia and Wahba (1999) later showed that propensity score matching substantially improved the observational estimates.

**Confounders**: age, education, race, prior earnings, employment history.
**Lesson**: matching works well when the covariate set fully captures why workers self-selected into training.

### Medical Treatment: Surgery vs. Medication

Surgeons select patients for surgery based on disease severity, age, comorbidities. Comparing surgery vs. medication outcomes naively is heavily confounded. Matching on a rich set of patient characteristics (APACHE score, comorbidity index, age, lab values) constructs a comparable medication group and allows estimation of the surgical ATT.

**Key diagnostic**: SMD on all pre-treatment clinical variables. Balance < 0.1 on each variable gives confidence the comparison is fair.

### Tech: Feature Adoption

A product team wants to know if users who adopted a new feature (e.g., a saved searches feature) have higher 90-day retention. Feature adoption is confounded by engagement level — high-engagement users both adopt features more readily and have higher baseline retention.

Matching approach: match each feature adopter to a non-adopter with similar baseline engagement score, tenure, and activity metrics. The matched ATT answers: "For users who adopted the feature, how much did it improve their retention relative to what it would have been without adoption?"

---

## Interview Questions

### Technical Questions

**Q1: What estimand does matching target and why?**

Matching typically targets the **ATT** — Average Treatment Effect on the Treated. The reason is mechanical: we find matches for treated units from the control group, so we're constructing counterfactuals for the treated units specifically. We estimate "what would treated units' outcomes have been under control?" — which is the ATT. If you want the ATE, you'd also need to construct counterfactuals for control units, which standard matching doesn't do. IPW (Chapter 5) is better suited for ATE.

---

**Q2: What is the Standardized Mean Difference (SMD) and why use it instead of a p-value for balance checking?**

The SMD measures the difference in covariate means between treated and control groups, normalized by the pooled standard deviation:

$$\text{SMD} = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{(s_T^2 + s_C^2)/2}}$$

It's preferred over t-test p-values because p-values depend on sample size — in a large matched sample, even a tiny and practically irrelevant imbalance will be "statistically significant." SMD measures the *magnitude* of imbalance in a scale-free way, regardless of sample size. The rule of thumb is $|\text{SMD}| < 0.1$.

---

**Q3: What is the curse of dimensionality in matching, and what is the standard solution?**

When matching on many covariates directly, the covariate space becomes extremely high-dimensional. In high dimensions, distances between points lose their discriminative power — all points are roughly equidistant from each other, so "nearest neighbors" aren't actually very similar. Exact matches are essentially impossible with continuous or high-dimensional features.

The standard solution is **propensity score matching**: estimate $e(X) = P(T=1 \mid X)$ (a single scalar) and match on it. Rosenbaum and Rubin (1983) proved that if unconfoundedness holds given $X$, it also holds given $e(X)$ — so you can do all your matching on a single number.

---

**Q4: What is caliper matching and when should you use it?**

Caliper matching only accepts a match if the distance between treated and control units is below a threshold $\delta$. Treated units with no control within the caliper are dropped. You should use it when:
- You want to enforce common support (only estimate the effect where you have genuinely comparable controls)
- The nearest-neighbor matches for some treated units are very poor (high bias from a bad match is worse than dropping the unit)

The tradeoff: a tighter caliper reduces bias but drops more treated units, increasing variance and narrowing the population for which you're estimating the ATT. A standard caliper for propensity score matching is $0.2 \times \text{std}(\text{logit}(e(X)))$.

---

**Q5: Matching removes observed confounding. What about unmeasured confounders?**

Matching cannot address unmeasured confounders. The unconfoundedness assumption requires that all variables affecting both treatment and outcome are included in $X$. If there's a hidden confounder, the matched estimate is still biased.

The response to this limitation is **sensitivity analysis** (Chapter 7): quantify how strong a hidden confounder would need to be to explain away your result. If your conclusion holds even under a sizable hidden confounder, it's more credible. If a small hidden confounder would flip the result, be cautious.

---

### Case Study Questions

**Case 1: Your team wants to evaluate a job training program for laid-off workers using observational data. Participation was voluntary. How would you use matching to estimate the program's effect on earnings?**

Start by drawing the DAG: what variables affect both participation and earnings? Likely candidates are age, education level, pre-program earnings, industry, and geographic region. Collect these covariates for both participants and non-participants.

Run nearest-neighbor Mahalanobis matching with a caliper. Check balance using SMD for every covariate — target < 0.1. If some covariates remain imbalanced, add them to a regression on the matched sample (bias-corrected estimator). Compute ATT on the matched sample. Report the estimate alongside the Love plot showing balance improvement, and note which assumption (unconfoundedness) is untestable. Run a sensitivity analysis (Rosenbaum bounds) to check robustness to hidden confounding.

---

**Case 2: A PM argues that users who use Feature X have 40% higher 30-day retention. Should you conclude the feature causes higher retention?**

No — this is almost certainly confounded. High-engagement users are more likely to discover and use Feature X, and they also have higher baseline retention. The 40% difference is an association, not a causal effect.

To estimate the causal effect: match each Feature X user to a non-user with similar baseline engagement score, tenure, activity level, and product tier. Compute the retention difference in the matched sample. Check that SMD < 0.1 on all covariates after matching. The matched ATT is the best estimate of the retention lift attributable to feature adoption, under the assumption that baseline engagement captures all the confounding.

---

**Case 3: You've run matching and your Love plot shows most covariates are balanced, but one covariate (prior earnings) still has an SMD of 0.3. What do you do?**

An SMD of 0.3 after matching is material — prior earnings is a strong predictor of future earnings, so this imbalance will bias the ATT estimate. Options:

1. **Tighten the caliper** on the distance metric: this will drop more treated units but improve matches.
2. **Add prior earnings explicitly** to the Mahalanobis distance computation if it wasn't already included, or increase its weight.
3. **Run regression adjustment** on the matched sample controlling for prior earnings — this mops up the residual imbalance.
4. **Use propensity score matching** (Chapter 5) with prior earnings as one of the inputs — propensity scores may give better matches on this specific covariate if it's a dominant driver of selection.

The worst option is to report the matched estimate without addressing the imbalance and hope the reviewer doesn't notice.

---

**Case 4: When is matching preferred over regression? When is regression preferred?**

| Situation | Prefer Matching | Prefer Regression |
|---|---|---|
| **Overlap concern** | Yes — matching explicitly enforces it | No — regression extrapolates silently |
| **Many covariates** | No — use propensity score instead | Yes — regression handles many covariates |
| **Non-linear relationships** | Yes (if matching on PS) | Requires flexible specification |
| **Interpretability** | Yes — matched pairs are inspectable | Results are coefficients |
| **Want ATE** | Harder — need two-way matching | Standard |
| **Robustness to model misspec** | Yes | No |

In practice, matching followed by regression adjustment on the matched sample combines the strengths of both: matching ensures overlap and removes most confounding; regression mops up residual imbalance and improves efficiency.
