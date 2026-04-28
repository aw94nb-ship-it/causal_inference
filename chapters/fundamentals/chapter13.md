# Chapter 13: Cross-Fitting

## The Problem: In-Sample ATE Overfitting

Both S- and T-learners have a subtle overfitting problem: they make predictions on the **same data they were trained on**. In standard supervised learning, cross-validation catches this — but here the goal is not out-of-sample prediction accuracy; it's computing an in-sample ATE. Even a perfectly cross-validated model can overfit the ATE estimate when predicting on its own training data.

**Why standard CV doesn't catch this**: Cross-validation measures prediction error on held-out data. But the ATE is computed over the full dataset, including training observations. A model that memorizes training data will produce biased ATE estimates even if its held-out prediction error looks fine.

---

## How Cross-Fitting Works

**Cross-fitting** solves this by splitting the data and separating training from prediction:

1. Split data $D = \{(x^i, y^i, z^i)\}$ into two equal halves $D_1$ and $D_2$ (stratified by treatment)
2. Train model $f$ on $D_1$
3. For each observation in $D_2$: compute $f(1, z^i)$ and $f(0, z^i)$, then compute $\text{ATE}_2 = \frac{1}{|D_2|}\sum_{D_2}[f(1,z^i) - f(0,z^i)]$
4. Swap roles: train on $D_2$, predict on $D_1$, compute $\text{ATE}_1$
5. Final ATE = average of $\text{ATE}_1$ and $\text{ATE}_2$

Each observation's ATE contribution is computed by a model that **never saw that observation during training**. This eliminates the in-sample overfitting problem.

**Key difference from standard CV:**

| | Standard CV | Cross-fitting |
|---|---|---|
| **Goal** | Measure prediction accuracy on held-out data | Estimate causal ATE without in-sample bias |
| **Split basis** | Random folds | Stratified by treatment status |
| **What you measure** | Out-of-sample prediction error | ATE from each fold, then averaged |
| **What it prevents** | Prediction overfitting | In-sample ATE overfitting |

---

## Data Structure

Same panel structure as T-learner — one row per unit with outcome Y, treatment X, and confounders Z. The split is stratified by treatment so each fold has a representative mix of treated and control units.

| unit_id | X (treatment) | Z1 | Z2 | Y | fold |
|---------|--------------|----|----|---|------|
| 001 | 1 | 0.3 | −0.5 | 4.1 | D1 |
| 002 | 0 | −1.2 | 0.8 | 1.3 | D1 |
| 003 | 1 | 0.7 | 0.2 | 5.6 | D2 |
| 004 | 0 | −0.4 | −0.9 | 0.9 | D2 |
| ... | ... | ... | ... | ... | ... |

**Step 1**: Train f₁ and f₀ on D1 only. Predict for every row in D2.
**Step 2**: Train f₁ and f₀ on D2 only. Predict for every row in D1.
**Step 3**: ATE = average of (f₁(z) − f₀(z)) across all rows, where each row's prediction came from the fold it was NOT in.

---

## Methods in Practice

1. **Stratified split**: split data 50/50, stratified by treatment X, so each half has similar treated/control proportions.
2. **Fold 1 → Fold 2**: train T-learner (f₁ on treated, f₀ on control) on fold 1. Predict τ(z) = f₁(z) − f₀(z) for each observation in fold 2. Average = ATE₂.
3. **Fold 2 → Fold 1**: swap. Train on fold 2, predict on fold 1. Average = ATE₁.
4. **Final ATE**: (ATE₁ + ATE₂) / 2.
5. **Extend to k folds**: for more stable estimates, use k-fold cross-fitting (same logic generalized). k=2 is standard; k=5 is common in Double Machine Learning.

**What good output looks like:**
- ATE fold 1→2: 2.03
- ATE fold 2→1: 1.97
- Cross-fit ATE: 2.00 (true = 2.0) — unbiased, compared to T-learner without cross-fitting which may overfit

---

## Code Example: S-Learner vs. T-Learner vs. Cross-Fitting

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 3000

# DGP: treatment X has a true ATE of 2.0
# Confounders Z1, Z2 — both affect X and Y
z1 = np.random.normal(0, 1, n)
z2 = np.random.normal(0, 1, n)
x = np.random.binomial(1, 1 / (1 + np.exp(-z1 - z2)))  # confounded assignment
y = 2.0 * x + 1.5 * z1 - z2 + np.random.normal(0, 1, n)  # true ATE = 2.0

df = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})
Z = df[["z1", "z2"]].values
XZ = df[["x", "z1", "z2"]].values

# --- Naive estimate ---
naive = df[df.x == 1]["y"].mean() - df[df.x == 0]["y"].mean()
print(f"Naive ATE:       {naive:.3f}  (biased)")

# --- S-Learner ---
s_model = GradientBoostingRegressor(n_estimators=100).fit(XZ, df["y"])
xz1 = df[["x", "z1", "z2"]].copy(); xz1["x"] = 1
xz0 = df[["x", "z1", "z2"]].copy(); xz0["x"] = 0
s_ate = (s_model.predict(xz1) - s_model.predict(xz0)).mean()
print(f"S-Learner ATE:   {s_ate:.3f}")

# --- T-Learner (no cross-fitting) ---
f1 = GradientBoostingRegressor(n_estimators=100).fit(Z[x == 1], df["y"][x == 1])
f0 = GradientBoostingRegressor(n_estimators=100).fit(Z[x == 0], df["y"][x == 0])
t_ate = (f1.predict(Z) - f0.predict(Z)).mean()
print(f"T-Learner ATE:   {t_ate:.3f}")

# --- T-Learner with Cross-Fitting ---
from sklearn.model_selection import train_test_split

D1, D2 = train_test_split(df, test_size=0.5, stratify=df["x"], random_state=0)

def t_learner_ate(train_df, pred_df):
    f1 = GradientBoostingRegressor(n_estimators=100).fit(
        train_df[train_df.x == 1][["z1", "z2"]], train_df[train_df.x == 1]["y"]
    )
    f0 = GradientBoostingRegressor(n_estimators=100).fit(
        train_df[train_df.x == 0][["z1", "z2"]], train_df[train_df.x == 0]["y"]
    )
    return (f1.predict(pred_df[["z1", "z2"]]) - f0.predict(pred_df[["z1", "z2"]])).mean()

ate_fold1 = t_learner_ate(D1, D2)
ate_fold2 = t_learner_ate(D2, D1)
cf_ate = (ate_fold1 + ate_fold2) / 2
print(f"Cross-fit ATE:   {cf_ate:.3f}")
print(f"True ATE:        2.000")
```

---

## Connection to Double Machine Learning (Chapter 8)

Cross-fitting is the foundation of **Double Machine Learning (DML)**. DML extends the idea: instead of just cross-fitting the outcome model, it also cross-fits the treatment model (propensity score), then combines both residuals to estimate the causal effect with reduced bias from high-dimensional confounders.

The cross-fitting principle is the same — train on one fold, predict on the other — but applied to both:
- The outcome model: $\hat{Y}$ from confounders Z
- The treatment model: $\hat{X}$ from confounders Z

The DML estimator then regresses outcome residuals $(Y - \hat{Y})$ on treatment residuals $(X - \hat{X})$ to recover the causal effect.

---

## Interview Questions

**Q: What is cross-fitting and why is it needed?**

S- and T-learners predict on the same data they trained on. This causes in-sample overfitting of the ATE — distinct from prediction overfitting, and not caught by standard cross-validation. Cross-fitting fixes this: split data into D₁ and D₂, train on D₁ and predict on D₂ to get ATE₂, swap to get ATE₁, and average. Each observation's ATE contribution is estimated by a model that never saw it during training.

**Q: How is cross-fitting different from k-fold cross-validation?**

Standard CV splits randomly and measures prediction accuracy on held-out data. Cross-fitting splits stratified by treatment status and measures ATE — not prediction accuracy. The goal is causal estimation, not predictive validation. A model can pass standard CV and still produce biased ATE estimates due to in-sample overfitting.
