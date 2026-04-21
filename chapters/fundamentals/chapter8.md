# Chapter 8: Double Machine Learning

When you have dozens or hundreds of confounders, standard linear regression breaks down: regularization (like LASSO) shrinks the treatment coefficient toward zero along with the noise, producing biased causal estimates. Double Machine Learning (DML), introduced by Chernozhukov et al. (2018), solves this by separating the problem — use any ML model to partial out the confounders from both $Y$ and $T$, then estimate the causal effect from the residuals. The key innovation is **cross-fitting**: fitting the nuisance models on held-out folds so that overfitting bias does not contaminate the final estimate. DML gives you the flexibility of modern ML with the statistical rigor of causal inference.

---

## The Problem with High-Dimensional Controls

Suppose you want to estimate the effect of price changes on demand, but you have 50+ customer features (demographics, browsing history, purchase history, session behavior) that all confound the relationship. The standard fix — include them all in a linear regression — fails in two ways:

1. **Overfitting**: with $p \approx n$, OLS fits noise and standard errors explode.
2. **Regularization bias**: LASSO and Ridge shrink *all* coefficients toward zero, including the treatment coefficient. The regularizer does not know that the treatment variable is "special" — it penalizes it just like any other predictor. The result is a biased estimate of the causal effect.

### Why Naive LASSO Fails

Consider a simple data-generating process:

$$Y = \theta T + \mathbf{X}\boldsymbol{\beta} + \varepsilon$$

where $\theta$ is the true treatment effect and $\mathbf{X}$ contains many confounders. Running LASSO on $(T, \mathbf{X})$ to predict $Y$ will shrink $\hat{\theta}$ toward zero. The amount of shrinkage depends on the penalty $\lambda$ — and there is no value of $\lambda$ that is simultaneously "right" for the high-dimensional nuisance parameters $\boldsymbol{\beta}$ and "right" (zero) for the treatment coefficient $\theta$.

This is the **regularization bias** problem: regularization is necessary to handle high-dimensional confounders, but it unavoidably distorts the treatment effect estimate.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate high-dimensional confounding
n = 500
p = 50  # 50 confounders — high-dimensional

# True parameters
theta_true = 2.0   # true treatment effect

# Confounders
X = np.random.randn(n, p)

# Treatment: depends on confounders
beta_t = np.random.randn(p) * 0.3
T = X @ beta_t + np.random.randn(n)

# Outcome: depends on treatment AND confounders
beta_y = np.random.randn(p) * 0.5
Y = theta_true * T + X @ beta_y + np.random.randn(n)

# --- Naive approach 1: OLS with all controls (p=50, works if n >> p) ---
TX = np.column_stack([T, X])
ols = LinearRegression().fit(TX, Y)
theta_ols = ols.coef_[0]

# --- Naive approach 2: LASSO on (T, X) -> read off T coefficient ---
from sklearn.linear_model import Lasso
alphas = np.logspace(-3, 1, 50)
lasso_results = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=5000).fit(TX, Y)
    lasso_results.append(lasso.coef_[0])

print(f"True theta:       {theta_true:.3f}")
print(f"OLS estimate:     {theta_ols:.3f}")
print(f"LASSO estimates (range): {min(lasso_results):.3f} to {max(lasso_results):.3f}")
print(f"LASSO estimate at optimal alpha: ~{np.median(lasso_results):.3f}")
print()
print("Naive LASSO shrinks the treatment coefficient — biased regardless of alpha.")
```

---

## The FWL Connection: Why Residualizing Works

Recall the **Frisch-Waugh-Lovell (FWL) theorem** from Chapter 6: in a linear regression of $Y$ on $T$ and $X$, the coefficient on $T$ is numerically identical to the coefficient you get from:

1. Regressing $Y$ on $X$ → get residuals $\tilde{Y}$
2. Regressing $T$ on $X$ → get residuals $\tilde{T}$
3. Regressing $\tilde{Y}$ on $\tilde{T}$

The residuals $\tilde{Y}$ and $\tilde{T}$ represent the parts of $Y$ and $T$ that cannot be explained by $X$ — the "X-adjusted" versions. The key insight: step 3 is a simple univariate regression on clean residuals, so nothing in step 3 biases the treatment coefficient.

**Double ML generalizes this**: instead of using linear regression in steps 1 and 2, use any ML model — LASSO, random forest, gradient boosting, neural networks. The residual regression in step 3 remains unbiased as long as the nuisance models in steps 1 and 2 are sufficiently accurate.

This is powerful because:
- ML models can capture nonlinear confounding that linear regression misses
- You can use all your variable selection tools freely on the nuisance models
- The final step is just a simple regression on residuals

---

## The Double ML Algorithm

### Setup: The Partially Linear Model

DML is designed for the **partially linear model**:

$$Y = \theta T + g(\mathbf{X}) + \varepsilon$$

where:
- $\theta$ is the **average treatment effect (ATE)** — the scalar we want to estimate
- $g(\mathbf{X})$ is an **arbitrary function** of the confounders (no parametric form assumed)
- $\varepsilon$ is noise with $E[\varepsilon \mid T, \mathbf{X}] = 0$

This generalizes linear regression: we only require linearity in $T$, not in $\mathbf{X}$. The function $g$ can be any nonlinear combination of features — we never need to specify its form.

Rewriting: $Y - g(\mathbf{X}) = \theta \cdot (T - e(\mathbf{X})) + \varepsilon$, where $e(\mathbf{X}) = E[T \mid \mathbf{X}]$. This is the residual form: residualized Y on residualized T gives $\theta$.

### The Algorithm

**Step 1**: Fit $\hat{m}(\mathbf{X}) = E[Y \mid \mathbf{X}]$ using any ML model (not including $T$).

**Step 2**: Fit $\hat{e}(\mathbf{X}) = E[T \mid \mathbf{X}]$ using any ML model.

**Step 3**: Compute residuals:

$$\tilde{Y}_i = Y_i - \hat{m}(\mathbf{X}_i), \quad \tilde{T}_i = T_i - \hat{e}(\mathbf{X}_i)$$

**Step 4**: Regress $\tilde{Y}$ on $\tilde{T}$ (no intercept) — the coefficient is the causal estimate:

$$\hat{\theta} = \frac{\sum_i \tilde{T}_i \tilde{Y}_i}{\sum_i \tilde{T}_i^2}$$

### Why Cross-Fitting Is Necessary

If you fit steps 1–2 on the same data you use in step 4, overfitting creates a problem. A model that perfectly memorizes the training data will produce near-zero residuals — and near-zero residuals in step 4 make the estimate noisy and biased.

**Cross-fitting** solves this using $K$-fold logic:
1. Split data into $K$ folds (typically $K=5$).
2. For each fold $k$: fit the nuisance models $\hat{m}$ and $\hat{e}$ on all other $K-1$ folds, then predict on fold $k$ to get out-of-sample residuals.
3. Concatenate the out-of-sample residuals from all folds → run the final regression on these.

Because residuals are always computed on held-out data, overfitting cannot inflate or deflate the final estimate. Chernozhukov et al. (2018) show this produces an estimator that is $\sqrt{n}$-consistent and asymptotically normal, even when the nuisance models converge at slower rates.

$$\hat{\theta}_{DML} = \frac{\sum_i \tilde{T}_i \tilde{Y}_i}{\sum_i \tilde{T}_i^2}$$

The variance of this estimator is:

$$\widehat{\text{Var}}(\hat{\theta}) = \frac{\frac{1}{n}\sum_i \tilde{T}_i^2 \hat{\varepsilon}_i^2}{\left(\frac{1}{n}\sum_i \tilde{T}_i^2\right)^2}$$

where $\hat{\varepsilon}_i = \tilde{Y}_i - \hat{\theta} \tilde{T}_i$.

---

## Manual DML Implementation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# -------------------------------------------------------
# Data generating process: high-dimensional confounding
# -------------------------------------------------------
n = 1000
p = 50
theta_true = 2.0

X = np.random.randn(n, p)

# Treatment depends on confounders (nonlinearly)
beta_t = np.zeros(p)
beta_t[:10] = np.random.randn(10) * 0.5   # only first 10 matter
T = X @ beta_t + 0.5 * X[:, 0] ** 2 + np.random.randn(n)

# Outcome depends on treatment + nonlinear confounders
beta_y = np.zeros(p)
beta_y[:10] = np.random.randn(10) * 0.5
Y = theta_true * T + X @ beta_y + 0.3 * X[:, 1] ** 2 + np.random.randn(n)

# -------------------------------------------------------
# Naive LASSO: include T in LASSO, read off coefficient
# -------------------------------------------------------
from sklearn.linear_model import Lasso
TX = np.column_stack([T, X])
lasso_cv = LassoCV(cv=5, max_iter=10000).fit(TX, Y)
theta_naive_lasso = lasso_cv.coef_[0]

# -------------------------------------------------------
# DML with cross-fitting
# -------------------------------------------------------
def double_ml(Y, T, X, n_splits=5, model_y=None, model_t=None):
    """
    Double ML with cross-fitting.
    
    Parameters
    ----------
    Y : outcome (n,)
    T : treatment (n,)
    X : confounders (n, p)
    n_splits : number of CV folds
    model_y : sklearn model for E[Y|X]
    model_t : sklearn model for E[T|X]
    
    Returns
    -------
    theta : ATE estimate
    se : standard error
    Y_res : residuals of Y
    T_res : residuals of T
    """
    if model_y is None:
        model_y = LassoCV(cv=5, max_iter=10000)
    if model_t is None:
        model_t = LassoCV(cv=5, max_iter=10000)
    
    n = len(Y)
    Y_res = np.zeros(n)
    T_res = np.zeros(n)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        # Step 1: fit E[Y|X] on training fold, predict on test fold
        m_hat = model_y.__class__(**model_y.get_params())
        m_hat.fit(X_train, Y_train)
        Y_res[test_idx] = Y_test - m_hat.predict(X_test)
        
        # Step 2: fit E[T|X] on training fold, predict on test fold
        e_hat = model_t.__class__(**model_t.get_params())
        e_hat.fit(X_train, T_train)
        T_res[test_idx] = T_test - e_hat.predict(X_test)
    
    # Step 4: regress Y_res on T_res
    theta = np.sum(T_res * Y_res) / np.sum(T_res ** 2)
    
    # Standard error (heteroskedasticity-robust)
    eps = Y_res - theta * T_res
    var_theta = np.mean(T_res ** 2 * eps ** 2) / (np.mean(T_res ** 2) ** 2) / n
    se = np.sqrt(var_theta)
    
    return theta, se, Y_res, T_res


# Run DML with LASSO nuisance models
theta_dml, se_dml, Y_res, T_res = double_ml(Y, T, X)

# OLS with all controls (baseline)
ols = LinearRegression().fit(np.column_stack([T, X]), Y)
theta_ols = ols.coef_[0]

print("=" * 50)
print(f"True theta:        {theta_true:.3f}")
print(f"OLS estimate:      {theta_ols:.3f}")
print(f"Naive LASSO:       {theta_naive_lasso:.3f}  ← biased by regularization")
print(f"DML estimate:      {theta_dml:.3f}  ± {1.96 * se_dml:.3f} (95% CI)")
print("=" * 50)

# 95% confidence interval
ci_lo = theta_dml - 1.96 * se_dml
ci_hi = theta_dml + 1.96 * se_dml
print(f"DML 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"True value in CI: {ci_lo <= theta_true <= ci_hi}")
```

### Intuition Check: What Do the Residuals Look Like?

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: T residuals vs T — confirms confounding was removed
axes[0].scatter(T, T_res, alpha=0.3, s=10)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Raw T')
axes[0].set_ylabel('T residuals (T - E[T|X])')
axes[0].set_title('T after partialling out X\n(residuals should be uncorrelated with X)')

# Right: Y_res vs T_res — this is the final DML regression
axes[1].scatter(T_res, Y_res, alpha=0.3, s=10)
x_line = np.linspace(T_res.min(), T_res.max(), 100)
axes[1].plot(x_line, theta_dml * x_line, color='red', linewidth=2,
             label=f'DML slope = {theta_dml:.2f}')
axes[1].set_xlabel('T residuals')
axes[1].set_ylabel('Y residuals')
axes[1].set_title('DML final step: regress Y_res on T_res')
axes[1].legend()

plt.tight_layout()
plt.savefig('dml_residuals.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Slope of residual regression = {theta_dml:.3f} (true = {theta_true:.3f})")
```

---

## Heterogeneous Treatment Effects with DML

So far we estimated a single ATE $\theta$. But treatment effects often vary across individuals. The **interactive regression model** extends DML to estimate the **Conditional Average Treatment Effect (CATE)**:

$$Y = \theta(\mathbf{X}) \cdot T + g(\mathbf{X}) + \varepsilon$$

where $\theta(\mathbf{X}) = E[Y(1) - Y(0) \mid \mathbf{X}]$ varies with covariates.

### The DML-CATE Approach

After computing residuals $\tilde{Y}$ and $\tilde{T}$ (same as before), we can rewrite:

$$\tilde{Y}_i \approx \theta(\mathbf{X}_i) \cdot \tilde{T}_i + \varepsilon_i$$

Dividing by $\tilde{T}_i$ (conceptually): $\tilde{Y}_i / \tilde{T}_i \approx \theta(\mathbf{X}_i)$. More precisely, we minimize:

$$\sum_i \left(\tilde{Y}_i - \theta(\mathbf{X}_i) \cdot \tilde{T}_i \right)^2$$

This is a **weighted regression** problem where $\tilde{T}_i$ are weights. We can plug in any model for $\theta(\cdot)$:
- **LinearDML**: $\theta(\mathbf{X}) = \mathbf{X}\boldsymbol{\gamma}$ — linear CATE
- **CausalForestDML**: $\theta(\mathbf{X})$ is a random forest — nonparametric CATE

---

## Full EconML Implementation

```python
# pip install econml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# -------------------------------------------------------
# Simulate: ad effectiveness with 50 behavioral features
# Business question: effect of ad exposure on purchase,
# controlling for 50 user behavioral features.
# The effect is HETEROGENEOUS — power users respond more.
# -------------------------------------------------------
n = 2000
p = 50

# Behavioral features (browsing, purchase history, demographics)
X = np.random.randn(n, p)
feature_names = [f'feature_{i}' for i in range(p)]

# Ad exposure probability depends on features (confounded)
# (Users with high feature_0 are more likely to see ads AND buy)
propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
T = np.random.binomial(1, propensity)   # binary: saw ad or not

# True CATE: effect is larger for users with high feature_0
# (power users respond more to ads)
true_cate = 0.5 + 1.5 * X[:, 0]        # heterogeneous effect
true_ate = np.mean(true_cate)

# Outcome: purchase (binary treatment, continuous outcome)
g_X = 2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2]  # confounders affect outcome
Y = true_cate * T + g_X + np.random.randn(n)

print(f"True ATE: {true_ate:.3f}")
print(f"True CATE range: [{true_cate.min():.2f}, {true_cate.max():.2f}]")

# -------------------------------------------------------
# LinearDML: assumes CATE is linear in X
# Good when you want interpretable heterogeneity
# -------------------------------------------------------
linear_dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingClassifier(n_estimators=100),
    discrete_treatment=True,    # T is binary
    cv=5,                        # 5-fold cross-fitting
    random_state=42
)

linear_dml.fit(Y, T, X=X, W=None)  # W = extra controls not used in CATE

# ATE estimate
ate_linear = linear_dml.ate(X)
ate_ci = linear_dml.ate_interval(X, alpha=0.05)
print(f"\nLinearDML ATE: {ate_linear:.3f} (95% CI: [{ate_ci[0]:.3f}, {ate_ci[1]:.3f}])")

# CATE estimates for each individual
cate_linear = linear_dml.effect(X)
print(f"LinearDML CATE range: [{cate_linear.min():.2f}, {cate_linear.max():.2f}]")
print(f"Correlation with true CATE: {np.corrcoef(cate_linear, true_cate)[0,1]:.3f}")

# -------------------------------------------------------
# CausalForestDML: nonparametric CATE (more flexible)
# Better when heterogeneity structure is unknown
# -------------------------------------------------------
cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingClassifier(n_estimators=100),
    discrete_treatment=True,
    n_estimators=200,
    cv=5,
    random_state=42
)

cf_dml.fit(Y, T, X=X)

ate_cf = cf_dml.ate(X)
ate_cf_ci = cf_dml.ate_interval(X, alpha=0.05)
print(f"\nCausalForestDML ATE: {ate_cf:.3f} (95% CI: [{ate_cf_ci[0]:.3f}, {ate_cf_ci[1]:.3f}])")

cate_cf = cf_dml.effect(X)
print(f"CausalForest CATE range: [{cate_cf.min():.2f}, {cate_cf.max():.2f}]")
print(f"Correlation with true CATE: {np.corrcoef(cate_cf, true_cate)[0,1]:.3f}")
```

### Plotting CATE vs. a Key Covariate

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sort by feature_0 (the driver of heterogeneity)
sort_idx = np.argsort(X[:, 0])

# Left plot: CausalForest CATE vs feature_0
axes[0].scatter(X[:, 0], cate_cf, alpha=0.2, s=8, label='Estimated CATE')
axes[0].scatter(X[:, 0], true_cate, alpha=0.2, s=8, color='red', label='True CATE')
axes[0].set_xlabel('feature_0 (user engagement score)')
axes[0].set_ylabel('Estimated CATE')
axes[0].set_title('CausalForestDML: CATE vs. User Engagement')
axes[0].legend()

# Right plot: compare LinearDML vs CausalForest vs true CATE
x_vals = X[sort_idx, 0]
axes[1].plot(x_vals, true_cate[sort_idx], label='True CATE', linewidth=2, color='black')
axes[1].plot(x_vals, cate_linear[sort_idx], label='LinearDML', linewidth=2,
             color='blue', linestyle='--')
axes[1].plot(x_vals, cate_cf[sort_idx], label='CausalForestDML', linewidth=2,
             color='green', linestyle='-.')
axes[1].set_xlabel('feature_0 (sorted)')
axes[1].set_ylabel('CATE')
axes[1].set_title('Estimated CATE: Linear vs. Forest')
axes[1].legend()
axes[1].axhline(true_ate, color='gray', linestyle=':', label='ATE')

plt.tight_layout()
plt.savefig('dml_cate.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Confidence Intervals on CATE

```python
# CausalForestDML provides pointwise confidence intervals
cate_cf_lb, cate_cf_ub = cf_dml.effect_interval(X, alpha=0.05)

# Show CI for first 10 individuals
print("Individual CATE estimates with 95% CI (first 10):")
print(f"{'Unit':>5} | {'True':>6} | {'Est.':>6} | {'95% CI':>20}")
print("-" * 45)
for i in range(10):
    print(f"{i:>5} | {true_cate[i]:>6.3f} | {cate_cf[i]:>6.3f} | "
          f"[{cate_cf_lb[i]:>6.3f}, {cate_cf_ub[i]:>6.3f}]")
```

---

## Real-World Business Examples

### Example 1: Pricing — Effect of Price on Demand

**Problem**: You want to estimate the price elasticity of demand. Prices are not randomly set — they vary by product, season, and region, all of which also affect demand. You have 60+ features.

```python
import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)
n = 3000

# Features: product category, region, season, customer segments (60 total)
p = 60
X = np.random.randn(n, p)

# Price is set higher for premium products and during peak seasons
# (both also drive higher demand — confounding)
log_price = 0.4 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5

# True price elasticity = -1.5 (demand drops 1.5% per 1% price increase)
# But elasticity varies: higher for commodity products (low feature_0)
true_elasticity = -1.5 - 0.5 * X[:, 0]   # heterogeneous

# Log demand
log_demand = true_elasticity * log_price + 0.6 * X[:, 0] + 0.2 * X[:, 2] + np.random.randn(n)

model = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingRegressor(n_estimators=100),
    cv=5, random_state=42
)
model.fit(log_demand, log_price, X=X)

ate = model.ate(X)
ate_ci = model.ate_interval(X, alpha=0.05)
true_ate = np.mean(true_elasticity)

print("Pricing Elasticity Estimation")
print(f"True average elasticity:  {true_ate:.3f}")
print(f"DML estimated elasticity: {ate:.3f} (95% CI: [{ate_ci[0]:.3f}, {ate_ci[1]:.3f}])")
print("Interpretation: a 1% price increase reduces demand by ~{:.1f}%".format(abs(ate)))
```

### Example 2: HR Analytics — Training Program Effect on Productivity

**Problem**: Employees who volunteer for training programs are already high performers. You need to control for 40+ employee characteristics (tenure, past performance, role, team, etc.) to isolate the training effect.

```python
np.random.seed(1)
n = 1500
p = 40  # employee characteristics

X = np.random.randn(n, p)
feature_names = ['tenure', 'past_perf', 'role_level', 'team_size'] + \
                [f'skill_{i}' for i in range(36)]

# High performers self-select into training
propensity = 1 / (1 + np.exp(-(0.8 * X[:, 1] + 0.3 * X[:, 0])))
T_training = np.random.binomial(1, propensity)

# True effect: training increases productivity by 0.8 units on average
# But effect is larger for junior employees (lower role_level)
true_effect = 0.8 - 0.4 * X[:, 2]
Y_productivity = true_effect * T_training + 1.5 * X[:, 1] + 0.5 * X[:, 0] + np.random.randn(n)

# Naive OLS (confounded: ignores self-selection)
from sklearn.linear_model import LinearRegression
naive_ols = LinearRegression().fit(
    np.column_stack([T_training, X]), Y_productivity
)

# DML
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingClassifier(n_estimators=100),
    discrete_treatment=True, cv=5, random_state=42
)
dml.fit(Y_productivity, T_training, X=X)

print("HR Analytics: Training Program Effect on Productivity")
print(f"True ATE:          {np.mean(true_effect):.3f}")
print(f"Naive OLS:         {naive_ols.coef_[0]:.3f}  ← upward bias from self-selection")
print(f"DML estimate:      {dml.ate(X):.3f}")
ci = dml.ate_interval(X, alpha=0.05)
print(f"DML 95% CI:        [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## When to Use DML

| Scenario | Linear Regression | Propensity Score / IPW | Double ML |
|---|---|---|---|
| Few confounders ($p < 10$) | Good | Good | Works but overkill |
| Many confounders ($p > 30$) | Biased / unstable | Unstable weights | **Recommended** |
| Nonlinear confounding | Biased | Biased if model wrong | **Handles naturally** |
| Continuous treatment | Easy | Requires extension | **Natural** |
| Binary treatment | Easy | Natural | Works |
| Heterogeneous effects (CATE) | Hard | Hard | **Natural with EconML** |
| Interpretability | High | Moderate | Moderate |
| Computational cost | Low | Low | Higher |
| Requires overlap assumption | No | Yes | Yes (for CATE) |

### Decision Checklist

Use DML when you check **any** of the following:
- You have more than ~30 control variables
- You suspect confounders affect the outcome nonlinearly
- Your treatment is continuous (e.g., price, dosage, spend)
- You need heterogeneous treatment effects
- You want to use a rich feature set without worrying about which controls to include

Use simpler methods when:
- You have a well-specified linear model with few controls
- Your dataset is small ($n < 200$) — cross-fitting needs enough data per fold
- Interpretability of the full model matters more than causal precision

---

## Summary

Double ML provides a principled way to use ML models for causal inference. The three key ideas are:

1. **Partial out confounders from both Y and T** (FWL generalization): residuals carry the causal signal.
2. **Cross-fitting**: always predict on held-out folds to avoid overfitting bias contaminating the estimate.
3. **Plug in any ML model** for the nuisance functions — the final causal estimate is still valid as long as the nuisance models are consistent.

The estimator:

$$\hat{\theta}_{DML} = \frac{\sum_i (T_i - \hat{e}(\mathbf{X}_i))(Y_i - \hat{m}(\mathbf{X}_i))}{\sum_i (T_i - \hat{e}(\mathbf{X}_i))^2}$$

is $\sqrt{n}$-consistent, asymptotically normal, and robust to moderate mis-specification of the nuisance models — making it the go-to method for high-dimensional observational causal inference.

---

## Interview Questions

### Technical Q&A

**Q1: Why does naive LASSO give a biased treatment effect, and how does DML fix it?**

LASSO regularizes all coefficients including the treatment coefficient. Since regularization pushes coefficients toward zero, it under-estimates the treatment effect. The penalty $\lambda$ that controls overfitting in the confounders also shrinks the treatment coefficient, and there is no $\lambda$ that correctly handles both simultaneously. DML fixes this by never putting the treatment in the same penalized regression as the confounders: nuisance models only involve $X$ (not $T$), so regularization only affects the confounders. The treatment effect comes from a simple unpenalized regression of $\tilde{Y}$ on $\tilde{T}$.

**Q2: What is cross-fitting and why is it necessary?**

Cross-fitting is the DML analog of cross-validation: split data into $K$ folds, fit nuisance models on $K-1$ folds, predict on the held-out fold. Residuals are always computed out-of-sample. Without cross-fitting, an overfit nuisance model produces near-zero in-sample residuals, which makes the final regression unstable. Cross-fitting ensures that the nuisance models' overfitting does not propagate into the causal estimate. Theoretically, it achieves "sample splitting" which is required for the $\sqrt{n}$-consistency proof.

**Q3: What does the DML estimator actually identify — ATE, ATT, or something else?**

Under the partially linear model $Y = \theta T + g(X) + \varepsilon$, the DML estimator identifies the **ATE** ($\theta$ is a constant). When you extend to the interactive model $Y = \theta(X) T + g(X) + \varepsilon$, DML identifies the **CATE** $\theta(X)$ pointwise. If you average the CATE over the population you get the ATE; over the treated you get the ATT. DML requires the standard unconfoundedness assumption: $Y(t) \perp T \mid X$ — it cannot handle unmeasured confounders.

**Q4: When would you use LinearDML vs. CausalForestDML?**

Use **LinearDML** when you believe the treatment effect heterogeneity is approximately linear in the features — it is more interpretable and provides parametric confidence intervals. Use **CausalForestDML** when heterogeneity is complex and nonlinear, you have enough data ($n > 500$) to support a forest, and you care about individual-level CATE estimates. CausalForestDML also provides honest confidence intervals via the Wager & Athey (2018) forest machinery, which is important for statistical inference on CATE.

**Q5: How do you choose the nuisance model in DML, and does the choice matter?**

The nuisance models (for $E[Y|X]$ and $E[T|X]$) are theoretically "free" — DML is valid for any consistent estimator. In practice, a better nuisance model (lower prediction error) gives smaller residuals with less noise, which improves the efficiency of the final estimate. Common choices: LASSO for sparse, high-dimensional settings; gradient boosting for moderate-dimensional data with nonlinearities; random forests for robustness. A key requirement is that both nuisance models converge at a rate faster than $n^{-1/4}$ (a mild condition), which nearly all modern ML models satisfy.

### Case Study Questions

**Case 1: Pricing Elasticity**

You work at an e-commerce company and need to estimate price elasticity of demand. Prices are set algorithmically based on competitor prices, inventory levels, and 50+ product/customer features — all of which also affect demand. Describe how you would use DML to estimate the elasticity. What is your outcome? Treatment? Nuisance models? How would you validate the result?

*Key points to hit*: outcome = log demand, treatment = log price, confounders = all pricing features. Use gradient boosting for nuisance models (continuous T and Y). Validate by checking residual balance (residuals of T should be uncorrelated with X), checking model fit in each fold, and sanity-checking the sign and magnitude of the elasticity against domain knowledge. Consider testing for heterogeneous elasticity by product category.

**Case 2: Ad Campaign Measurement**

Your marketing team ran a targeted ad campaign where users with higher engagement were more likely to be shown ads (not randomized). You have 80+ behavioral features. The team wants to know: (a) the average lift in purchases, and (b) which user segments respond most. How would you approach this? What assumptions are required?

*Key points to hit*: this calls for DML with CausalForestDML for CATE. The key assumption is unconfoundedness: all variables that affect both ad exposure and purchase are in X. Discuss potential unmeasured confounders (e.g., intent signals not captured). Validate the first-stage (propensity model): if the model can predict T well, confounding is properly accounted for. Report CATE by segment and prioritize high-CATE segments for future campaigns.

**Case 3: Feature Evaluation at Scale**

A product team wants to estimate the causal effect of a new in-app feature on user retention. The feature was rolled out organically (not A/B tested), and usage correlates with many behavioral signals. You have 1 million users and 100+ features. Walk through your DML implementation decisions: choice of $K$, nuisance models, outcome definition, and how you would report results to stakeholders.

*Key points to hit*: with $n=1M$, use $K=5$ or $K=10$ folds (more folds = less variance in nuisance models). Use gradient boosting or neural networks for nuisance (handle scale). Define outcome carefully (7-day retention? 30-day? LTV?). Report ATE with confidence intervals, and segment CATE by user cohort for actionability. Communicate the key assumption (no unmeasured confounders) clearly to stakeholders.
