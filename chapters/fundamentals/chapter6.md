# Chapter 6: Linear Regression for Causal Inference

Linear regression is one of the most widely used tools in causal inference — not because it's fancy, but because it directly implements the adjustment formula from Chapter 2 under linearity assumptions. When relationships are linear and you've measured all confounders, the coefficient on your treatment variable in a multiple regression *is* the Average Treatment Effect (ATE). This chapter builds the conceptual and mathematical scaffolding: why regression works causally, the Frisch-Waugh-Lovell theorem that explains the mechanics, the dangerous "bad controls" problem, how to estimate heterogeneous effects with interactions, and how to compute valid standard errors in practice.

---

## Regression as Causal Adjustment

### The Link to Chapter 2

Chapter 2 introduced the adjustment formula. For a treatment $T$, outcome $Y$, and confounders $X$:

$$E[Y \mid do(T=t)] = \sum_x E[Y \mid T=t, X=x] \cdot P(X=x)$$

When the data-generating process is linear, this formula reduces to something beautiful: an OLS regression with the confounders included.

Suppose the true SCM is:

$$Y := \tau T + \beta X + \varepsilon$$

where $X$ is a confounder (it causes both $T$ and $Y$) and $\varepsilon \perp T, X$. If we naively regress $Y$ on $T$ alone, the omitted-variable $X$ creates bias because $X$ is correlated with $T$.

But if we regress $Y$ on both $T$ and $X$:

$$\hat{Y} = \hat{\alpha} + \hat{\tau} T + \hat{\beta} X$$

then $\hat{\tau}$ is an unbiased estimate of the ATE $\tau$ — the causal effect of $T$ on $Y$, holding $X$ fixed.

### Why This Works: Omitted Variable Bias Formula

The bias from omitting $X$ in a simple regression of $Y$ on $T$ alone is:

$$\text{Bias} = \hat{\beta} \cdot \frac{\text{Cov}(T, X)}{\text{Var}(T)}$$

where $\hat{\beta}$ is the coefficient $X$ would get if included, and $\frac{\text{Cov}(T, X)}{\text{Var}(T)}$ is the regression coefficient of $X$ on $T$ (how much $X$ changes per unit of $T$).

Bias is zero only when:
1. $X$ has no effect on $Y$ ($\hat{\beta} = 0$), or
2. $T$ and $X$ are uncorrelated (as in an RCT)

Otherwise, you need to include $X$.

### Assumptions for Causal Interpretation

Three conditions must hold for OLS to give the causal ATE:

1. **Linearity**: the true relationship is linear (or close enough)
2. **No unmeasured confounders**: all variables that cause both $T$ and $Y$ are in the regression
3. **Overlap (positivity)**: for each value of $X$, we observe both treated and untreated units

Under these, the coefficient on $T$ equals the ATE. This is the regression analogue of the adjustment formula — it's doing the same standardization, just leveraging the linearity to do it in one shot.

### Code: OLS with Controls Recovers the True ATE

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 2000

# Confounder: higher-income users spend more AND are more likely to use the feature
income = np.random.normal(50, 15, n)

# Treatment: probability of feature adoption increases with income (confounded)
p_treat = 1 / (1 + np.exp(-(income - 50) / 15))
treatment = np.random.binomial(1, p_treat, n)

# Outcome: revenue. True ATE = 10 (using the feature generates $10 more revenue)
# Income also independently drives revenue
revenue = 10 * treatment + 2 * income + np.random.normal(0, 20, n)

df = pd.DataFrame({'revenue': revenue, 'treatment': treatment, 'income': income})

# Naive regression (no controls) -- biased because treatment correlates with income
model_naive = smf.ols('revenue ~ treatment', data=df).fit()

# Controlled regression -- unbiased ATE
model_controlled = smf.ols('revenue ~ treatment + income', data=df).fit()

print("True ATE: 10.0")
print(f"Naive estimate (no controls):      {model_naive.params['treatment']:.2f}")
print(f"Controlled estimate (with income): {model_controlled.params['treatment']:.2f}")
```

Output shows the naive estimate is inflated (income makes both treatment and revenue higher), while the controlled estimate recovers ~10.

---

## The Frisch-Waugh-Lovell (FWL) Theorem

### Statement

The **Frisch-Waugh-Lovell (FWL) theorem** is the single most important algebraic result in applied econometrics. It tells you exactly what a regression coefficient *means* mechanically.

**Theorem**: In the regression of $Y$ on $T$ and controls $X$:

$$Y = \alpha + \beta_T T + \beta_X X + \varepsilon$$

the coefficient $\hat{\beta}_T$ is numerically identical to the coefficient from:

1. Regress $Y$ on $X$ → get residuals $\tilde{Y} = Y - \hat{Y}(X)$
2. Regress $T$ on $X$ → get residuals $\tilde{T} = T - \hat{T}(X)$
3. Regress $\tilde{Y}$ on $\tilde{T}$ (with no other controls)

The resulting coefficient from step 3 equals the formula:

$$\hat{\beta}_T = \frac{\text{Cov}(\tilde{T}, \tilde{Y})}{\text{Var}(\tilde{T})}$$

where $\tilde{T} = T - \hat{T}(X)$ is the part of treatment not explained by controls, and $\tilde{Y} = Y - \hat{Y}(X)$ is the part of the outcome not explained by controls.

### Intuition

FWL says: **regression "partials out" the controls**. The coefficient on $T$ is not comparing all variation in $T$ to all variation in $Y$ — it's comparing only the *residual* variation in $T$ (the part that doesn't look like $X$) to the *residual* variation in $Y$.

Think of it this way: you're asking "among units that look the same on $X$, how does $T$ relate to $Y$?" The residuals $\tilde{T}$ and $\tilde{Y}$ are the "within-$X$" variation — they represent comparisons across units that have been made comparable on the controls.

This is why FWL makes the causal logic transparent:
- $\tilde{T}$ = the variation in treatment that is "as good as random" given $X$ (the causal signal)
- $\tilde{Y}$ = the variation in outcome net of the direct effect of $X$
- Their ratio is the causal effect of $T$ on $Y$

### Numerical Demonstration

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(0)
n = 1000

# DGP: years of education on wages, controlling for experience
# Education and experience are correlated (people enter school at different ages)
experience = np.random.uniform(0, 30, n)
education = 12 + 0.3 * (30 - experience) + np.random.normal(0, 2, n)  # more exp = less ed

# True wage equation: each year of education adds $2k, experience adds $0.5k
wages = 2.0 * education + 0.5 * experience + np.random.normal(0, 5, n)

df = pd.DataFrame({'wages': wages, 'education': education, 'experience': experience})

# --- Method 1: Direct multivariate OLS ---
X_full = df[['education', 'experience']].values
y = df['wages'].values
reg_full = LinearRegression().fit(X_full, y)
coef_direct = reg_full.coef_[0]  # coefficient on education

# --- Method 2: FWL (partialling out) ---
# Step 1: regress wages on experience, get residuals
reg_wages_on_exp = LinearRegression().fit(df[['experience']], df['wages'])
wage_resid = df['wages'] - reg_wages_on_exp.predict(df[['experience']])

# Step 2: regress education on experience, get residuals
reg_ed_on_exp = LinearRegression().fit(df[['experience']], df['education'])
ed_resid = df['education'] - reg_ed_on_exp.predict(df[['experience']])

# Step 3: regress wage residuals on education residuals
reg_fwl = LinearRegression().fit(ed_resid.values.reshape(-1, 1), wage_resid.values)
coef_fwl = reg_fwl.coef_[0]

print("=== Frisch-Waugh-Lovell Theorem ===")
print(f"True coefficient on education:    2.00")
print(f"Method 1 (direct OLS):            {coef_direct:.6f}")
print(f"Method 2 (FWL partialling out):   {coef_fwl:.6f}")
print(f"Difference:                       {abs(coef_direct - coef_fwl):.2e}")
```

The two methods give coefficients that are numerically identical (difference on the order of $10^{-12}$, just floating-point rounding).

### Why FWL Matters for Causal Inference

FWL has several practical consequences:

1. **Orthogonalization is equivalent to controlling**: you don't need multivariate regression to "control" for something — you can partial it out first, then do simple regression on the residuals. This is the key idea behind Double Machine Learning (Chapter 8).

2. **Sample size for inference**: the effective sample size for estimating $\beta_T$ depends on the variance of $\tilde{T}$, not $T$. If controls absorb most of the variation in $T$, your estimate is precise — but you also have little independent variation left to work with. If $R^2$ of the $T$ on $X$ regression is near 1, your estimate will be very noisy (multicollinearity).

3. **Interpretation is always "holding $X$ fixed"**: every regression coefficient implicitly compares units that differ in that variable but have the same values of all other included variables.

---

## Partialling Out / Orthogonalization

### The Concept

Partialling out (also called orthogonalization) is FWL applied deliberately as a technique. It's especially useful when:

- You want to visually show the treatment-outcome relationship after removing confounders
- You're building toward Double ML (where the "partialling out" step uses nonlinear ML models)
- You want to understand which covariates the regression coefficient is sensitive to

The workflow is always:
1. Fit a model of $Y$ on controls $X$ → residuals $\tilde{Y}$
2. Fit a model of $T$ on controls $X$ → residuals $\tilde{T}$
3. The effect of $T$ on $Y$ is the slope of $\tilde{Y}$ on $\tilde{T}$

### Example: Education and Wages (Mincer Equation)

The Mincer earnings equation estimates wage returns to education controlling for labor market experience. FWL lets us see the "pure" education effect after removing experience's influence.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

np.random.seed(42)
n = 500

# Simulate Mincer-style data
# People with more experience tend to have less formal education
experience = np.random.uniform(0, 30, n)
education = np.clip(16 - 0.4 * experience + np.random.normal(0, 3, n), 8, 22)
# Log wages (Mincer equation): log(wage) = alpha + beta_educ*educ + beta_exp*exp + ...
log_wage = 1.5 + 0.10 * education + 0.04 * experience + np.random.normal(0, 0.3, n)

df = pd.DataFrame({
    'log_wage': log_wage,
    'education': education,
    'experience': experience
})

# Full OLS
result = smf.ols('log_wage ~ education + experience', data=df).fit()
print("=== Full OLS ===")
print(result.params[['education', 'experience']].round(4))

# FWL: partial out experience
def partial_out(y, X_controls, df):
    """Remove the linear effect of X_controls from y. Returns residuals."""
    reg = LinearRegression().fit(df[X_controls], df[y])
    return df[y] - reg.predict(df[X_controls])

wage_resid = partial_out('log_wage', ['experience'], df)
educ_resid = partial_out('education', ['experience'], df)

# FWL coefficient
fwl_coef = np.cov(educ_resid, wage_resid)[0, 1] / np.var(educ_resid)
print(f"\n=== FWL coefficient on education ===")
print(f"FWL:      {fwl_coef:.4f}")
print(f"Full OLS: {result.params['education']:.4f}")

# Visualization: the "partialled-out" relationship
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Raw relationship (confounded)
axes[0].scatter(df['education'], df['log_wage'], alpha=0.3, s=10)
axes[0].set_xlabel('Education (years)')
axes[0].set_ylabel('Log wage')
axes[0].set_title('Raw: Education vs Wages\n(confounded by experience)')

# Partialled-out relationship (causal)
axes[1].scatter(educ_resid, wage_resid, alpha=0.3, s=10, color='steelblue')
x_line = np.linspace(educ_resid.min(), educ_resid.max(), 100)
axes[1].plot(x_line, fwl_coef * x_line, 'r-', linewidth=2, label=f'slope = {fwl_coef:.3f}')
axes[1].set_xlabel('Education residual (net of experience)')
axes[1].set_ylabel('Log wage residual (net of experience)')
axes[1].set_title('FWL: Education vs Wages\n(experience partialled out)')
axes[1].legend()

plt.tight_layout()
plt.savefig('fwl_partialout.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\nInterpretation: each additional year of education (holding experience fixed)")
print(f"is associated with a {fwl_coef:.1%} increase in wages.")
```

The slope in the right panel is the *causal* slope — variation in education that is uncorrelated with experience — and it matches the multivariate regression coefficient exactly.

---

## The "Bad Controls" Problem in Regression

Adding more controls to a regression is *not* always better. The DAG determines which variables are safe to include. Three categories of bad controls can bias your estimate.

### Bad Control 1: Mediators

If a variable $M$ lies on the causal path $T \to M \to Y$, controlling for it **blocks the very effect you are trying to measure**.

$$T \to M \to Y$$

Example: you want to estimate the effect of "seeing an ad" ($T$) on "purchasing" ($Y$). Between them is "clicking the ad" ($M$), a mediator. If you control for clicks, you only estimate the effect of seeing an ad that *doesn't* result in a click — you've removed the primary channel.

**What happens mechanically**: when you control for $M$, regression compares units with the same $M$ value. But $M$ is caused by $T$, so "same $M$" means "different $T$ → $M$ relationship" — you're not comparing the full effect of $T$ anymore.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(123)
n = 3000

# DGP: Ad exposure -> Click -> Purchase
# True total effect of ad exposure on purchase
ad_exposure = np.random.binomial(1, 0.5, n)
click = 0.3 * ad_exposure + np.random.normal(0, 0.3, n)  # clicks caused by exposure
click_binary = (click > 0.15).astype(int)

# Purchase is caused by both exposure (direct effect=2) AND clicks (click effect=5)
# Total effect of ad = 2 (direct) + 0.3 * 5 (through clicks) = 3.5
purchase_prob = 0.1 + 0.20 * ad_exposure + 0.50 * click_binary
purchase = np.random.binomial(1, np.clip(purchase_prob, 0, 1), n)

df = pd.DataFrame({
    'purchase': purchase,
    'ad_exposure': ad_exposure,
    'click': click_binary
})

# Correct: total effect of ad exposure (do NOT control for click)
model_correct = smf.ols('purchase ~ ad_exposure', data=df).fit()

# Wrong: controlling for click blocks part of the causal path
model_bad = smf.ols('purchase ~ ad_exposure + click', data=df).fit()

print("=== Mediator as Bad Control ===")
print(f"True total effect of ad exposure: ~0.35 (direct + through clicks)")
print(f"Correct (no mediator control):     {model_correct.params['ad_exposure']:.3f}")
print(f"Bad control (controlling clicks):  {model_bad.params['ad_exposure']:.3f}")
print()
print("Controlling for 'click' strips out the click channel,")
print("so we only estimate the residual direct effect -- severely underestimating the total.")
```

**When is controlling for a mediator valid?** Only when you specifically want the **direct effect** of $T$ on $Y$ that does NOT go through $M$ (e.g., "does seeing an ad increase purchase *independent* of clicking?"). This is a different, more specific research question — and you must be explicit about it.

### Bad Control 2: Colliders

A **collider** $C$ is caused by both $T$ and $Y$ (or by both $T$ and something that causes $Y$):

$$T \to C \leftarrow Y$$

Conditioning on $C$ **opens** a spurious association between $T$ and $Y$. This was covered in Chapter 2. The practical example: if you restrict your analysis to "users who clicked" (the collider), you induce a spurious correlation between ad quality and product quality among clickers — because both cause clicking.

### Bad Control 3: M-Bias (Collider on a Back-door Path)

M-bias is subtler. Consider:

```
X1 → T
X1 → C ← X2
X2 → Y
```

Here, $C$ looks like it could be a confounder (it's associated with both $T$ and $Y$ through $X_1$ and $X_2$). But $C$ is actually a **collider** on the path $T \leftarrow X_1 \to C \leftarrow X_2 \to Y$. That path is **already blocked** because $C$ is a collider and we haven't conditioned on it. If we control for $C$, we open that spurious path and **introduce bias where none existed**.

The graph forms an "M" shape if you draw it:

```
X1    X2
 \   /
  \ /
   C
   |
   (blocked)
```

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(7)
n = 10000

# M-bias simulation
# X1 -> T (direct cause of treatment)
# X2 -> Y (direct cause of outcome)
# X1 -> C <- X2 (C is a collider -- caused by both X1 and X2)
# T has NO direct effect on Y (true causal effect = 0)

X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)

# C is caused by X1 and X2
C = 0.7 * X1 + 0.7 * X2 + np.random.normal(0, 0.5, n)

# T is caused by X1 only
T = 0.5 * X1 + np.random.normal(0, 1, n)

# Y is caused by X2 only -- true effect of T on Y is ZERO
Y = 0.8 * X2 + np.random.normal(0, 1, n)

df = pd.DataFrame({'Y': Y, 'T': T, 'C': C, 'X1': X1, 'X2': X2})

# Correct: do not control for C (no controls needed since T and Y share no common cause)
model_correct = smf.ols('Y ~ T', data=df).fit()

# Bad: controlling for C (the collider) opens the X1-X2 back-door path
model_bad = smf.ols('Y ~ T + C', data=df).fit()

print("=== M-Bias: Controlling for a Collider ===")
print(f"True causal effect of T on Y:      0.000")
print(f"Correct (no C control):             {model_correct.params['T']:.4f}")
print(f"Bad control (controlling for C):    {model_bad.params['T']:.4f}")
print()
print("Controlling for C creates a spurious T->Y association.")
print("X1 and X2 become correlated within levels of C (explaining away C).")
print("This leaks into T-Y correlation even though there is none.")
```

### Summary: Control Variable Decision Rules

| Variable type | Relationship | Should you control? |
|---|---|---|
| Confounder | Causes both T and Y | Yes — always |
| Pure predictor of Y | Affects Y, not T | Yes — reduces variance, no bias |
| Mediator | T → M → Y | Only if you want direct effect |
| Collider | T → C ← Y (or something related) | No — opens spurious paths |
| M-bias collider | On a back-door path, already blocked | No — introduces bias |
| Post-treatment variable | Caused by T | Generally no |

**The rule of thumb**: always draw the DAG first. Never add controls because "it can't hurt to control for more things." It can and does hurt.

---

## Heterogeneous Treatment Effects in Regression

### Why Effects Vary

The ATE is an average — it may mask substantial variation. A pricing discount might work well for price-sensitive customers and not at all for brand loyalists. Interaction terms let you model this **Conditional Average Treatment Effect (CATE)**.

### The Interaction Model

Extend the standard regression to include a treatment-covariate interaction:

$$Y = \alpha + \beta_1 T + \beta_2 X + \beta_3 (T \times X) + \varepsilon$$

The treatment effect for a unit with covariate value $X = x$ is:

$$\text{CATE}(x) = \frac{\partial E[Y]}{\partial T} = \beta_1 + \beta_3 x$$

- $\beta_1$ = treatment effect when $X = 0$ (the "baseline" effect)
- $\beta_3$ = how much the treatment effect *changes* per unit increase in $X$
- The full ATE = $\beta_1 + \beta_3 \cdot E[X]$

### Example: Pricing Discount and Purchase Rate by Customer Segment

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

np.random.seed(2024)
n = 2000

# Customer price sensitivity score (0=not sensitive, 1=very sensitive)
price_sensitivity = np.random.uniform(0, 1, n)

# Treatment: 10% discount (randomly assigned)
discount = np.random.binomial(1, 0.5, n)

# True CATE: effect of discount = 0.05 + 0.30 * price_sensitivity
# Price-sensitive customers respond much more to discounts
true_cate = 0.05 + 0.30 * price_sensitivity

# Purchase probability
purchase_prob = 0.20 + true_cate * discount + 0.10 * price_sensitivity
purchase = np.random.binomial(1, np.clip(purchase_prob, 0, 1), n)

df = pd.DataFrame({
    'purchase': purchase,
    'discount': discount,
    'price_sensitivity': price_sensitivity
})

# Interaction model
model = smf.ols('purchase ~ discount * price_sensitivity', data=df).fit()
print(model.summary().tables[1])

# Extract coefficients
beta1 = model.params['discount']               # effect at price_sensitivity = 0
beta3 = model.params['discount:price_sensitivity']  # interaction term

print(f"\nTreatment effect at price_sensitivity=0:    {beta1:.3f} (true: 0.050)")
print(f"Interaction term:                             {beta3:.3f} (true: 0.300)")
print(f"ATE (avg price sensitivity = {price_sensitivity.mean():.2f}):   "
      f"{beta1 + beta3 * price_sensitivity.mean():.3f} "
      f"(true: {0.05 + 0.30 * price_sensitivity.mean():.3f})")

# Plot: CATE as a function of price sensitivity
sensitivity_range = np.linspace(0, 1, 100)
estimated_cate = beta1 + beta3 * sensitivity_range
true_cate_line = 0.05 + 0.30 * sensitivity_range

plt.figure(figsize=(8, 5))
plt.plot(sensitivity_range, estimated_cate, 'b-', linewidth=2, label='Estimated CATE')
plt.plot(sensitivity_range, true_cate_line, 'r--', linewidth=2, label='True CATE')
plt.fill_between(sensitivity_range,
                 estimated_cate - 1.96 * model.bse['discount:price_sensitivity'] * sensitivity_range,
                 estimated_cate + 1.96 * model.bse['discount:price_sensitivity'] * sensitivity_range,
                 alpha=0.2, label='95% CI (approx)')
plt.xlabel('Price Sensitivity Score')
plt.ylabel('Treatment Effect (CATE)')
plt.title('Heterogeneous Effect of Discount on Purchase Rate')
plt.legend()
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.tight_layout()
plt.savefig('cate_interaction.png', dpi=100, bbox_inches='tight')
plt.show()
```

### Multiple Interactions and Subgroup Analysis

When you have a categorical moderator (e.g., customer segment), use dummy variables:

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(99)
n = 3000

# Three customer segments with different treatment responses
segment = np.random.choice(['budget', 'mid', 'premium'], size=n)
treatment = np.random.binomial(1, 0.5, n)

# True effects: budget=0.15, mid=0.08, premium=0.02
true_effect = {'budget': 0.15, 'mid': 0.08, 'premium': 0.02}
base_rate = {'budget': 0.20, 'mid': 0.35, 'premium': 0.50}

purchase_prob = np.array([base_rate[s] + true_effect[s] * t
                          for s, t in zip(segment, treatment)])
purchase = np.random.binomial(1, np.clip(purchase_prob, 0, 1), n)

df = pd.DataFrame({'purchase': purchase, 'treatment': treatment, 'segment': segment})

# Model with segment interactions (budget is the reference)
model = smf.ols('purchase ~ treatment * C(segment, Treatment("budget"))', data=df).fit()

# Extract CATEs per segment
from itertools import product
print("=== CATEs by Segment ===")
print(f"{'Segment':<10} {'Estimated CATE':>15} {'True CATE':>12}")
for seg, te in true_effect.items():
    if seg == 'budget':
        cate = model.params['treatment']
    else:
        cate = (model.params['treatment'] +
                model.params.get(f'treatment:C(segment, Treatment("budget"))[T.{seg}]', 0))
    print(f"{seg:<10} {cate:>15.3f} {te:>12.3f}")
```

---

## Standard Errors and Inference

### OLS Standard Errors Assume Homoskedasticity

Classical OLS standard errors assume:

$$\text{Var}(\varepsilon_i) = \sigma^2 \quad \forall i \quad \text{(constant variance)}$$

This is rarely true in practice. If the error variance depends on covariates (heteroskedasticity), the OLS SEs are wrong — usually too small, leading to spurious statistical significance.

**Heteroskedasticity is the norm, not the exception** in business data:
- Larger markets have more variable outcomes
- High-revenue customers have more variable spend
- Seasonal effects create systematic variance changes

### Robust (Heteroskedasticity-Consistent) Standard Errors

The **HC3** robust sandwich estimator (White, 1980) is valid under arbitrary heteroskedasticity:

$$\widehat{\text{Var}}(\hat{\beta})_{\text{HC3}} = (X'X)^{-1} \left(\sum_i \frac{x_i x_i' \hat{\varepsilon}_i^2}{(1-h_{ii})^2}\right) (X'X)^{-1}$$

where $h_{ii}$ is the $i$-th diagonal element of the hat matrix $H = X(X'X)^{-1}X'$. The $(1-h_{ii})^2$ correction makes HC3 less biased than HC0 in small samples.

**Practical rule**: always use robust SEs unless you have a specific reason to believe homoskedasticity holds.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 1000

# Heteroskedastic DGP: variance grows with ad_spend
ad_spend = np.random.uniform(1, 100, n)
treatment = np.random.binomial(1, 0.5, n)

# Error variance scales with ad_spend (heteroskedastic)
error_std = 2 + 0.3 * ad_spend
sales = 5.0 * treatment + 0.8 * ad_spend + np.random.normal(0, error_std, n)

df = pd.DataFrame({'sales': sales, 'treatment': treatment, 'ad_spend': ad_spend})

model = smf.ols('sales ~ treatment + ad_spend', data=df).fit()

# Standard OLS SEs (assume homoskedasticity)
result_ols = model
# Robust SEs (HC3)
result_robust = model.get_robustcov_results(cov_type='HC3')

print("=== Standard Errors Comparison ===")
print(f"{'':20} {'OLS SE':>10} {'Robust HC3 SE':>15} {'Ratio':>8}")
for var in ['treatment', 'ad_spend']:
    ols_se = result_ols.bse[var]
    rob_se = result_robust.bse[var]
    print(f"{var:20} {ols_se:10.4f} {rob_se:15.4f} {rob_se/ols_se:8.3f}")

print(f"\nTreatment coefficient: {model.params['treatment']:.3f} (true: 5.0)")
print(f"\nOLS p-value:    {result_ols.pvalues['treatment']:.4f}")
print(f"Robust p-value: {result_robust.pvalues['treatment']:.4f}")
```

### Clustered Standard Errors

When observations within a group (cluster) are correlated, even robust HC SEs are wrong. Classic cases:

- **Users within markets**: marketing experiments often assign treatment at the market level, but you have user-level data. Users within a market share the same environment.
- **Students within schools**: educational interventions.
- **Transactions within customer**: the same customer's purchases are correlated.

Clustered SEs account for within-cluster correlation without assuming a specific correlation structure. The formula:

$$\widehat{\text{Var}}(\hat{\beta})_{\text{cluster}} = (X'X)^{-1} \left(\sum_{g=1}^{G} X_g' \hat{\varepsilon}_g \hat{\varepsilon}_g' X_g\right) (X'X)^{-1}$$

where $g$ indexes clusters, and $X_g$, $\hat{\varepsilon}_g$ are the design matrix and residuals for cluster $g$.

**Rule of thumb**: cluster at the level of treatment assignment. If you randomized at the market level, cluster at the market level — regardless of the unit of observation.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(2024)
n_markets = 40
users_per_market = 250
n = n_markets * users_per_market

# Market-level treatment assignment (randomized at market level)
market_id = np.repeat(np.arange(n_markets), users_per_market)
market_treatment = np.repeat(np.random.binomial(1, 0.5, n_markets), users_per_market)

# Market-level random effect (creates within-cluster correlation)
market_effect = np.repeat(np.random.normal(0, 5, n_markets), users_per_market)

# User-level ad spend (varies by user)
ad_spend = np.random.uniform(0, 50, n)

# True treatment effect: $8 per user
sales = 8.0 * market_treatment + 0.5 * ad_spend + market_effect + np.random.normal(0, 3, n)

df = pd.DataFrame({
    'sales': sales,
    'treatment': market_treatment,
    'ad_spend': ad_spend,
    'market': market_id
})

# Fit base model
model = smf.ols('sales ~ treatment + ad_spend', data=df).fit()

# Different SE specifications
result_ols = model
result_hc3 = model.get_robustcov_results(cov_type='HC3')
result_cluster = model.get_robustcov_results(
    cov_type='cluster',
    groups=df['market']
)

print("=== SE Comparison: Market-Level Clustering ===")
print(f"{'SE type':<20} {'Treatment coef':>15} {'SE':>10} {'t-stat':>8} {'p-value':>10}")

for name, res in [('OLS (wrong)', result_ols),
                   ('HC3 robust', result_hc3),
                   ('Clustered (correct)', result_cluster)]:
    coef = res.params['treatment']
    se = res.bse['treatment']
    tstat = res.tvalues['treatment']
    pval = res.pvalues['treatment']
    print(f"{name:<20} {coef:>15.3f} {se:>10.3f} {tstat:>8.2f} {pval:>10.4f}")

print(f"\nTrue effect: 8.0")
print(f"\nNote: OLS SE is too small (ignores market-level correlation).")
print(f"Clustered SE is larger -- reflecting the true uncertainty.")
print(f"Number of clusters: {n_markets} (should be >30 for cluster SEs to be reliable)")
```

### How Many Clusters Do You Need?

Clustered SEs are asymptotic in the number of clusters $G$, not the number of observations $n$. Rules of thumb:
- $G \geq 30$: clustered SEs generally reliable
- $G < 30$: consider bootstrap (wild bootstrap is the preferred method), or use the bias-corrected "CR2" estimator
- $G < 10$: be very cautious; consider randomization inference instead

---

## When Linear Regression Fails

### Binary Outcomes: LPM vs Logit

When $Y \in \{0, 1\}$, OLS fits a **Linear Probability Model (LPM)**:

$$P(Y = 1 \mid T, X) = \alpha + \beta_T T + \beta_X X$$

The LPM has a bad reputation because it can predict probabilities outside $[0, 1]$ and has structural heteroskedasticity. But for causal inference of ATE/ATT, **LPM often works fine**:

- The coefficient $\beta_T$ is a valid estimate of the average causal effect on the probability scale
- It's directly interpretable: $\beta_T = 0.05$ means treatment increases the probability of outcome by 5 percentage points
- Robust SEs fix the heteroskedasticity problem

Logit/probit is better for **prediction** (respects the $[0,1]$ constraint) but the coefficients are not directly interpretable as causal effects — you need to compute marginal effects, and the ATE involves averaging over a nonlinear function.

**Practical recommendation**: use LPM with robust SEs for causal inference of binary outcomes. Use logit when you need good predicted probabilities (e.g., for IPW).

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(55)
n = 5000

# Binary outcome: did the user convert?
age = np.random.normal(35, 10, n)
treatment = np.random.binomial(1, 0.5, n)

# True effect: treatment increases conversion by 8 percentage points (constant)
log_odds = -2 + 0.4 * treatment + 0.03 * age
prob = 1 / (1 + np.exp(-log_odds))
conversion = np.random.binomial(1, prob, n)

df = pd.DataFrame({'conversion': conversion, 'treatment': treatment, 'age': age})

# LPM
lpm = smf.ols('conversion ~ treatment + age', data=df).fit(
    cov_type='HC3'  # robust SEs
)

# Logit
logit = smf.logit('conversion ~ treatment + age', data=df).fit(disp=False)
# Average marginal effect
ame = logit.get_margeff()

print("=== LPM vs Logit for Binary Outcome ===")
print(f"True ATE on probability scale: ~0.080 (varies slightly due to nonlinearity)")
print(f"\nLPM coefficient on treatment:  {lpm.params['treatment']:.4f}")
print(f"Logit AME on treatment:        {ame.margeff[0]:.4f}")
print(f"\nFor causal ATE: LPM gives a direct, interpretable estimate.")
print(f"Logit coefficient ({logit.params['treatment']:.3f}) is on log-odds scale -- needs conversion.")
```

### Nonlinear Treatment Effects and High Dimensions

Linear regression assumes:
1. The treatment effect is **constant** (or at most linear in covariates via interactions)
2. The confounders' effects are **linear**
3. The number of confounders is **small relative to sample size**

When these fail:
- **Nonlinear heterogeneous effects**: use causal forests (Chapter 9/10) or Double ML with flexible learners
- **Many confounders** (high-dimensional $X$): regularized regression (LASSO) for confounders can fail because LASSO selects for prediction, not for confounding control. Use Double ML (Chapter 8), which explicitly partial out $T$ and $Y$ using ML
- **Complex interactions among confounders**: same solution — Double ML or causal forests

The key preview for Chapter 8: Double ML is exactly the FWL theorem, but with nonparametric ML models doing the partialling-out step instead of linear regression. The final step (regressing $\tilde{Y}$ on $\tilde{T}$) stays linear, ensuring a valid causal estimate with valid standard errors.

| Situation | Approach |
|---|---|
| Linear effects, few controls | OLS with controls |
| Binary outcome | LPM with robust SEs |
| Heterogeneous effects | Interaction terms (linear) or causal forests |
| Many controls (p >> 30) | Double ML (Chapter 8) |
| Nonlinear confounders | Double ML with flexible learners |
| Unknown confounders | IV (Chapter 9), RDD (Chapter 11) |

---

## Interview Questions

### Technical Q&A

**Q1: What conditions are required for OLS to give an unbiased estimate of the ATE?**

A: Three conditions: (1) **No unmeasured confounders** — all variables that cause both treatment and outcome must be included as controls. (2) **Linearity** — the true relationship between the outcome, treatment, and controls must be linear (or well-approximated by a linear function). (3) **Overlap/positivity** — for every combination of control values, we must observe both treated and untreated units. If any of these fail, OLS estimates may be biased. The first condition is untestable from data alone and requires domain knowledge or a DAG.

**Q2: Explain the Frisch-Waugh-Lovell theorem. Why does it matter for causal inference?**

A: FWL states that in a regression of $Y$ on $T$ and controls $X$, the coefficient on $T$ equals the coefficient from regressing the residuals of $Y$ on $X$ against the residuals of $T$ on $X$. Mechanically: $\hat{\beta}_T = \text{Cov}(\tilde{T}, \tilde{Y}) / \text{Var}(\tilde{T})$, where $\tilde{T}$ and $\tilde{Y}$ are the parts of $T$ and $Y$ unexplained by $X$. The causal implication is that regression "controls for" $X$ by comparing variation in $T$ and $Y$ that is *orthogonal to* $X$ — i.e., variation that cannot be attributed to differences in $X$. This also motivates Double ML: you can do the partialling-out with any flexible ML model and still get a valid causal estimate in the final regression step.

**Q3: What is the "bad controls" problem? Give three types of bad controls.**

A: Bad controls are variables that, when added to a regression, bias the treatment effect estimate rather than reducing bias. Three types: (1) **Mediators** — variables on the causal path $T \to M \to Y$. Controlling blocks the effect you want to measure; you'd only get the direct effect. (2) **Colliders** — variables caused by both $T$ and $Y$ (or their causes). Conditioning on a collider opens a spurious association. (3) **M-bias colliders** — variables on a back-door path that is already blocked. Conditioning on such a variable opens the blocked path and introduces bias where none existed. The principle: always determine whether a control variable is a confounder, mediator, or collider *before* adding it, using a DAG.

**Q4: When should you use clustered standard errors, and how do you choose the clustering level?**

A: Use clustered SEs when observations within a group share common unobserved factors — which makes their error terms correlated. The key rule: **cluster at the level of treatment assignment**. If treatment was randomized at the market level (but you observe user-level data), cluster at the market level. The reason: treatment is perfectly correlated within a market, so market-level shocks that affect outcomes create a dependency the standard error must account for. Using OLS or HC3 SEs in this setting understates uncertainty because they treat each user as an independent draw, ignoring the fact that all users in a market got the same treatment assignment.

**Q5: Why can the Linear Probability Model (LPM) be preferred over logit for estimating causal effects?**

A: For causal inference, the goal is usually to estimate the ATE — the average effect of treatment on the probability of the outcome. LPM gives this directly as the coefficient on treatment, with a natural "percentage point" interpretation. Logit gives coefficients on the log-odds scale; to get the ATE you must compute average marginal effects, averaging a nonlinear function over the data. For treatment effect estimation (not prediction), LPM with robust SEs is usually fine: the predicted probabilities outside $[0,1]$ are irrelevant since we care about the coefficient, not individual predictions, and the heteroskedasticity is handled by robust SEs. Logit is preferred when you need well-calibrated predicted probabilities (e.g., for propensity score estimation).

### Case Study Questions

**Case 1**: Your team runs an A/B test to measure the impact of a new recommendation feature on purchase rate. The feature is rolled out to 50% of users randomly. You have user-level data with features like: account age, past purchase count, device type, and whether the user clicked on a recommendation (which only exists for treated users). You're asked to run a regression controlling for all available features to "increase precision." What would you warn about?

*Key issues*: "Whether the user clicked on a recommendation" is only defined for treated users (it's both a post-treatment variable and a mediator — the recommendation leads to clicks which lead to purchases). Including it as a control would: (a) introduce selection bias (can't observe it for control users), and (b) block the mediating path if imputed. The other covariates (account age, past purchases, device) are pre-treatment and safe to include — they'll reduce residual variance and improve precision. You should run two models: (1) unadjusted for the causal estimate, (2) adjusted for pre-treatment covariates for a more precise estimate. Make sure SEs are robust (and consider clustering if there's market-level structure).

**Case 2**: You want to estimate the effect of ad spend on sales. You have data on 500 cities over 12 months. Treatment (ad spend) varies by city and month. You run OLS and get a positive coefficient. How would you make the standard errors credible, and what potential biases would you worry about?

*Key points*: Cluster SEs at the city level (treatment varies at city level; city-level shocks create within-city correlation across months). Alternatively, cluster at city × week if shocks are temporal. Bias concerns: (1) omitted variable bias — cities with higher sales potential may receive more ad spend (confounding); (2) reverse causality — high sales months may trigger more spend (simultaneity); (3) spillovers — ad spend in one city may affect neighboring cities (SUTVA violation). Solutions: include city and time fixed effects (absorbs time-invariant city confounders and common time trends), use lagged ad spend, or run a designed experiment.

**Case 3**: You're estimating the effect of a pricing experiment (10% discount) on conversion. You have customer segments (budget, mid-tier, premium). Your stakeholder asks: "Does the discount work equally well across segments?" How do you model this, and how do you test it?

*Solution*: Include interaction terms between discount and segment dummies. The CATE for each segment is $\beta_{\text{discount}} + \beta_{\text{discount} \times \text{segment}}$. To test whether effects are heterogeneous, run an F-test (or likelihood ratio test) comparing the model with interactions to the model without — the null hypothesis is that all interaction coefficients are jointly zero. Present CATEs with confidence intervals for each segment. A practical implication: if the budget segment has a much higher CATE, you can improve ROI by targeting the discount at that segment only.

**Case 4**: A colleague runs a regression of employee salary on years of experience, controlling for "current job title." They find that experience has a near-zero effect. Is this a valid finding?

*This is the mediator problem*: job title is largely determined by years of experience (more experience → senior title → higher salary). Job title sits on the causal path $\text{experience} \to \text{title} \to \text{salary}$. Controlling for title blocks the primary channel through which experience affects salary. The near-zero coefficient reflects only the direct effect of experience on salary *holding title fixed* — i.e., does seniority within the same title pay more? That's a different, narrower question. The total effect of experience on salary requires a model that does *not* control for title.
