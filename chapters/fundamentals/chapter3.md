# Chapter 3: Applying Causal Inference

Chapters 1 and 2 gave you the foundations — causality vs. correlation, potential outcomes, SCMs, DAGs, and the adjustment formula. This chapter is about *applying* all of it. We cover the end-to-end workflow for a real causal inference problem, the Python tools that support it, how causal models make predictions under intervention, and when to use a causal model vs. a predictive ML model.

---

## Applying Causal Inference: The End-to-End Workflow

A causal inference project follows a distinct rhythm from a standard ML project. The table below contrasts them:

| Stage | ML Project | Causal Inference Project |
|-------|-----------|--------------------------|
| **Problem definition** | Predict $\hat{Y}$ from $X$ | Estimate effect of $T$ on $Y$, adjusting for $X$ |
| **Data exploration** | Feature distributions, correlations | Treatment variation, overlap, confounder balance |
| **Modeling** | Minimize prediction error | Identify and estimate causal estimand |
| **Validation** | Hold-out RMSE, AUC | Placebo tests, refutation checks, sensitivity analysis |
| **Output** | Predictions | Treatment effect estimates + uncertainty |
| **Action** | Rank/score/recommend | Decide on policy/intervention |

The causal inference workflow in practice:

```
Define causal question
       ↓
Build DAG (list variables, draw edges, state assumptions)
       ↓
Check identification (is the effect estimable from data?)
       ↓
Check data requirements (overlap, positivity, sample size)
       ↓
Estimate (adjustment, matching, IPW, DiD, IV, ...)
       ↓
Validate (refutation tests, sensitivity analysis)
       ↓
Communicate (effect size, confidence interval, assumptions)
```

---

## Steps to Formulate Your Problem Using Graphs

This section takes the 5-step DAG process from Chapter 2 and makes it concrete with real business examples.

### Step 1: Write a Crisp Causal Question

A good causal question has three parts:
- **Treatment** $T$: the thing you're considering intervening on
- **Outcome** $Y$: the thing you care about changing
- **Population**: who you're measuring this for

Examples:

| Bad (vague) | Good (causal) |
|-------------|--------------|
| "Does email marketing work?" | "What is the effect of sending a weekly promotional email on 30-day purchase probability among active users?" |
| "Is income related to health?" | "What is the effect of a $1,000 income increase on self-reported health score?" |
| "Does the new UI help retention?" | "What is the ATE of the redesigned onboarding flow on 7-day retention among new signups?" |

### Step 2: List All Variables in the DGP

For any causal question, brainstorm every variable that plausibly affects $T$, $Y$, or both. Categorize them:

| Role | Definition | Example (email campaign) |
|------|-----------|--------------------------|
| **Treatment** | The intervention | Received promotional email |
| **Outcome** | What you measure | Made a purchase (30 days) |
| **Confounder** | Causes both $T$ and $Y$ | Engagement score, tenure |
| **Mediator** | On the causal path $T \to M \to Y$ | Opened email |
| **Instrument** | Affects $T$ but not $Y$ directly | Day of week email sent |
| **Collider** | Caused by both $T$ and $Y$ | Customer service contact |
| **Noise** | Affects $Y$ but not $T$ | Competitor promotions |

### Step 3: Draw the DAG

Use the variable list to draw edges. For each pair $(A, B)$, ask: *"Could A directly cause B? Does knowing A help predict B, over and above everything else in the graph?"*

**Example: Evaluating a Discount Campaign**

Variables: discount offered ($D$), customer tier ($C$), opened email ($O$), purchase ($P$), prior purchase history ($H$).

```
H → D → O → P
C → D       ↑
C ──────────┘
H ──────────┘
```

Interpretation:
- $H$ (history) and $C$ (tier) both influence whether we send a discount and whether the customer buys
- $O$ (email open) is a **mediator** on the path $D \to O \to P$ — don't control for it if you want the total effect of $D$
- $H$ and $C$ are **confounders** — control for them

### Step 4: Identify the Estimand

Given your DAG, determine whether the causal effect is **identifiable** from observed data, and which formula to use.

Common identification strategies:

| Situation | Strategy |
|-----------|----------|
| All confounders observed | Adjustment formula (Chapter 2) |
| Many confounders, high-dimensional | Propensity score matching/IPW (Chapter 5) |
| Unmeasured confounders, valid instrument | Instrumental variables (Chapter 9) |
| Panel data, stable confounders | Difference-in-Differences (Chapter 10) |
| One treated unit | Synthetic control (Chapter 11) |

### Step 5: Validate Assumptions

Before estimating, check:

- **Overlap/Positivity**: Are there treated and untreated units across all covariate values? Plot propensity score distributions — if they don't overlap, your estimate relies on extrapolation.
- **No unmeasured confounders**: Discuss with domain experts. This is an untestable assumption — but sensitivity analysis (Chapter 7) can bound how much a hidden confounder would need to be to change your conclusion.
- **SUTVA**: Does one unit's treatment affect another's outcome? (Network effects, spillovers.)

---

## Simulator Tools

Python has a rich ecosystem for causal inference. Here are the key tools and when to use them.

### Tool Overview

| Library | Strengths | Best For |
|---------|-----------|----------|
| **DoWhy** (Microsoft) | Unified API, built-in refutation tests | End-to-end causal workflow |
| **EconML** (Microsoft) | Heterogeneous treatment effects, DML | CATE estimation, policy |
| **CausalML** (Uber) | Uplift modeling, tree-based CATE | Marketing, targeting |
| **CausalPy** (PyMC) | Bayesian causal inference | DiD, RDD, synthetic control |
| **Statsmodels** | Regression, IV (2SLS) | Classical econometrics |
| **DAGitty** (web) | Visual DAG editor, adjustment set finder | DAG design |

### DoWhy: The End-to-End Causal Workflow

DoWhy follows four explicit steps that match the causal inference workflow: **Model → Identify → Estimate → Refute**.

```python
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel

np.random.seed(42)
n = 2000

# Simulate a confounded observational dataset
# True DGP: discount -> purchase (ATE = 0.10)
# Confounder: customer tenure affects both discount probability and purchase

tenure = np.random.exponential(2, n)           # years as customer
discount = np.random.binomial(1, np.clip(0.3 + 0.1 * tenure, 0, 1), n)
purchase = np.random.binomial(
    1,
    np.clip(0.2 + 0.10 * discount + 0.05 * tenure, 0, 1),
    n
)

df = pd.DataFrame({"discount": discount, "purchase": purchase, "tenure": tenure})

# Step 1: Model — define the causal graph
model = CausalModel(
    data=df,
    treatment="discount",
    outcome="purchase",
    common_causes=["tenure"],         # confounders to adjust for
    graph="""
        digraph {
            tenure -> discount;
            tenure -> purchase;
            discount -> purchase;
        }
    """
)

# Step 2: Identify — what adjustment set is needed?
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Step 3: Estimate — use linear regression adjustment
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
print(f"\nEstimated ATE: {estimate.value:.3f}")   # should be ~0.10

# Step 4: Refute — sanity checks on the estimate
# Random common cause refutation: adding a random confounder shouldn't change estimate much
refute_random = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause"
)
print(refute_random)

# Placebo treatment refutation: replacing treatment with random noise → effect should vanish
refute_placebo = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute"
)
print(refute_placebo)
```

### Manual Simulation: Building Intuition for SCMs

Before using a library, it's valuable to simulate directly from an SCM. This lets you verify that your estimation method recovers the truth:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def simulate_scm(n=5000, seed=0):
    """
    SCM for a job training program:
    - Age (U): affects both selection into program and earnings
    - Program (T): training program participation
    - Earnings (Y): monthly earnings after program
    True ATE = 500 (program raises earnings by $500/month)
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(35, 8, n)                                         # exogenous
    # Selection: older workers less likely to join (administrative reasons)
    p_treat = 1 / (1 + np.exp(0.1 * (age - 35)))
    treatment = rng.binomial(1, p_treat)
    # Outcome: earnings driven by age (experience) + treatment effect
    earnings = 3000 + 50 * (age - 35) + 500 * treatment + rng.normal(0, 300, n)

    return pd.DataFrame({"age": age, "treatment": treatment, "earnings": earnings})

df = simulate_scm()

# Naive estimate (ignores confounding)
naive_ate = df[df.treatment==1]["earnings"].mean() - df[df.treatment==0]["earnings"].mean()
print(f"Naive ATE: ${naive_ate:.0f}")          # biased — age confounds

# Adjusted estimate (control for age)
model = LinearRegression().fit(df[["treatment", "age"]], df["earnings"])
adjusted_ate = model.coef_[0]
print(f"Adjusted ATE: ${adjusted_ate:.0f}")    # ~$500, the true effect
```

### Using DoWhy for Sensitivity Analysis

A key advantage of DoWhy is built-in **refutation tests** that help you assess whether your estimate is reliable:

```python
# Data subset refutation: estimate shouldn't change much on a random subset
refute_subset = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8
)
print(refute_subset)
```

If your estimate changes dramatically under these refutations, that's a red flag — either the model is fragile or an assumption is violated.

---

## Predicting Outcomes Under Interventions

This is where causal models shine: **counterfactual prediction** — answering "what would happen if we intervened?"

### Three Types of Causal Predictions

| Type | Question | Formula |
|------|----------|---------|
| **ATE** | What's the average effect of T on Y in the population? | $E[Y(1)] - E[Y(0)]$ |
| **CATE** | What's the effect for a specific subgroup with covariates $X$? | $E[Y(1) - Y(0) \mid X = x]$ |
| **ITE** | What's the effect for this specific individual? | $Y_i(1) - Y_i(0)$ (never directly observed) |

### Counterfactual Prediction Workflow

Given a fitted causal model, predicting under an intervention means:

1. Set $T = t$ for all units (or for a specific unit)
2. Use the structural equation to predict $Y$ given the intervention
3. Compare to the prediction under $T = 0$

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(7)
n = 3000

# DGP: email marketing campaign
# Confounders: engagement_score, days_since_purchase
engagement = np.random.uniform(0, 1, n)
days_since = np.random.exponential(30, n)

# Treatment: email sent (biased toward engaged users)
p_email = 1 / (1 + np.exp(-3 * (engagement - 0.5)))
email_sent = np.random.binomial(1, p_email)

# Outcome: purchase probability
# True CATE varies by engagement — high-engagement users respond more
true_effect = 0.05 + 0.10 * engagement   # heterogeneous effect
purchase = np.random.binomial(
    1,
    np.clip(0.1 + true_effect * email_sent + 0.2 * engagement - 0.001 * days_since, 0, 1),
    n
)

df = pd.DataFrame({
    "email_sent": email_sent,
    "engagement": engagement,
    "days_since": days_since,
    "purchase": purchase
})

# Fit outcome model controlling for confounders
model = LinearRegression().fit(
    df[["email_sent", "engagement", "days_since"]],
    df["purchase"]
)

# Counterfactual prediction: predict for every unit under T=1 and T=0
df_treat = df.copy(); df_treat["email_sent"] = 1
df_ctrl  = df.copy(); df_ctrl["email_sent"]  = 0

y_hat_1 = model.predict(df_treat[["email_sent", "engagement", "days_since"]])
y_hat_0 = model.predict(df_ctrl[["email_sent", "engagement", "days_since"]])

# Individual treatment effect estimates (ITEs)
df["ite_estimate"] = y_hat_1 - y_hat_0

print(f"Estimated ATE:  {df['ite_estimate'].mean():.3f}")
print(f"True ATE:       {true_effect.mean():.3f}")

# Targeting: who benefits most from the email?
print("\nTop 5 most responsive users (highest estimated effect):")
print(df.nlargest(5, "ite_estimate")[["engagement", "days_since", "ite_estimate"]])
```

### Policy Simulation

A powerful use case: simulate the effect of a proposed policy *before* running a full experiment.

```python
# Policy question: "If we send emails only to users with engagement > 0.6,
# what's the expected lift in purchases?"

target_group = df[df.engagement > 0.6]
targeted_lift = target_group["ite_estimate"].mean()
n_targeted = len(target_group)

all_lift = df["ite_estimate"].mean()
n_all = len(df)

print(f"Lift if email everyone:           {all_lift:.3f} avg effect, {n_all} users")
print(f"Lift if email high-engagement:    {targeted_lift:.3f} avg effect, {n_targeted} users")
print(f"\nTargeted strategy lifts effect by: {targeted_lift/all_lift - 1:.0%}")
# High-engagement users respond more — targeted sending is more efficient
```

---

## Operating Differently: Causal vs. ML Models

Understanding when to reach for each tool is a critical skill.

### When to Use a Predictive ML Model

- You want to **rank, score, or forecast** — not understand causes
- You're okay with correlational features (e.g., zip code as a proxy for income)
- The distribution at deployment is similar to training
- Examples: fraud detection, demand forecasting, recommendation systems, churn scoring

### When to Use a Causal Model

- You want to **intervene** — change a variable and measure the downstream effect
- You're evaluating a **policy or product decision**
- Features may change under the intervention (distribution shift)
- You need to answer **"what if?"** or **"why?"** questions
- Examples: A/B test analysis, policy evaluation, pricing optimization, attribution

### The Deployment Gap

The most common failure mode: using a predictive model to make causal decisions.

```python
import numpy as np

np.random.seed(1)
n = 5000

# DGP: ad spend -> clicks -> revenue
# True causal effect of ad spend on revenue: +$2 per $1 spend
ad_spend = np.random.uniform(100, 1000, n)
clicks = 0.05 * ad_spend + np.random.normal(0, 5, n)
revenue = 2.0 * ad_spend + np.random.normal(0, 50, n)  # ad_spend directly causes revenue

# A predictive model learns the correlation between clicks and revenue
# (clicks is highly correlated with ad_spend, which causes revenue)
from sklearn.linear_model import LinearRegression

# Naive product manager: "let's predict revenue from clicks and optimize clicks"
pred_model = LinearRegression().fit(
    clicks.reshape(-1, 1),
    revenue
)
print(f"Predictive model: revenue = {pred_model.coef_[0]:.1f} * clicks")
# Model says: increase clicks → more revenue

# The problem: clicks don't cause revenue directly.
# Ad spend causes both. If we "optimize" for clicks by non-ad means
# (e.g., clickbait content), revenue won't follow.

# Causal model correctly attributes to ad_spend
causal_model = LinearRegression().fit(
    ad_spend.reshape(-1, 1),
    revenue
)
print(f"Causal model: revenue = {causal_model.coef_[0]:.1f} * ad_spend")
# ~$2 per $1 spend — the true effect
```

### Decision Framework

```
Is the goal to PREDICT an outcome for a given set of features?
    └─ Yes → Predictive ML (optimize for accuracy)

Is the goal to ESTIMATE the EFFECT of an intervention?
    └─ Yes → Is treatment randomly assigned (A/B test)?
                ├─ Yes → Simple difference in means or regression
                └─ No  → Observational causal inference
                            ├─ All confounders measured? → Adjustment / matching / IPW
                            ├─ Valid instrument? → IV
                            ├─ Panel data? → DiD
                            └─ Single treated unit? → Synthetic control
```

### Side-by-Side Comparison

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

np.random.seed(42)
n = 2000

# DGP: user engagement score confounds both feature adoption and retention
engagement = np.random.normal(0, 1, n)
feature_adopted = np.random.binomial(1, 1/(1+np.exp(-2*engagement)))   # high-eng -> adopts
retention = 0.5 + 0.3 * engagement + 0.05 * feature_adopted + np.random.normal(0, 0.1, n)

df = pd.DataFrame({
    "engagement": engagement,
    "feature_adopted": feature_adopted,
    "retention": retention
})

# --- Predictive ML goal: predict retention ---
pred_features = df[["engagement", "feature_adopted"]]
pred_model = LinearRegression().fit(pred_features, df["retention"])
cv_r2 = cross_val_score(pred_model, pred_features, df["retention"], cv=5).mean()
print(f"Predictive R²: {cv_r2:.3f}")
# High R² — good predictions, but tells us nothing about the effect of feature adoption

# --- Causal goal: estimate effect of feature adoption on retention ---
# Naive (ignores confounding)
naive = df[df.feature_adopted==1]["retention"].mean() - df[df.feature_adopted==0]["retention"].mean()

# Adjusted (controls for engagement confounder)
causal_model = LinearRegression().fit(df[["feature_adopted", "engagement"]], df["retention"])
adjusted = causal_model.coef_[0]

print(f"\nNaive feature effect: {naive:.3f}")    # Inflated by engagement confounding
print(f"Adjusted effect:      {adjusted:.3f}")  # ~0.05, the true effect
```

---

## Meta-Learners: Using ML to Evaluate the Adjustment Formula

The adjustment formula requires computing $E[Y \mid T=t, Z_1, \ldots, Z_p]$ across every combination of confounders. With a small number of discrete confounders you can stratify directly — but in practice you have many, often continuous confounders. **Meta-learners** solve this by using an ML model to estimate the conditional expectation, then plugging that model into the adjustment formula.

### Empirical vs. Data-Generating Distribution

The adjustment formula is defined over the **data-generating distribution** $P$ — the true, infinite-population joint distribution of all variables. In practice we only have a finite dataset, which follows the **empirical distribution** $P_E$: each observed row $\{x^i, y^i, z_1^i, \ldots, z_p^i\}$ has probability $1/n$.

When you apply the adjustment formula to your dataset under $P_E$, the confounder weights $P_E(Z_1 = z_1, \ldots, Z_p = z_p)$ equal $1/n$ for each observed row. The formula simplifies to:

$$P_E(Y = y \mid do(X = x)) = \frac{1}{n} \sum_{i} P_E\!\left(Y = y \mid X = x,\, z_1^i, \ldots, z_p^i\right)$$

For continuous $Y$, this becomes:

$$E[Y \mid do(X = x)] = \frac{1}{n} \sum_{i} E\!\left[Y \mid X = x,\, z_1^i, \ldots, z_p^i\right]$$

**Key insight**: as $n \to \infty$, $P_E$ converges to $P$ — so the empirical adjustment formula converges to the true interventional expectation. The ML model's job is to estimate the inner conditional expectation $E[Y \mid X = x, z^i]$ for each row.

---

### S-Learner ("Single")

The S-learner trains a **single** ML model $f(x, z_1, \ldots, z_p)$ to predict $Y$ from treatment $X$ and all confounders $Z$. The ATE is estimated by evaluating this model at $X=1$ and $X=0$ for every observation and averaging the difference:

$$\widehat{\text{ATE}} = \frac{1}{n} \sum_i \left[ f(1, z_1^i, \ldots, z_p^i) - f(0, z_1^i, \ldots, z_p^i) \right]$$

**The S-learner's failure mode**: when training a model to predict $Y$ from $X$ and $Z$, the training process exploits all correlations — including the correlation between $Z$ and $Y$. Tree-based models (decision trees, random forests, gradient boosting) may route the entire prediction through $Z$ and never split on $X$. This makes $f$ insensitive to $X$: changing $X$ from 0 to 1 produces no change in $f$, so the estimated ATE is exactly 0 — not because there is no effect, but as an artifact of the model-fitting process.

**Variable selection warning**: unlike predictive ML, you cannot use cross-validation to select which confounders $Z$ to include. Cross-validation optimizes prediction accuracy, not causal identification. Dropping a confounder from the model can introduce Simpson's paradox-style bias that cross-validation will never detect. **Confounders are selected based on the DAG — period.**

---

### T-Learner ("Two")

The T-learner fixes the S-learner's X-dropping problem by training **two separate models**: one on the treated units and one on the control units.

1. Fit $f_1$ on the treated subsample $\{(z^i, y^i) : X^i = 1\}$ to predict $E[Y \mid X=1, Z=z]$
2. Fit $f_0$ on the control subsample $\{(z^i, y^i) : X^i = 0\}$ to predict $E[Y \mid X=0, Z=z]$
3. For each observation, predict under both models and average the difference:

$$\widehat{\text{ATE}} = \frac{1}{n} \sum_i \left[ f_1(z_1^i, \ldots, z_p^i) - f_0(z_1^i, \ldots, z_p^i) \right]$$

Because $X$ is no longer a feature in either model, the issue of it being dropped cannot arise. Each model sees only one treatment arm and must explain $Y$ through the confounders alone.

| | S-Learner | T-Learner |
|---|---|---|
| **Models trained** | 1 | 2 |
| **X as a feature** | Yes (risky) | No |
| **X-dropping risk** | Yes — can give ATE = 0 | No |
| **Sample per model** | Full dataset | Half (treated or control only) |

---

### Cross-Fitting

Both S- and T-learners have a subtle overfitting problem: they make predictions on the **same data they were trained on**. Even a perfectly cross-validated model can overfit the ATE estimate when predicting on its own training data — standard CV doesn't catch this because it measures prediction accuracy, not ATE bias.

**Cross-fitting** solves this: split data into D₁ and D₂ (stratified by treatment), train on D₁ and predict on D₂, swap, and average the two ATE estimates. Each observation's contribution is computed by a model that never saw it during training.

Cross-fitting is the foundation of **Double Machine Learning (DML)** — covered in Chapter 8. Full treatment with code: **Chapter 13**.

---

### Code Example: S-Learner vs. T-Learner vs. Cross-Fitting

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

# --- T-Learner ---
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

## Multi-Armed Bandits and Reinforcement Learning

### The Peeking Problem in A/B Tests

A standard A/B test runs for a fixed duration determined upfront by a power calculation. A tempting shortcut: check results mid-experiment and stop early if one arm looks better. This is called **peeking**, and it introduces bias.

Why? Over the course of a four-week experiment, random fluctuations will cause arm A to look better than B at some points and worse at others — even if the true effects are identical. If you stop the moment A pulls ahead, you are conditioning on a random upswing and mistaking noise for signal. The resulting estimate is biased upward in favor of A, and your false positive rate balloons well above the nominal 5%.

**The core rule**: the decision to stop must be made independently of the data, or you must use a sequential testing procedure (e.g., always-valid inference, group sequential tests) that explicitly accounts for multiple looks.

---

### Multi-Armed Bandits

Instead of stopping early, a **multi-armed bandit** takes a different approach: rather than keeping allocation fixed at 50/50 until the end, it dynamically adjusts the proportion of traffic sent to each arm as evidence accumulates — routing more traffic toward the better-performing arm while still exploring the worse arm.

This is a form of **reinforcement learning**: the algorithm learns which arm is better over time and exploits that knowledge, while continuing to explore to avoid locking in prematurely.

**The metric bandits optimize is regret** — the cumulative opportunity cost of not always choosing the best arm:

$$\text{Regret}(T) = \sum_{t=1}^{T} \left[ \mu^* - \mu_{a_t} \right]$$

where $\mu^*$ is the expected reward of the best arm and $\mu_{a_t}$ is the expected reward of the arm chosen at time $t$. A good bandit algorithm keeps regret sublinear in $T$ — it converges toward always choosing the best arm.

### Bandits vs. A/B Tests

| | A/B Test | Multi-Armed Bandit |
|---|---|---|
| **Allocation** | Fixed (e.g., 50/50) throughout | Dynamic — shifts toward better arm |
| **Goal** | Estimate causal effect precisely | Minimize regret (maximize reward) |
| **Sample efficiency** | Less — some traffic always goes to worse arm | More — exploits evidence faster |
| **Causal estimate** | Unbiased (randomization preserved) | Potentially biased (arm assignment becomes correlated with time and outcomes) |
| **When to use** | Need a clean causal estimate | Optimizing reward in production; can tolerate some estimation bias |

**The key tradeoff**: bandits are more sample-efficient and reduce the cost of running the inferior arm, but the dynamic allocation means arm assignment is no longer independent of potential outcomes — violating the randomization assumption that makes A/B test estimates unbiased. If you need a clean causal estimate (e.g., for a business case or policy decision), use a fixed A/B test. If you primarily care about maximizing the metric during the experiment itself, a bandit may be appropriate.

### Causal Inference + Bandits

In some settings, historical bandit data is used to evaluate a new policy — a problem called **off-policy evaluation**. Because the bandit algorithm assigned arms non-randomly (it favored better arms over time), naive estimates of arm performance are confounded: the better arm was shown to users at times when outcomes may have been independently higher.

Causal inference tools — specifically inverse probability weighting (IPW, Chapter 5) — are used to correct for this: re-weight observations by the inverse of the probability that the bandit assigned the observed arm at that time step. This recovers an unbiased estimate of what a different policy would have achieved.

### Code Example: Epsilon-Greedy Bandit vs. Fixed A/B Test

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 2000          # total time steps
true_p = [0.10, 0.15]   # true conversion rates: arm 0 is worse, arm 1 is better

# --- Fixed A/B Test (50/50 allocation) ---
ab_rewards = []
ab_assignments = np.random.binomial(1, 0.5, T)   # fixed random assignment
for t in range(T):
    arm = ab_assignments[t]
    reward = np.random.binomial(1, true_p[arm])
    ab_rewards.append(reward)

ab_total = sum(ab_rewards)

# --- Epsilon-Greedy Bandit ---
epsilon = 0.1    # explore with probability epsilon, exploit otherwise
counts  = [0, 0]
values  = [0.0, 0.0]   # running mean reward per arm
bandit_rewards = []
bandit_arms    = []

for t in range(T):
    if np.random.rand() < epsilon or sum(counts) < 2:
        arm = np.random.randint(2)       # explore
    else:
        arm = int(np.argmax(values))     # exploit

    reward = np.random.binomial(1, true_p[arm])
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]   # incremental mean update
    bandit_rewards.append(reward)
    bandit_arms.append(arm)

bandit_total = sum(bandit_rewards)
pct_arm1 = bandit_arms.count(1) / T

print(f"A/B Test  — total reward: {ab_total}  |  arm 1 share: 50.0%")
print(f"Bandit    — total reward: {bandit_total}  |  arm 1 share: {pct_arm1:.1%}")
print(f"Bandit regret reduction vs A/B: {bandit_total - ab_total} additional conversions")

# Peeking bias demonstration
cumulative_ab = np.cumsum(
    [np.random.binomial(1, true_p[a]) - np.random.binomial(1, true_p[1-a])
     for a in ab_assignments]
)
# Random fluctuations cause A to look better or worse at various interim points
# Stopping at the first "significant" crossing introduces bias
```

---

## Interview Questions

### Technical Questions

**Q1: What is a counterfactual prediction and how is it different from a standard ML prediction?**

A **standard ML prediction** answers: "Given that this unit has features $X$, what do we expect $Y$ to be?"

A **counterfactual prediction** answers: "What would $Y$ be *if we intervened* and set $T$ to some value, holding all else equal?" It requires a causal model — we predict $Y$ under $T=1$ and $T=0$ separately and take the difference. A standard ML model can predict well without supporting this operation if it uses confounded features.

---

**Q2: What is CATE and why is it more useful than ATE in many business applications?**

**CATE** (Conditional Average Treatment Effect) is $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ — the average treatment effect for a subgroup defined by covariates $X = x$.

In business, the ATE tells you the average effect across everyone, but you often care about *who to target*. CATE lets you identify which users respond most to a treatment (e.g., email, discount, feature), enabling **targeted interventions** that improve efficiency — e.g., send discounts only to price-sensitive users, not to those who'd buy anyway.

---

**Q3: What are refutation tests in DoWhy and why do they matter?**

Refutation tests probe whether your causal estimate is robust to violations of assumptions:

- **Random common cause**: Add a random noise variable as a confounder — estimate shouldn't change much.
- **Placebo treatment**: Replace the treatment with a random permutation — effect should be ~zero.
- **Data subset**: Estimate on 80% of data — estimate should be stable.

If any refutation test changes your estimate substantially, it suggests the estimate is sensitive to assumptions or data artifacts — a red flag that warrants investigation.

---

**Q4: When would you use simulation before running a real experiment?**

Simulation (from an SCM) is useful for:
1. **Power analysis** — how large does the sample need to be to detect a given effect size?
2. **Method selection** — does your proposed estimation method (e.g., matching, IPW) recover the true effect under your assumed DGP?
3. **Sensitivity analysis** — how much does the estimate change if an assumption is slightly violated?
4. **Communicating the model** — stakeholders can understand a simulation more easily than abstract formulas.

---

**Q5: What is the difference between the S-learner and T-learner?**

Both are meta-learners — they use ML models to evaluate the adjustment formula when there are many confounders.

- **S-learner**: trains a single model $f(X, Z_1, \ldots, Z_p)$. ATE = average of $f(1, z^i) - f(0, z^i)$ across all rows. Risk: tree-based models may ignore $X$ entirely if $Z$ explains $Y$ well, returning ATE = 0 spuriously.
- **T-learner**: trains two models, $f_1$ on treated units and $f_0$ on control units. ATE = average of $f_1(z^i) - f_0(z^i)$. Since $X$ is not a feature in either model, the X-dropping failure mode is impossible.

---

**Q6: What is cross-fitting and why is it needed for meta-learners?**

S- and T-learners make predictions on the same data they were trained on. This causes in-sample overfitting of the ATE — distinct from prediction overfitting, and not caught by standard cross-validation.

Cross-fitting fixes this: split data into $D_1$ and $D_2$, train on $D_1$ and predict on $D_2$ to get $\text{ATE}_2$, then swap to get $\text{ATE}_1$, and average. Each observation's ATE contribution is computed by a model that never saw that observation during training. Cross-fitting is the foundation of Double Machine Learning (Chapter 8).

---

### Case Study Questions

**Case 1: Your company is deciding whether to build a personalized pricing model or do a pricing experiment. How do you advise?**

- A **predictive pricing model** (ML) can optimize prices to maximize predicted revenue — but it assumes the learned price-demand relationship holds under intervention. If the model exploits confounds (e.g., high-value customers were historically charged less), acting on it will backfire.
- A **pricing experiment** (RCT) establishes the true causal effect of price on purchase probability, giving an unbiased estimate to optimize against.
- Recommendation: run a factorial A/B test on prices across segments first, use the causal estimates to build a decision policy, then validate the policy with a holdout test.

---

**Case 2: A teammate builds a churn model with 90% AUC. The PM wants to use it to target users for a retention campaign. What questions do you ask?**

1. Does the model use features that are causally related to churn, or just predictive proxies?
2. Among the high-churn-risk users the model flags, would the retention treatment actually help? (Treatment effect may be near zero for already-committed churners.)
3. What is the **incremental** effect of the campaign — i.e., what's the CATE, not just the prediction?
4. Consider building an **uplift model** (CATE estimator) that predicts *who will respond to the campaign*, not just who will churn — these are different populations.

---

**Case 3: Explain when policy simulation from observational data can go wrong.**

Policy simulation extrapolates the estimated treatment effect to a new intervention. It can fail when:
1. **Positivity fails**: The intervention affects subgroups where we had no treatment data — estimates rely on model extrapolation.
2. **Model misspecification**: If the outcome model is wrong (wrong functional form, missing interactions), the simulated counterfactuals will be biased.
3. **SUTVA violation**: The simulation assumes no spillover effects — if the policy treats many users simultaneously, interactions between units may change the effect.
4. **Distribution shift**: The policy changes which units are treated, potentially selecting a population different from the training data.

Always pair a simulation with a subsequent experiment to validate the predicted effect.

---

**Case 4: A colleague says "I used cross-validation to select which confounders to include in my causal model — I kept only the ones that improved prediction accuracy." What's the problem?**

This is a fundamental error. Cross-validation optimizes for predictive accuracy on held-out data — it has no ability to detect confounding bias. Dropping a true confounder from the model can introduce Simpson's paradox-style reversal in the causal estimate, yet the cross-validated prediction error may improve (the dropped variable was collinear and added noise to prediction).

In causal inference, variables are included because they are confounders — causes of both treatment and outcome — as determined by the DAG. That decision must be made before any data is touched. Cross-validation cannot substitute for causal reasoning about the data-generating process.

---

**Case 5: Your S-learner returns an ATE of exactly 0, but you believe the treatment has a real effect. What's likely happening and how do you fix it?**

The S-learner's failure mode: a tree-based model trained on $(X, Z_1, \ldots, Z_p)$ may route all its prediction through $Z$ and never split on $X$ — especially when $Z$ is highly predictive of $Y$ and $X$ is correlated with $Z$. The model becomes insensitive to $X$, so evaluating $f(1, z^i) - f(0, z^i)$ returns 0 for every observation.

Fix: switch to a **T-learner** — train separate models $f_1$ and $f_0$ on the treated and control subsamples respectively. Since $X$ is no longer a feature in either model, the issue cannot arise. Add **cross-fitting** to avoid in-sample overfitting of the ATE estimate.

---

**Case 6: Your PM wants to stop an A/B test after one week because the treatment looks significantly better. What do you say?**

This is the **peeking problem** — stopping early when one arm appears better introduces bias. Over the course of the experiment, random fluctuations will cause the treatment to look better at some interim points even if the true effect is zero. Stopping the moment it crosses a significance threshold conditions on a random upswing and inflates the false positive rate well above 5%.

Options: (1) commit to the pre-specified sample size and don't look until the end; (2) use a **sequential testing procedure** (group sequential test or always-valid p-values) that explicitly corrects for multiple looks; or (3) switch to a **multi-armed bandit** if the goal is to minimize regret during the experiment rather than obtain an unbiased causal estimate. If a clean causal estimate is needed for a business decision, hold the line on the fixed test duration.
