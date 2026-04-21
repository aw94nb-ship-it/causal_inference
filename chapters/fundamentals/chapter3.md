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
