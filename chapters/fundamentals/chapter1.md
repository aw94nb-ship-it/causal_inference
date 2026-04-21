# Chapter 1: Introducing Causality

---

## Introducing Causality

### What Is Causality?

**Causality** is a relationship where one event (the *cause*) directly produces a change in another event (the *effect*). This is distinct from mere *association* — the fact that two things tend to occur together.

The cleanest way to express a causal question is with the **do-operator**, introduced by Judea Pearl:

$$P(Y \mid do(X = x))$$

This reads: "What is the probability of $Y$ if we *intervene* and set $X$ to $x$?" — as opposed to:

$$P(Y \mid X = x)$$

which just asks: "Among observations where $X$ happens to equal $x$, what is $Y$?"

The difference is enormous. The first is a causal question. The second is a statistical one.

**Concrete example:** Suppose $X$ = carrying an umbrella, $Y$ = getting wet.

- $P(\text{wet} \mid \text{has umbrella})$ is low — people who carry umbrellas tend to stay dry.
- But $P(\text{wet} \mid do(\text{has umbrella} = \text{True}))$ is about the same as baseline — *forcing* someone to carry an umbrella doesn't change weather or their behavior.

The correlation exists because both umbrella-carrying and dryness are caused by a third thing: checking the forecast.

---

### The Causal Hierarchy (Pearl's Ladder of Causation)

| Level | Question | Example | Tool |
|-------|----------|---------|------|
| **Association** | What is? | "Users who click ads buy more." | Observational data, ML |
| **Intervention** | What if we do X? | "What happens to revenue if we show the ad?" | Experiments, do-calculus |
| **Counterfactual** | What if we had done X differently? | "Would this user have bought if we hadn't shown the ad?" | Structural causal models |

Most ML lives at level 1. Causal inference targets levels 2 and 3.

---

## Contrasts Between Causal and Predictive Models

### The Core Difference

| Dimension | Predictive ML | Causal Inference |
|-----------|--------------|-----------------|
| **Goal** | Minimize prediction error on held-out data | Estimate the effect of an intervention |
| **Key assumption** | i.i.d. data, stable distribution | Correct causal graph / identification strategy |
| **Spurious correlations** | Fine to use if they predict | Must be removed — they break under intervention |
| **Output** | $\hat{Y}$ | $E[Y(1) - Y(0)]$, i.e., the treatment effect |
| **Evaluation** | RMSE, AUC, accuracy | Bias of causal estimate, coverage of CIs |

### When to Use Each — Visual Guide

```
┌─────────────────────────────────┐     ┌──────────────────────────────────────┐
│  Use CAUSALITY when your        │     │  Use MACHINE LEARNING when your      │
│  actions make an IMPACT here    │     │  actions DEPEND on the outcome       │
└────────────────┬────────────────┘     └──────────────────────┬───────────────┘
                 │ (you intervene on X)                        │ (you react to Y)
                 ▼                                             │
        X ──────────────── F ──────────────────────────► Y ───┘
                        (your model)

Causality = Causal Inference + A/B Testing / RCTs
```

The arrow directions are the key:

- **Causal inference**: you *control* $X$ (a treatment, policy, or feature) and want to measure the downstream impact on $Y$. You're acting *on* $X$.
- **Machine learning**: $Y$ is something that will happen regardless (churn, fraud, demand), and you want to predict it so your decisions can *depend on* $Y$. You're reacting *to* $Y$.

**Quick test** — ask yourself: *"Am I changing $X$ to influence $Y$, or am I predicting $Y$ to decide what to do next?"*
- Changing $X$ to influence $Y$ → causal inference
- Predicting $Y$ to inform action → machine learning

### When ML and Causal Inference Coincide — and When They Don't

**When they coincide:** If the future looks like the past — same users, same behavior patterns, same environment — ML works well and cross-validation gives you a reliable estimate of real-world performance. The correlations the model learned still hold, so predictions are accurate and decisions based on them are sound. A churn model trained on last quarter's data will likely work fine next quarter if nothing structurally changed.

**When they diverge:** The moment something *new* happens — a policy change, a product intervention, an economic shock, a new user segment — ML breaks down. The model learned correlations under the old regime, but the new situation changes the relationships between variables. Cross-validation won't catch this because it only tests the model on more data from the same distribution. A causal model, on the other hand, is built around *mechanisms* rather than surface correlations. Mechanisms are more stable: even when the environment shifts, the underlying cause-and-effect relationships tend to hold.

**A concrete example:** An ML model learns that users who receive discount emails have high purchase rates, so it flags "send discount" as a good action. But this correlation exists because discounts were historically sent to already-engaged users. When you roll out discounts broadly (the intervention), the correlation breaks — now you're sending to less engaged users and the lift is much smaller. A causal model, trained to estimate the *effect* of the discount, would have caught this.

### Why a Good Predictive Model Can Fail as a Causal Model

Classic example: **hospital mortality prediction**.

A model trained on hospital data might learn that patients with *fewer* procedures have higher mortality. This is because very sick patients receive many procedures. If a hospital used this model to *reduce* procedures (intervention), outcomes would worsen — the model has the causal arrow backwards.

Another example: **ice cream sales → drowning deaths**. A regression model can predict drownings well using ice cream sales. But a policy that bans ice cream will not reduce drownings.

### The Intervention Gap

A predictive model learns:

$$\hat{Y} = f(X_1, X_2, \ldots, X_p)$$

This is valid as long as the *joint distribution* of features doesn't change. But when you *intervene* on a feature (e.g., change a product's price), you're forcing $P(X_1)$ to change, which may break the relationships the model relied on. This is called **distribution shift under intervention**.

Causal models instead target:

$$E[Y \mid do(T = t)] = E_{X}[E[Y \mid T = t, X]]$$

This formula is the **adjustment formula** — it removes confounding by standardizing over $X$. Here's why each piece matters:

**Why not just compute $E[Y \mid T = t]$?**
Because people who naturally receive $T = t$ are not representative of the full population — they have a different (skewed) distribution of $X$. For example, if sicker patients are more likely to get Treatment A, then $E[Y \mid T=A]$ is dragged down by case severity, not just the treatment effect.

**Inner expectation: $E[Y \mid T = t, X]$**
Fix a specific value of $X$ and estimate the average outcome within that homogeneous subgroup. Within a stratum of $X$, there's no confounding — everyone has the same $X$.

**Outer expectation: $E_X[\cdots]$**
Average those within-stratum estimates over the **population distribution of $X$** — i.e., $P(X)$, not $P(X \mid T = t)$. This is the key move: you re-weight to the full population, not just the people who happened to select into treatment.

Expanded, the formula reads:

$$E[Y \mid do(T=t)] = \sum_x E[Y \mid T=t, X=x] \cdot P(X=x)$$

**One-line intuition:** Estimate the effect *within each type of person* (fixing $X$), then average over *how many of each type exist in the real world*.

This is exactly what a randomized experiment gives you for free — randomization forces $P(X \mid T=t) = P(X)$, so the two distributions are already aligned. When you don't have an experiment, this formula recreates that balance manually.

### Code Example: Where a Predictive Model Fails

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 5000

# True DGP: age (U) causes both treatment and outcome
# Treatment does NOT cause outcome here — purely spurious correlation
age = np.random.normal(50, 10, n)
treatment = (age > 55).astype(int)          # older patients get treatment A
outcome = 80 - 0.5 * age + np.random.normal(0, 5, n)  # outcome driven by age, not treatment

df = pd.DataFrame({"age": age, "treatment": treatment, "outcome": outcome})

# Naive predictive model (ignores confounding)
naive_model = LinearRegression().fit(df[["treatment"]], df["outcome"])
print(f"Naive coefficient on treatment: {naive_model.coef_[0]:.2f}")
# => Negative! Looks like treatment HURTS, but that's just because
#    older (sicker) patients are more likely to receive treatment.

# Correct causal estimate: control for age (the confounder)
causal_model = LinearRegression().fit(df[["treatment", "age"]], df["outcome"])
print(f"Causal coefficient on treatment: {causal_model.coef_[0]:.2f}")
# => ~0: treatment has no effect, which is the truth

# Key insight: the naive model may predict well in-sample
#              but gives the WRONG answer to "what if we intervene?"
```

---

## Experimental Studies

### What Is an Experiment?

An **experiment** (also called a **Randomized Controlled Trial**, or RCT) is the gold standard for causal inference. The key feature: **treatment is assigned randomly**, independent of any characteristic of the unit.

Random assignment breaks the $\text{Confounder} \rightarrow \text{Treatment}$ arrow. This means:

$$E[Y(1)] = E[Y \mid T = 1] \quad \text{and} \quad E[Y(0)] = E[Y \mid T = 0]$$

So the simple **difference in means** is an unbiased estimator of the ATE:

$$\widehat{ATE} = \bar{Y}_{T=1} - \bar{Y}_{T=0}$$

### Why Randomization Works

Without randomization, treated and untreated groups may differ systematically (e.g., sicker patients choose a treatment). With randomization:

$$P(T = 1 \mid X) = 0.5 \quad \forall X$$

This makes treatment independent of all covariates (observed and unobserved):

$$T \perp\!\!\!\perp (Y(0), Y(1))$$

So any difference in outcomes between groups is *caused* by the treatment, not by pre-existing differences.

### A/B Testing: The Tech Industry's RCT

In product analytics, A/B tests are RCTs. Users are randomly assigned to control (A) or treatment (B). Any difference in the metric of interest is causally attributable to the change.

**Example**: Does a new onboarding flow increase 30-day retention?

- **Treatment**: new onboarding flow
- **Control**: existing onboarding flow  
- **Outcome**: 30-day retention (binary)
- **Estimand**: ATE = $E[\text{retain}(1)] - E[\text{retain}(0)]$

### Hypothesis Testing in Experiments

Once we have the ATE estimate, we want to know if it's statistically significant. The standard test:

$$t = \frac{\bar{Y}_1 - \bar{Y}_0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}}$$

Under $H_0: \text{ATE} = 0$, this follows a t-distribution.

### Code Example: Simulating and Analyzing an RCT

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(0)  # fix the random number generator so results are identical every run
n = 2000           # sample size — number of simulated users

# Simulate an A/B test: new onboarding flow
# True effect: +5% retention
age = np.random.normal(30, 8, n)            # covariate (doesn't affect assignment)
treatment = np.random.binomial(1, 0.5, n)  # random assignment

# Potential outcomes
# baseline retention ~40%, treatment adds 5pp, age adds a small effect
baseline = 0.40 + 0.005 * (age - 30)
y0 = np.random.binomial(1, np.clip(baseline, 0, 1))           # control outcome
y1 = np.random.binomial(1, np.clip(baseline + 0.05, 0, 1))   # treated outcome

# Observed outcome (we only see one)
y_obs = treatment * y1 + (1 - treatment) * y0

df = pd.DataFrame({"treatment": treatment, "age": age, "retained": y_obs})

# Estimate ATE: simple difference in means
y1_mean = df[df.treatment == 1]["retained"].mean()
y0_mean = df[df.treatment == 0]["retained"].mean()
ate_hat = y1_mean - y0_mean

# T-test
t_stat, p_value = stats.ttest_ind(
    df[df.treatment == 1]["retained"],
    df[df.treatment == 0]["retained"]
)

print(f"Retention (treatment): {y1_mean:.3f}")
print(f"Retention (control):   {y0_mean:.3f}")
print(f"Estimated ATE:         {ate_hat:.3f}")
print(f"p-value:               {p_value:.4f}")

# Confidence interval
n1 = (df.treatment == 1).sum()
n0 = (df.treatment == 0).sum()
se = np.sqrt(df[df.treatment==1]["retained"].var()/n1 + df[df.treatment==0]["retained"].var()/n0)
ci_low = ate_hat - 1.96 * se
ci_high = ate_hat + 1.96 * se
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
```

### Limitations of Experiments

| Challenge | Description |
|-----------|-------------|
| **Cost** | Running experiments requires infrastructure and time |
| **Ethics** | Can't randomize harmful treatments (e.g., smoking) |
| **Feasibility** | Can't randomize some interventions (e.g., minimum wage policy) |
| **Generalizability** | Lab results may not hold in the real world (external validity) |
| **Network effects** | Users interact — violates SUTVA (stable unit treatment value assumption) |

When experiments aren't feasible, we turn to **observational methods**.

---

## Observational Studies

### What Is an Observational Study?

In an **observational study**, treatment is not randomly assigned — we observe units as they naturally select into treatment. This is the norm in:

- Healthcare data (patients choose or are prescribed treatments)
- Social science (people self-select into programs)
- Business (users self-select into features, customers opt into promotions)
- Economics (policies affect different regions differently)

The core challenge: **selection bias**. Units that select into treatment likely differ from those that don't, in ways that also affect the outcome.

$$E[Y \mid T = 1] - E[Y \mid T = 0] \neq E[Y(1)] - E[Y(0)]$$

The difference between the left side (what we observe) and the right side (what we want) is called **selection bias**:

$$\underbrace{E[Y \mid T=1] - E[Y \mid T=0]}_{\text{observed difference}} = \underbrace{E[Y(1)] - E[Y(0)]}_{\text{ATE}} + \underbrace{E[Y(0) \mid T=1] - E[Y(0) \mid T=0]}_{\text{selection bias}}$$

**Left side** — what you *observe* in data: average outcome among people who got treatment, minus average outcome among people who didn't.

**Right side** — what you *want*: the true causal effect (ATE) — what would happen if you took the same person and switched their treatment.

They're not equal because the two groups are **different kinds of people** to begin with.

#### Breaking Down the Selection Bias Term

The selection bias term $E[Y(0) \mid T=1] - E[Y(0) \mid T=0]$ asks: **"How would the treated group have done, had they not been treated — compared to how the control group actually did?"**

If these differ, the two groups were already different *before* any treatment. That pre-existing gap bleeds into your naive comparison and masquerades as a treatment effect.

**Loyalty program example:**
- Observed difference: loyalty members buy 3x more → looks like a huge effect
- True ATE: maybe 1.1x — the program's actual lift
- Selection bias: 1.9x — they were already heavy buyers before enrolling

#### When Does Selection Bias Vanish?

In a **randomized experiment**. Random assignment makes the treated and control groups identical on average *before* treatment:

$$E[Y(0) \mid T=1] = E[Y(0) \mid T=0]$$

The selection bias term becomes zero, and the observed difference equals the true ATE. This is the core reason RCTs are the gold standard.

### Example: College Education and Earnings

**Naive approach**: Compare earnings of college graduates vs. non-graduates.

**Problem**: People who go to college likely have higher ability, more motivated parents, and better economic circumstances — all of which *independently* raise earnings. So the naive comparison confounds the *effect of college* with the *effect of pre-existing advantages*.

**Selection bias in action**: Even in a world where college had zero causal effect, graduates would still earn more — because the kind of person who goes to college is already different.

### Example: Evaluating a Loyalty Program

Suppose you want to know if your customer loyalty program increases purchase frequency.

- Users who enroll likely **already buy more** (self-selection into the program)
- Naive comparison: loyalty members buy 3x more than non-members
- But: would they have bought 2.5x more anyway?

Without randomization, you need observational methods to estimate the true effect of the program.

### Overview of Observational Methods

| Method | Key Idea | When to Use |
|--------|----------|-------------|
| **Regression adjustment** | Control for confounders in a regression | Observed confounders, linear relationships |
| **Matching** | Find similar untreated units for each treated unit | Want comparability without parametric assumptions |
| **Propensity score methods** | Model probability of treatment, then reweight/match | Many confounders, high-dimensional |
| **Instrumental variables (IV)** | Find a variable that affects treatment but not outcome directly | Unmeasured confounding, valid instrument available |
| **Difference-in-Differences (DiD)** | Compare trends over time across groups | Panel data, parallel trends assumption |
| **Regression Discontinuity (RDD)** | Exploit a threshold cutoff in treatment assignment | Natural cutoff exists |
| **Synthetic Control** | Construct a weighted counterfactual from control units | Single treated unit (e.g., one state, one company) |

### Code Example: Selection Bias in Observational Data

```python
import numpy as np
import pandas as pd

np.random.seed(1)
n = 3000

# True DGP: loyalty program has a small TRUE effect of +1 purchase/month
# But high-frequency buyers are MORE likely to enroll (selection bias)

# Latent "purchase propensity" — higher means naturally buys more
propensity = np.random.normal(0, 1, n)

# Enrollment in loyalty program: biased toward high-propensity buyers
enroll_prob = 1 / (1 + np.exp(-2 * propensity))  # sigmoid
enrolled = np.random.binomial(1, enroll_prob)

# Observed purchases: driven by propensity + small treatment effect
purchases = 5 + 3 * propensity + 1.0 * enrolled + np.random.normal(0, 1, n)

df = pd.DataFrame({
    "enrolled": enrolled,
    "propensity": propensity,
    "purchases": purchases
})

# Naive estimate: compare enrolled vs not enrolled
naive_ate = df[df.enrolled==1]["purchases"].mean() - df[df.enrolled==0]["purchases"].mean()
print(f"Naive ATE estimate: {naive_ate:.2f}")  # Severely overestimates true effect of 1.0!

# Corrected estimate: control for propensity (the confounder)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(df[["enrolled", "propensity"]], df["purchases"])
print(f"Adjusted ATE estimate: {model.coef_[0]:.2f}")  # ~1.0, the true effect
```

---

## Interview Questions

### Technical Questions

**Q1: What is the fundamental problem of causal inference?**

We can never observe both potential outcomes $Y_i(1)$ and $Y_i(0)$ for the same unit at the same time. Once a unit receives treatment, we observe $Y_i(1)$ but can never observe what $Y_i(0)$ would have been (the counterfactual), and vice versa.

---

**Q2: What's the difference between $P(Y \mid X = x)$ and $P(Y \mid do(X = x))$?**

- $P(Y \mid X = x)$ is a conditional probability — it filters the *observed* population to cases where $X = x$.
- $P(Y \mid do(X = x))$ is an interventional distribution — it asks what would happen if we *forcibly set* $X = x$ for all units, breaking any confounding.

They're equal only when $X$ is independent of its causes (e.g., in a randomized experiment).

---

**Q3: Why does randomization eliminate confounding?**

Randomization makes treatment assignment $T$ independent of all baseline covariates (observed and unobserved): $T \perp\!\!\!\perp (Y(0), Y(1))$. This means the treated and control groups are comparable on average, so any difference in outcomes is attributable to treatment.

---

**Q4: Define selection bias and give its formula.**

Selection bias is the difference between the naive observed group difference and the true ATE:

$$\text{Selection bias} = E[Y(0) \mid T=1] - E[Y(0) \mid T=0]$$

It's nonzero when treated and untreated units would have different outcomes even without treatment — i.e., they're *systematically different* before treatment.

---

**Q5: What's the difference between ATE and ATT?**

- **ATE** (Average Treatment Effect): $E[Y(1) - Y(0)]$ — the average effect across *all* units in the population.
- **ATT** (Average Treatment Effect on the Treated): $E[Y(1) - Y(0) \mid T = 1]$ — the average effect among units that *actually received* treatment.

These differ when the treatment effect is heterogeneous and treatment assignment is correlated with the magnitude of the effect (as in self-selection scenarios).

---

### Case Study Questions

**Case 1: Evaluating a new feature's impact**

*"We launched a dark mode feature. 30% of users opted in. Their session duration is 20% higher than users without dark mode. Does dark mode cause longer sessions?"*

**Answer framework:**
- This is an observational study — users self-selected into dark mode.
- Selection bias: power users / engaged users are more likely to turn on dark mode *and* have longer sessions regardless.
- Cannot conclude causality from this comparison alone.
- To estimate the causal effect:
  - **Best**: Run an A/B test where dark mode is randomly enabled for a group
  - **If already shipped**: Use a quasi-experimental method — e.g., DiD (compare change in session duration for early adopters vs. later adopters), or propensity score matching on user characteristics

---

**Case 2: Healthcare — does a drug reduce blood pressure?**

*"In our observational claims data, patients who took Drug A have lower blood pressure than those who didn't. Can we conclude Drug A is effective?"*

**Answer framework:**
- Selection bias: physicians prescribe Drug A to patients with elevated blood pressure — so the treated group may be *sicker* at baseline, biasing the naive estimate downward.
- Or: patients who adhere to medication may be healthier overall, biasing the estimate upward (healthy user bias).
- Approaches: propensity score matching or IPW on age, BMI, comorbidities; or instrumental variables (e.g., prescribing physician preference as an instrument).

---

**Case 3: Policy — does minimum wage increase unemployment?**

*"Economists want to know if raising the minimum wage causes higher unemployment. How would you approach this?"*

**Answer framework:**
- Can't randomize minimum wage across workers.
- Classic approach: **Difference-in-Differences** — compare employment trends in states that raised minimum wage vs. neighboring states that didn't (Card & Krueger 1994).
- Key assumption: parallel trends — the states would have followed similar employment trends absent the policy change.
- Also consider: **Synthetic Control** if only one state changed the law.

---

**Case 4: When can't you run an A/B test?**

*"Name situations where an A/B test is infeasible and what you'd do instead."*

- **Legal/ethical**: Can't randomize pricing by user (price discrimination), can't randomize job offers by race.
- **Network effects**: Showing ads to half a social network — treated users interact with untreated users, violating SUTVA. Use cluster randomization.
- **Already shipped**: Feature rolled out globally before analysis. Use DiD with pre-launch data.
- **Rare events**: Not enough power for small effects. Use longer window, Bayesian methods, or proxy metrics.
- **One-unit treatment**: E.g., a country-level policy. Use synthetic control.
