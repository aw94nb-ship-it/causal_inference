# Chapter 2: Causal Models and the Adjustment Formula

The previous chapter introduced the *idea* of causality. This chapter gives you the machinery to reason about it precisely: **Structural Causal Models (SCMs)**, **Directed Acyclic Graphs (DAGs)**, and the **adjustment formula** — the workhorse technique for removing confounding bias from observational data.

---

## Simpson's Paradox: Why We Need a Causal Framework

Before diving into tools, here's a puzzle that shows why raw statistics aren't enough.

### The Kidney Stone Problem

A hospital is evaluating two treatments for kidney stones. Here's the data:

| | Treatment A | Treatment B |
|---|---|---|
| **Small stones** | **93%** (81/87) | 87% (234/270) |
| **Large stones** | **73%** (192/263) | 63% (50/80) |
| **Overall** | 78% (273/350) | **83%** (289/350) |

*Table 2.1: Recovery rates by treatment and stone size*

Treatment A is better for **small stones** (93% vs 87%) and better for **large stones** (73% vs 63%). Yet overall, Treatment B looks better (83% vs 78%).

This is **Simpson's Paradox** — an aggregated statistic reverses when you condition on a third variable.

**What's going on?** Doctors tend to give Treatment A to the harder cases (large stones). Large stones have worse recovery rates regardless of treatment. So Treatment A is associated with harder cases, dragging down its aggregate average — even though it's the better treatment.

The lesson: **raw comparisons are contaminated by how treatment was assigned**. To answer "which treatment is better?", we need to control for the confounding variable (stone size).

---

## Structural Causal Models

A **Structural Causal Model (SCM)** is a formal way to describe how variables are generated — including the causal mechanisms behind them.

### Components

An SCM consists of:
1. **Endogenous variables** — the variables we model (e.g., stone size $S$, treatment $T$, recovery $R$)
2. **Exogenous variables** (noise) — unobserved factors unique to each unit, written $U_S$, $U_T$, $U_R$
3. **Structural equations** — one equation per endogenous variable, specifying how it's caused

### The `:=` Operator

Structural equations use `:=` (the *assignment* or *walrus* operator) instead of `=` to encode **causal direction**:

$$Y := f(X, U_Y)$$

This means: "Once you know $X$, you compute $Y$ from $X$ and noise $U_Y$." If you change $Y$, $X$ does not change. This is unlike a math equation $Y = X$ where changing either side affects the other.

### Kidney Stone SCM

$$S := U_S \quad \text{(stone size is exogenous — determined by biology)}$$

$$T := g(S, U_T) \quad \text{(treatment depends on stone size + physician's judgment)}$$

$$R := f(S, T, U_R) \quad \text{(recovery depends on stone size, treatment, and patient factors)}$$

The causal graph for this system:

```
S ──→ T
│     │
└──→ R
```

$S$ (stone size) causes both $T$ (treatment choice) and $R$ (recovery). This makes $S$ a **confounder**.

---

## Directed Acyclic Graphs (DAGs)

A **Directed Acyclic Graph (DAG)** is a visual representation of an SCM. Each node is a variable; each directed edge ($A \rightarrow B$) means "$A$ is a direct cause of $B$."

### What "Acyclic" Means

**Acyclic** means there are no cycles — you can never follow the arrows and return to your starting node. This encodes the assumption that causality is one-directional in time: causes precede their effects.

If you think you have a cycle (e.g., $\text{price} \leftrightarrow \text{demand}$), you can usually resolve it by adding a time index:

$$\text{price}_{t-1} \rightarrow \text{demand}_t \rightarrow \text{price}_t$$

### What an Edge Means

$A \rightarrow B$ means: "$A$ is a direct cause of $B$, given all other variables in the graph." Equivalently, the structural equation for $B$ includes $A$ as an input.

An **absent edge** is also a claim: it means $A$ has no direct causal effect on $B$ (though they may still be associated via a common cause).

### Three Fundamental Structures

Understanding these three graph patterns is the foundation of all causal reasoning:

**1. Chain (Mediation)**
$$X \rightarrow M \rightarrow Y$$
$X$ causes $Y$ through a mediator $M$. If you condition on $M$, you block this path.

**2. Fork (Common Cause / Confounding)**
$$X \leftarrow C \rightarrow Y$$
$C$ is a confounder — it causes both $X$ and $Y$, creating a spurious association between them. Conditioning on $C$ removes this.

**3. Collider**
$$X \rightarrow C \leftarrow Y$$
$C$ is a collider — caused by both $X$ and $Y$. Conditioning on $C$ *opens* a spurious path between $X$ and $Y$ (selection bias).

> **Key rule**: A path between $X$ and $Y$ is **blocked** by conditioning on a non-collider on the path, or by *not* conditioning on a collider. A path is **open** when all non-colliders are uncontrolled and all colliders are conditioned on.

---

## The Structural (Adjustment) Formula

### The Problem with Naive Comparisons

In the kidney stone example, naive comparison gives:

$$E[R \mid T = A] - E[R \mid T = B] = 78\% - 83\% = -5\%$$

This is **biased** because stone size $S$ confounds the relationship. Doctors preferentially give Treatment A to large-stone patients, who have worse outcomes regardless.

### The Fix: Standardization (Adjustment Formula)

To estimate the effect of *intervening* on treatment (i.e., $do(T = A)$), we need to **remove** the influence of $S$ on $T$ and re-aggregate. The formula:

$$E[R \mid do(T = A)] = \sum_s E[R \mid T = A, S = s] \cdot P(S = s)$$

In words: compute the treatment effect *within each stratum of $S$*, then average across the distribution of $S$ (not conditional on $T$).

More generally, for any set of confounders $\mathbf{X}$:

$$E[Y \mid do(T = t)] = \sum_{\mathbf{x}} E[Y \mid T = t, \mathbf{X} = \mathbf{x}] \cdot P(\mathbf{X} = \mathbf{x})$$

### Applying to Kidney Stones

Using the marginal distribution of stone size:

$$P(S = \text{small}) = \frac{87 + 270}{700} = \frac{357}{700} \approx 0.51$$
$$P(S = \text{large}) = \frac{263 + 80}{700} = \frac{343}{700} \approx 0.49$$

Adjusted estimate for Treatment A:

$$E[R \mid do(T=A)] = 0.93 \times 0.51 + 0.73 \times 0.49 \approx 0.474 + 0.358 = \mathbf{83.2\%}$$

Adjusted estimate for Treatment B:

$$E[R \mid do(T=B)] = 0.87 \times 0.51 + 0.63 \times 0.49 \approx 0.444 + 0.309 = \mathbf{75.3\%}$$

**ATE:**

$$\text{ATE} = E[R \mid do(T=A)] - E[R \mid do(T=B)] = 83.2\% - 75.3\% = +7.9\%$$

Treatment A is better — which aligns with what we saw within each stratum. The adjustment formula recovers the truth.

### The Positivity Assumption

The adjustment formula requires that, for every combination of confounders $\mathbf{X} = \mathbf{x}$ and treatment $T = t$, there is some data:

$$0 < P(T = t \mid \mathbf{X} = \mathbf{x}) < 1$$

If some combination never occurs (e.g., Treatment A was never given to small-stone patients), we have no information about that cell and cannot compute the adjustment. This is the **positivity** (or **overlap**) assumption — it cannot be fixed by collecting more data if the combination was excluded by design.

---

## Interventions and RCTs: The DAG Perspective

A **Randomized Controlled Trial (RCT)** is, in graph terms, an **intervention on the DAG**. When we randomize treatment:

- We surgically **remove all incoming arrows to $T$**
- The resulting "mutilated graph" has $T$ as a root node with no parents

Before (observational):
```
S → T → R
 ↘     ↗
```

After randomization (RCT):
```
T → R
    ↑
    S
```

With no confounders of $T$, the adjustment formula reduces to a simple difference in means:

$$E[R \mid do(T=A)] = E[R \mid T=A]$$

This is why RCTs are the gold standard — they make the causal and observational distributions identical.

---

## Building a DAG: 5 Steps

Building a good DAG is an act of domain expertise + systematic process. Here's the workflow:

### Step 1: List All Relevant Variables

Write down every variable that plausibly plays a role in the data-generating process. Include:
- The treatment $T$ and outcome $Y$
- Any variable that causes $T$, causes $Y$, or causes both
- Don't leave out potential confounders — missing a confounder is one of the biggest risks in causal inference

> **Tip**: Consult domain experts. A missing node can invalidate your entire analysis.

### Step 2: Draw the Edges

For each pair of variables $(A, B)$, ask: *"Does $A$ directly cause $B$?"* If yes, draw $A \rightarrow B$.

A useful heuristic: **time ordering**. A variable that comes earlier in time cannot be caused by something that comes later. If $A$ happens before $B$, the only permissible edge is $A \rightarrow B$.

For simultaneous relationships (like price-demand feedback), use time-indexed variables:
$$\text{price}_{t-1} \rightarrow \text{demand}_t \rightarrow \text{price}_t$$

Another heuristic: ask "*Can $B$ be expressed as a function of $A$ and noise, holding everything else fixed?*" If yes, $A \rightarrow B$.

### Step 3: State Your Assumptions Explicitly

Every absent edge is an assumption. Make them visible:
- "We assume age doesn't affect which treatment was prescribed"
- "We assume no unmeasured confounders between $T$ and $Y$"

The more explicit your assumptions, the more easily others can challenge them — which is good.

### Step 4: Align on Analysis Objectives

Confirm with your team (and stakeholders) what causal question you're actually answering:
- ATE across the whole population?
- ATT (among those who received treatment)?
- Effect in a specific subgroup?

Different questions may require different identification strategies.

### Step 5: Check Positivity

Verify that for each level of your confounders, you have treated *and* untreated units. A simple check: cross-tabulate treatment against each confounder. If any cell is empty, you have a positivity violation.

---

## Code Examples

### Simulating from an SCM

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# Kidney stone SCM
# S: stone size (0 = small, 1 = large)
# T: treatment (0 = B, 1 = A) — doctors prefer A for large stones
# R: recovery

S = np.random.binomial(1, 0.49, n)                          # ~49% large stones
# Treatment assignment: large stone -> more likely to get Treatment A
T = np.random.binomial(1, 0.3 + 0.5 * S, n)                # P(T=A|small)=0.3, P(T=A|large)=0.8
# Recovery: large stones harder to treat, but A is better
R = np.random.binomial(1, 0.87 - 0.24 * S + 0.06 * T, n)  # Treatment A adds ~6pp

df = pd.DataFrame({"stone_size": S, "treatment": T, "recovery": R})
df["stone_label"] = df.stone_size.map({0: "small", 1: "large"})
df["treatment_label"] = df.treatment.map({0: "B", 1: "A"})

print("Recovery rates by treatment and stone size:")
print(df.groupby(["stone_label", "treatment_label"])["recovery"].mean().round(3))

print("\nMarginal (naive) comparison:")
naive = df.groupby("treatment_label")["recovery"].mean()
print(naive.round(3))
print(f"Naive ATE (A - B): {naive['A'] - naive['B']:.3f}")
```

### Computing the Adjustment Formula

```python
# Adjustment formula: standardize over the confounder (stone size)

# Step 1: Estimate E[R | T=t, S=s] within each stratum
strata = df.groupby(["treatment_label", "stone_label"])["recovery"].mean().unstack()
print("\nOutcome by treatment and stone size:")
print(strata.round(3))

# Step 2: Get marginal distribution of S (NOT conditional on T)
p_stone = df["stone_label"].value_counts(normalize=True)
print(f"\nMarginal P(stone size): {p_stone.round(3).to_dict()}")

# Step 3: Apply adjustment formula
ate_A = (strata.loc["A", "small"] * p_stone["small"] +
         strata.loc["A", "large"] * p_stone["large"])

ate_B = (strata.loc["B", "small"] * p_stone["small"] +
         strata.loc["B", "large"] * p_stone["large"])

print(f"\nAdjusted E[R | do(T=A)]: {ate_A:.3f}")
print(f"Adjusted E[R | do(T=B)]: {ate_B:.3f}")
print(f"Adjusted ATE (A - B):    {ate_A - ate_B:.3f}")
# Positive! Treatment A is better, matching what we see within strata
```

### Visualizing Simpson's Paradox

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Plot 1: Naive (aggregated)
naive_rates = df.groupby("treatment_label")["recovery"].mean()
axes[0].bar(naive_rates.index, naive_rates.values, color=["steelblue", "coral"])
axes[0].set_title("Naive Comparison\n(Aggregated)")
axes[0].set_ylabel("Recovery Rate")
axes[0].set_ylim(0.6, 1.0)

# Plot 2: Small stones only
small = df[df.stone_label == "small"].groupby("treatment_label")["recovery"].mean()
axes[1].bar(small.index, small.values, color=["steelblue", "coral"])
axes[1].set_title("Small Stones Only\n(A is better)")
axes[1].set_ylim(0.6, 1.0)

# Plot 3: Large stones only
large = df[df.stone_label == "large"].groupby("treatment_label")["recovery"].mean()
axes[2].bar(large.index, large.values, color=["steelblue", "coral"])
axes[2].set_title("Large Stones Only\n(A is better)")
axes[2].set_ylim(0.6, 1.0)

for ax in axes:
    ax.set_xlabel("Treatment")

plt.suptitle("Simpson's Paradox: Kidney Stone Example", fontsize=13)
plt.tight_layout()
plt.show()
```

### Regression as the Continuous Analog of Adjustment

When the treatment is continuous (or there are many confounders), you can't do discrete stratification. Linear regression is the continuous analog of the adjustment formula:

```python
from sklearn.linear_model import LinearRegression

# Binary example — regression should match adjustment formula
X = pd.get_dummies(df[["treatment_label", "stone_label"]], drop_first=True)
model = LinearRegression().fit(X, df["recovery"])

print("Regression coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.3f}")
# treatment_label_B coefficient ~ negative of the adjusted ATE
```

> **Note**: Linear regression gives the correct adjusted ATE only if the model is correctly specified (no interaction between treatment and confounders, linear effects). For more flexible adjustment, see Chapter 8 (Double Machine Learning).

---

## The 5-Step Causal Inference Process

To summarize the end-to-end workflow introduced in this chapter:

```
1. Define the question       → What intervention? What outcome?
2. List relevant variables   → Treatment, outcome, all potential confounders
3. Build the DAG             → Draw causal arrows, state absent-edge assumptions
4. Identify the estimand     → Is the ATE identifiable? What formula to use?
5. Check data requirements   → Positivity? Enough variation in treatment?
```

This process applies regardless of which estimation method you use (regression, matching, propensity scores, etc.). The graph tells you *what* to adjust for; the method tells you *how*.

---

## Interview Questions

### Technical Questions

**Q1: What is Simpson's Paradox and how does causality resolve it?**

Simpson's Paradox occurs when an association that appears in aggregate data reverses or disappears when the data is stratified. It's resolved by identifying the correct causal structure — specifically, whether the stratifying variable is a confounder (should be controlled for) or a mediator/collider (should not be controlled for). The causal DAG determines which direction is correct, not the data alone.

---

**Q2: Explain the adjustment formula. When does it apply?**

The adjustment formula computes the interventional expectation by standardizing over confounders:

$$E[Y \mid do(T=t)] = \sum_{\mathbf{x}} E[Y \mid T=t, \mathbf{X}=\mathbf{x}] \cdot P(\mathbf{X}=\mathbf{x})$$

It applies when: (1) all confounders $\mathbf{X}$ are observed, (2) you've correctly identified them via a DAG (the backdoor criterion), and (3) the positivity assumption holds.

---

**Q3: What is the positivity assumption? What breaks if it fails?**

Positivity requires that every unit has a nonzero probability of receiving each treatment level, within every stratum of confounders: $0 < P(T=t \mid \mathbf{X}=\mathbf{x}) < 1$ for all $t, \mathbf{x}$.

If it fails, there are combinations of covariates where one treatment was never observed. We have no data to estimate the effect in those cells, and the adjustment formula is undefined there. Increasing sample size doesn't fix this — it's a structural gap in the data.

---

**Q4: What's the difference between a confounder, a mediator, and a collider? Why does it matter?**

| Type | Structure | Should you control for it? |
|------|-----------|---------------------------|
| **Confounder** | $T \leftarrow C \rightarrow Y$ | Yes — opens a backdoor path |
| **Mediator** | $T \rightarrow M \rightarrow Y$ | No (unless you want direct effect only) — blocks causal path |
| **Collider** | $T \rightarrow C \leftarrow Y$ | No — conditioning on it opens a spurious path |

Controlling for the wrong type of variable introduces bias rather than removing it. This is why you need a causal graph — the data alone can't tell you which type a variable is.

---

**Q5: How does an RCT relate to the do-operator?**

An RCT implements $do(T=t)$ by design. It surgically removes all arrows into $T$ in the DAG, making $T$ independent of all confounders. As a result, $E[Y \mid do(T=t)] = E[Y \mid T=t]$ — the interventional and observational distributions coincide, and a simple difference in means gives the causal effect.

---

### Case Study Questions

**Case 1: You find that users who use Feature X have 2x higher LTV. Your PM wants to roll it out to everyone. What do you say?**

This is a classic confounding problem. Power users or highly engaged users are more likely to both adopt Feature X *and* have high LTV regardless. Before recommending a rollout:
1. Draw the DAG — what drives Feature X adoption? What else drives LTV?
2. Check if confounders are measurable (tenure, engagement score, plan tier)
3. Apply adjustment formula (stratify or regress on confounders)
4. Ideally: propose an A/B test before full rollout
5. If already shipped, use DiD or propensity score matching

---

**Case 2: A data scientist says "I'll just put all variables in a regression to control for everything." What's wrong with this approach?**

Adding all available variables is dangerous because:
- You might control for **mediators** (blocking the causal path you want to measure)
- You might condition on **colliders** (opening spurious paths that create bias)
- You might introduce **multicollinearity** without any causal benefit

The right approach is to use a DAG to determine the **minimal adjustment set** — the variables you *need* to control for, and nothing more. The adjustment formula requires correct causal structure, not just more covariates.

---

**Case 3: Your team is comparing two recommendation algorithms using historical data. The new algorithm was used more during the holiday season. How do you adjust for this?**

Holiday season is a confounder — it affects both which algorithm was deployed *and* user engagement/conversion. Steps:
1. Add time/season to the causal graph
2. Stratify by time period (apply adjustment formula across seasons)
3. Or use regression with time fixed effects
4. Better: for future comparisons, use an interleaved experiment (simultaneous A/B test) to eliminate temporal confounding by design
