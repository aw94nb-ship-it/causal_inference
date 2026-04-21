# Chapter 7: Advanced DAGs — Identification and the Do-Calculus

Chapter 2 introduced the three fundamental graph structures (chains, forks, colliders) and the adjustment formula. That's enough to handle simple, three-variable cases. Real-world systems are larger: marketing mix models may have a dozen variables, clinical studies track dozens of biomarkers, and recommender systems involve user behavior, product quality, and platform interventions all interacting. This chapter develops the systematic tools for handling complexity: d-separation for reading independence from any graph, the back-door criterion for knowing exactly which variables to control for, a catalog of good and bad controls, the front-door criterion for cases where back-door fails, and Pearl's do-calculus as a general-purpose identification engine.

---

## Dealing with Complex Graphs

### The Scaling Problem

With three variables, you can see by inspection which paths matter. With seven variables, the number of possible paths between any two nodes can be enormous, and eyeballing is unreliable. We need an algorithm.

The algorithm is **d-separation** (directional separation). It tells you, for any DAG and any set of conditioning variables $Z$, whether two nodes $A$ and $B$ are statistically independent.

### Paths and Blocking

A **path** between nodes $A$ and $B$ is any sequence of edges connecting them, ignoring arrow directions. A path can be blocked or unblocked.

**A path is blocked by a set $Z$ if at least one of the following holds:**

1. The path contains a **non-collider** (a chain $A \to M \to B$ or a fork $A \leftarrow M \to B$) and $M \in Z$ — conditioning on a non-collider blocks the path.

2. The path contains a **collider** ($A \to C \leftarrow B$) and neither $C$ nor any of $C$'s descendants are in $Z$ — a collider that is NOT conditioned on (and has no conditioned descendants) blocks the path.

The reverse: a path is **open** when all non-colliders are unconditioned and at least one collider is conditioned on (or has a conditioned descendant).

### D-Separation

**Definition**: Two nodes $A$ and $B$ are **d-separated** by a set $Z$ if *every* path between $A$ and $B$ is blocked given $Z$. If d-separated, then $A \perp\!\!\!\perp B \mid Z$ in any distribution faithful to the DAG.

If even one path is unblocked, $A$ and $B$ are **d-connected** given $Z$ — they are (generally) associated.

### Worked Example: Enumerating Paths

Consider this 5-variable DAG:

```
       U (unobserved)
      ↗ ↘
    T     Y
    ↓     ↑
    M ────┘
    ↑
    X
```

Variables: $T$ = treatment, $Y$ = outcome, $M$ = mediator, $X$ = pre-treatment covariate, $U$ = unmeasured confounder.

Edges: $U \to T$, $U \to Y$, $T \to M$, $M \to Y$, $X \to M$.

Paths from $T$ to $Y$ (ignoring direction):
1. $T \to M \to Y$ — forward path (causal): non-colliders are $M$. Open unless we condition on $M$.
2. $T \leftarrow U \to Y$ — back-door path (confounding): non-collider is $U$. Open unless we condition on $U$.

If we condition on $\{X\}$:
- Path 1: $X$ is not on this path → still open
- Path 2: $X$ is not on this path → still open

Conditioning on $X$ alone doesn't help because neither path goes through $X$. We would need to condition on $U$ to close path 2 — but $U$ is unobserved. This is precisely when back-door adjustment fails and we need other tools (IV, front-door, do-calculus).

### D-separation in Python with NetworkX

```python
import networkx as nx
from itertools import combinations

def get_all_paths(G, source, target):
    """Get all simple paths ignoring edge direction."""
    G_undirected = G.to_undirected()
    return list(nx.all_simple_paths(G_undirected, source, target))

def is_collider_on_path(G, path, node_idx):
    """Check if node at path[node_idx] is a collider on the path."""
    if node_idx == 0 or node_idx == len(path) - 1:
        return False
    prev_node = path[node_idx - 1]
    next_node = path[node_idx + 1]
    curr_node = path[node_idx]
    # Collider: both neighbors point INTO the current node
    return (G.has_edge(prev_node, curr_node) and G.has_edge(next_node, curr_node))

def path_is_blocked(G, path, conditioning_set):
    """Check if a path is blocked given the conditioning set Z."""
    for i in range(1, len(path) - 1):
        node = path[i]
        if is_collider_on_path(G, path, i):
            # Collider: path blocked if node AND its descendants are NOT in Z
            descendants = nx.descendants(G, node) | {node}
            if not descendants.intersection(conditioning_set):
                return True  # collider not conditioned on -> blocks path
        else:
            # Non-collider: path blocked if node IS in Z
            if node in conditioning_set:
                return True
    return False

def d_separated(G, A, B, Z):
    """Check if A and B are d-separated given Z."""
    paths = get_all_paths(G, A, B)
    if not paths:
        return True  # no paths -> d-separated
    return all(path_is_blocked(G, path, set(Z)) for path in paths)

# Build example DAG
G = nx.DiGraph()
G.add_edges_from([('U', 'T'), ('U', 'Y'), ('T', 'M'), ('M', 'Y'), ('X', 'M')])

print("=== D-Separation Queries ===")
queries = [
    ('T', 'Y', []),
    ('T', 'Y', ['U']),
    ('T', 'Y', ['M']),
    ('T', 'Y', ['U', 'M']),
    ('X', 'Y', []),
    ('X', 'Y', ['M']),
]
for A, B, Z in queries:
    result = d_separated(G, A, B, Z)
    print(f"d-sep({A}, {B} | {Z}): {result}")
```

---

## The Back-door Criterion

### Definition

The back-door criterion gives a precise, mechanical test for whether a set of variables $Z$ is sufficient to identify the causal effect of $T$ on $Y$.

**Definition**: A set of variables $Z$ satisfies the **back-door criterion** relative to $(T, Y)$ in a DAG $G$ if:
1. No node in $Z$ is a descendant of $T$
2. $Z$ blocks every **back-door path** from $T$ to $Y$ — paths that have an arrow *into* $T$

A back-door path is any path from $T$ to $Y$ that begins with an arrow pointing into $T$ (i.e., $T \leftarrow \ldots \to Y$). These paths represent confounding: they create a spurious association between $T$ and $Y$ that is not due to $T$'s causal effect.

**Back-door adjustment formula**: if $Z$ satisfies the back-door criterion, then:

$$P(Y \mid do(T=t)) = \sum_z P(Y \mid T=t, Z=z) \cdot P(Z=z)$$

This is exactly the adjustment formula from Chapter 2. The back-door criterion tells you *when* you're allowed to use it.

### Why Condition 1 Matters

Condition 1 says don't include any descendant of $T$ in $Z$. Why? Because $T$'s descendants are post-treatment variables — controlling for them introduces the bad control problems discussed in Chapter 6 (blocking mediators, conditioning on colliders that are caused by $T$).

### Worked Example: 3-Variable DAG

```
W (confounder)
↓         ↓
T ──────→ Y
```

Back-door paths from $T$ to $Y$: $T \leftarrow W \to Y$ — one path, through $W$.

$Z = \{W\}$:
- Condition 1: $W$ is not a descendant of $T$ ✓
- Condition 2: $\{W\}$ blocks the path $T \leftarrow W \to Y$ (conditioning on the non-collider $W$) ✓

So $\{W\}$ satisfies the back-door criterion. Controlling for $W$ in OLS gives the causal effect.

### Worked Example: 5-Variable DAG

```
     A
    / \
   ↓   ↓
   T   B
    \ / \
     ↓   ↓
     C   Y
      \ /
       ↓
```

Let's be concrete:
- $A$ = age
- $T$ = received new drug
- $B$ = baseline health score (caused by age)
- $C$ = intermediate biomarker (caused by drug and baseline health)
- $Y$ = recovery

Edges: $A \to T$, $A \to B$, $T \to C$, $B \to C$, $B \to Y$, $C \to Y$.

Back-door paths from $T$ to $Y$ (paths into $T$, excluding forward paths):
1. $T \leftarrow A \to B \to Y$ — open. Non-colliders: $A$, $B$. All uncontrolled.
2. $T \leftarrow A \to B \to C \to Y$ — open. Non-colliders: $A$, $B$, $C$.

But $C$ is a mediator ($T \to C \to Y$) — conditioning on $C$ would block the causal path. What's the minimal adjustment set?

Try $Z = \{A\}$:
- Condition 1: $A$ is not a descendant of $T$ ✓
- Path 1: $T \leftarrow A \to B \to Y$ — $A \in Z$, so this path is blocked at $A$ ✓
- Path 2: $T \leftarrow A \to B \to C \to Y$ — $A \in Z$, blocked ✓

$Z = \{A\}$ satisfies back-door! We only need to control for age.

Try $Z = \{B\}$:
- Condition 1: $B$ is not a descendant of $T$ ✓
- Path 1: $T \leftarrow A \to B \to Y$ — $B$ is a non-collider, $B \in Z$, blocked ✓
- Path 2: $T \leftarrow A \to B \to C \to Y$ — $B$ is a non-collider, blocked ✓

$Z = \{B\}$ also satisfies back-door. Both age and baseline health are valid adjustment sets.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 5000

# DGP matching the 5-variable example above
# True effect of drug (T) on recovery (Y) via biomarker (C)
age = np.random.normal(50, 10, n)
drug = (age > 55).astype(float) + np.random.normal(0, 0.5, n)  # older patients more likely treated
drug = (drug > 0.5).astype(float)

baseline_health = -0.5 * age + np.random.normal(0, 5, n)  # older = worse baseline
biomarker = 3.0 * drug + 0.4 * baseline_health + np.random.normal(0, 2, n)
recovery = 2.0 * biomarker + 0.3 * baseline_health + np.random.normal(0, 5, n)

# True total effect of drug on recovery:
# drug -> biomarker (3.0) -> recovery (2.0) = 6.0 total (via mediator)

df = pd.DataFrame({
    'recovery': recovery, 'drug': drug, 'age': age,
    'baseline_health': baseline_health, 'biomarker': biomarker
})

# No controls (confounded)
m0 = smf.ols('recovery ~ drug', data=df).fit()

# Control for age (valid adjustment set 1)
m1 = smf.ols('recovery ~ drug + age', data=df).fit()

# Control for baseline_health (valid adjustment set 2)
m2 = smf.ols('recovery ~ drug + baseline_health', data=df).fit()

# Control for biomarker (BAD -- mediator)
m3 = smf.ols('recovery ~ drug + biomarker', data=df).fit()

print("=== Back-door Adjustment ===")
print(f"True total effect of drug:              ~6.0")
print(f"No controls (biased):                    {m0.params['drug']:.3f}")
print(f"Control for age (valid):                 {m1.params['drug']:.3f}")
print(f"Control for baseline_health (valid):     {m2.params['drug']:.3f}")
print(f"Control for biomarker (bad -- mediator): {m3.params['drug']:.3f}")
```

---

## Good and Bad Controls

This section is a comprehensive reference. Use the DAG as your guide — not intuition about "what seems related."

### Good Control: Confounder

A **confounder** is a variable that causes both $T$ and $Y$. It creates a fork:

```
C
↓ ↓
T   Y
```

Confounders are the primary source of observational bias. Controlling for them closes the back-door path and removes the spurious association.

**Rule**: always control for confounders.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(1)
n = 3000

# Confounder: user tenure affects both feature adoption AND revenue
tenure = np.random.exponential(2, n)  # years on platform
treatment = (0.4 * tenure + np.random.normal(0, 1, n) > 1).astype(float)  # tenure -> adoption
revenue = 5.0 * treatment + 8.0 * tenure + np.random.normal(0, 5, n)

df = pd.DataFrame({'revenue': revenue, 'treatment': treatment, 'tenure': tenure})

m_no_ctrl = smf.ols('revenue ~ treatment', data=df).fit()
m_ctrl = smf.ols('revenue ~ treatment + tenure', data=df).fit()

print("Confounder example (true effect = 5.0):")
print(f"  No control:             {m_no_ctrl.params['treatment']:.2f}  (biased)")
print(f"  Controlling for tenure: {m_ctrl.params['treatment']:.2f}  (unbiased)")
```

### Good Control: Pure Outcome Predictor

A variable $W$ that affects $Y$ but not $T$:

```
W → Y    (W has no arrow to T)
T → Y
```

$W$ is not a confounder — there's no back-door path through $W$. But including it in regression reduces the residual variance of $Y$, which:
- **Improves precision** (smaller SE on the treatment coefficient)
- **Does not bias** the treatment effect estimate

This is valuable in experiments: including pre-treatment covariates that predict the outcome (like baseline outcome value) can substantially increase power.

**Rule**: including pure outcome predictors is safe and often beneficial.

### Bad Control: Mediator

A **mediator** $M$ is on the causal path from $T$ to $Y$:

$$T \to M \to Y$$

Controlling for $M$ blocks this path. You estimate only the "direct" effect of $T$ on $Y$ that does *not* go through $M$ — which may be zero or much smaller than the total effect.

**Real-world example**: estimating the effect of "seeing an ad" ($T$) on "purchase" ($Y$), controlling for "clicking the ad" ($M$). Clicks are the primary mechanism; controlling for them removes the main channel of effect.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 5000

# T = saw ad, M = clicked ad (mediator), Y = purchase
saw_ad = np.random.binomial(1, 0.5, n)

# Click probability: higher if saw ad (mediator)
click_prob = 0.02 + 0.15 * saw_ad  # baseline click rate 2%, goes to 17% with ad
clicked = np.random.binomial(1, click_prob, n)

# Purchase: both direct effect of seeing ad AND effect via click
# Direct effect (saw ad but didn't click): 0.03
# Via click: 0.25
purchase_prob = 0.05 + 0.03 * saw_ad + 0.25 * clicked
purchase = np.random.binomial(1, np.clip(purchase_prob, 0, 1), n)

df = pd.DataFrame({'purchase': purchase, 'saw_ad': saw_ad, 'clicked': clicked})

# Total effect (correct for measuring ad impact)
m_total = smf.ols('purchase ~ saw_ad', data=df).fit()

# Direct effect only (correct only if you specifically want "effect of seeing ad, not via clicking")
m_direct = smf.ols('purchase ~ saw_ad + clicked', data=df).fit()

# True total effect = direct (0.03) + through click (0.15 * 0.25 = 0.0375) ≈ 0.0675
print("=== Mediator as Bad Control ===")
print(f"True total effect:             ~{0.03 + 0.15 * 0.25:.4f}")
print(f"Total effect (no mediator ctrl): {m_total.params['saw_ad']:.4f}")
print(f"Direct effect (controls click):  {m_direct.params['saw_ad']:.4f}")
print()
print("Controlling for 'clicked' strips out the click channel.")
print("Use total effect for ROI measurement; direct effect only for specific decomposition.")
```

### Bad Control: Collider

A **collider** $C$ is caused by both $T$ and $Y$ (or their causes):

$$T \to C \leftarrow Y$$

As established in Chapter 2, conditioning on a collider **opens** a spurious path. By restricting to a level of $C$, you make $T$ and $Y$ correlated even when they have no causal relationship.

**Real-world example**: estimating the effect of team size ($T$) on project success ($Y$). If you restrict to "projects that got funded" ($C$), and both team size and project quality drive funding, you've conditioned on a collider. Within funded projects, poor team size may be "compensated" by high project quality, inducing a spurious negative correlation.

### Bad Control: M-Bias

M-bias is the most counterintuitive case. The DAG is:

```
X1 ──→ T
 ↘
  C
 ↗
X2 ──→ Y
```

More precisely:
- $X_1 \to T$ and $X_1 \to C$
- $X_2 \to Y$ and $X_2 \to C$
- There is NO arrow between $T$ and $Y$ (true causal effect = 0 in this example)
- $C$ is a collider on the path $T \leftarrow X_1 \to C \leftarrow X_2 \to Y$

**Before conditioning on $C$**: the path $T \leftarrow X_1 \to C \leftarrow X_2 \to Y$ is **blocked** at the collider $C$. There is no confounding. A simple regression of $Y$ on $T$ gives an unbiased zero.

**After conditioning on $C$**: the collider $C$ becomes conditioned, which **opens** the path. Now $X_1$ and $X_2$ become correlated within levels of $C$ (explaining away the value of $C$). This induces a spurious $T$-$Y$ association through $X_1$-$X_2$ correlation.

The graph has an "M" shape:

```
X1    X2
 ↓  ↗  ↓
  C    Y
  ↑
  (collider -- don't condition!)
```

**Real-world example: College Admissions**. Suppose:
- $X_1$ = high school athletic achievement → causes $T$ = athletic scholarship award AND → causes applying via Common App ($C$)
- $X_2$ = high school academic achievement → causes $Y$ = college GPA AND → causes Common App ($C$)
- True effect of $T$ (athletic scholarship) on $Y$ (GPA) = 0

If you analyze only students who applied via Common App (conditioning on $C$), you open the $X_1$-$X_2$ path. Within Common App applicants, if $X_1$ is high (explaining why they applied), $X_2$ need not be as high (and vice versa). This makes $X_1$ and $X_2$ negatively correlated within Common App applicants, inducing a spurious correlation between athletic scholarships and GPA.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(7)
n = 20000

# M-bias simulation
# X1: athletic achievement (affects T and C)
# X2: academic achievement (affects Y and C)
# True causal effect of T on Y = 0

X1 = np.random.normal(0, 1, n)  # athletic achievement
X2 = np.random.normal(0, 1, n)  # academic achievement

# C = Common App usage (collider -- caused by both X1 and X2)
C_latent = 0.8 * X1 + 0.8 * X2 + np.random.normal(0, 0.5, n)
C = (C_latent > 0).astype(int)  # binary: used Common App or not

# T = athletic scholarship (caused by X1)
T = 0.6 * X1 + np.random.normal(0, 1, n)

# Y = college GPA (caused by X2, NOT by T)
Y = 0.7 * X2 + np.random.normal(0, 1, n)

df = pd.DataFrame({'Y': Y, 'T': T, 'C': C, 'X1': X1, 'X2': X2})

# Correct: do not control for C
m_correct = smf.ols('Y ~ T', data=df).fit()

# Bad: controlling for the collider C
m_collider = smf.ols('Y ~ T + C', data=df).fit()

# Also bad: restricting to C=1 (same as conditioning on C)
df_common_app = df[df['C'] == 1]
m_restricted = smf.ols('Y ~ T', data=df_common_app).fit()

print("=== M-Bias: Collider on Back-door Path ===")
print(f"True causal effect of T on Y:    0.0000")
print(f"Correct (no C control):           {m_correct.params['T']:.4f}")
print(f"Bad: controlling for C:           {m_collider.params['T']:.4f}  (bias introduced!)")
print(f"Bad: restricting to C=1:          {m_restricted.params['T']:.4f}  (same bias)")
print()
print("Adding C as a control creates a spurious negative correlation.")
print("X1 and X2 become correlated within levels of C,")
print("and this leaks into the T-Y association.")
```

### Neutral Control: Variable That Only Affects Y

If a variable $W$ affects only $Y$ (not $T$, not a collider, not a mediator):

```
T → Y ← W
```

$W$ is not on any back-door path. Including it doesn't bias the estimate but reduces residual variance — a pure win for efficiency.

**Example**: including "device type" when estimating the effect of a new feature on session length. If device type doesn't influence feature adoption probability (random assignment), it's a pure outcome predictor — safe and useful to include.

### Summary Table: Control Variable Decision Rules

| Control Type | DAG Structure | Include? | Effect of Including |
|---|---|---|---|
| Confounder | Causes T and Y | Yes | Removes confounding bias |
| Pure Y predictor | Affects Y, not T | Yes (optional) | Reduces variance, no bias change |
| Mediator | T → M → Y | Only for direct effect | Blocks causal path, underestimates total |
| Collider | T → C ← Y | No | Opens spurious path |
| M-bias collider | Collider on back-door path | No | Introduces bias where none existed |
| Post-treatment var | Caused by T | No | Post-treatment contamination |

---

## The Front-door Criterion

### When Back-door Fails

The back-door criterion requires that you can measure and condition on enough variables to close all back-door paths. If there is an **unmeasured confounder** $U$ on a back-door path, the criterion cannot be satisfied with observed variables alone.

```
  U (unobserved)
 ↙          ↘
T ──→ M ──→ Y
```

Here, $U$ creates a back-door path $T \leftarrow U \to Y$. Since $U$ is unobserved, we cannot condition on it. But notice that $M$ is a measured mediator. The **front-door criterion** exploits this structure.

### Definition

A set of variables $M$ satisfies the **front-door criterion** relative to $(T, Y)$ if:

1. All directed paths from $T$ to $Y$ go through $M$ (i.e., $M$ "intercepts" the entire effect)
2. There are no unblocked back-door paths from $T$ to $M$ (conditional on the empty set)
3. All back-door paths from $M$ to $Y$ are blocked by $T$

When these hold, the causal effect is identified by the **front-door formula**:

$$P(Y \mid do(T=t)) = \sum_m P(M=m \mid T=t) \sum_{t'} P(Y \mid T=t', M=m) \cdot P(T=t')$$

### Intuition

The formula does two adjustments:

1. **$\sum_m P(M=m \mid T=t)$**: estimate the effect of $T$ on $M$. Since there's no back-door from $T$ to $M$ (unmeasured $U$ doesn't affect $M$ directly), this is identified by simple conditioning.

2. **$\sum_{t'} P(Y \mid T=t', M=m) \cdot P(T=t')$**: estimate the effect of $M$ on $Y$, adjusting for $T$ as the confounder between $M$ and $Y$. The variable $T$ blocks the back-door path $M \leftarrow T \leftarrow U \to Y$.

The key insight: even though $U$ is unobserved, the front-door criterion identifies the causal effect by combining two identifiable pieces.

### Classic Example: Smoking, Tar, and Cancer

This is Pearl's original motivating example. In the 1960s:
- $T$ = smoking
- $M$ = tar deposits in lungs
- $Y$ = lung cancer
- $U$ = genetic predisposition (unobserved): may cause both smoking tendency and cancer

The backdoor path $T \leftarrow U \to Y$ is unblocked (genetics unmeasured). But:
- All of smoking's effect on cancer is mediated through tar ($T \to M \to Y$) — no direct path assumed
- Tar deposits have no backdoor from smoking ($T$ doesn't affect $M$ except causally)
- The backdoor from $M$ to $Y$ through $U$ is blocked by conditioning on $T$

Front-door formula in this context:

$$P(\text{cancer} \mid do(\text{smoke})) = \sum_m P(\text{tar} = m \mid \text{smoke}) \sum_{t'} P(\text{cancer} \mid \text{smoke} = t', \text{tar} = m) \cdot P(\text{smoke} = t')$$

This identifies the causal effect of smoking on cancer without needing the genetic confounder.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

# Front-door: Smoking -> Tar -> Cancer, with unobserved genetic confounder
genetics = np.random.normal(0, 1, n)  # unobserved

# Smoking: caused by genetics
smoke_prob = 1 / (1 + np.exp(-0.5 * genetics))
smokes = np.random.binomial(1, smoke_prob, n)

# Tar: caused by smoking only (no direct effect of genetics on tar)
tar_level = 3.0 * smokes + np.random.normal(0, 1, n)

# Cancer: caused by tar AND genetics (genetics is back-door confounder)
cancer_prob = 1 / (1 + np.exp(-(0.8 * tar_level + 0.7 * genetics - 2)))
cancer = np.random.binomial(1, cancer_prob, n)

df = pd.DataFrame({'smokes': smokes, 'tar': tar_level, 'cancer': cancer})

# True effect via do-calculus: what is P(cancer | do(smoke=1)) - P(cancer | do(smoke=0))?
# Use the true DGP to compute (oracle)
tar_if_smoke1 = 3.0 * 1 + np.random.normal(0, 1, n)
tar_if_smoke0 = 3.0 * 0 + np.random.normal(0, 1, n)
# Average cancer prob under each intervention (using true DGP, averaging over genetics)
p_cancer_smoke1 = (1 / (1 + np.exp(-(0.8 * tar_if_smoke1 + 0.7 * genetics - 2)))).mean()
p_cancer_smoke0 = (1 / (1 + np.exp(-(0.8 * tar_if_smoke0 + 0.7 * genetics - 2)))).mean()
true_ate = p_cancer_smoke1 - p_cancer_smoke0

# Naive estimate (biased -- ignores genetic confounder)
naive_ate = df[df['smokes'] == 1]['cancer'].mean() - df[df['smokes'] == 0]['cancer'].mean()

# Front-door formula estimate
# Step 1: estimate P(tar | smokes=1) and P(tar | smokes=0)
tar_given_smoke1 = df[df['smokes'] == 1]['tar'].values
tar_given_smoke0 = df[df['smokes'] == 0]['tar'].values
p_smoke1 = smokes.mean()

# Step 2: for each value of tar, estimate P(cancer | smokes=t', tar=m) averaged over P(T)
# Use a linear approximation for E[cancer | smokes, tar]
import statsmodels.formula.api as smf
model_cancer = smf.ols('cancer ~ smokes + tar', data=df).fit()

# Front-door adjustment (using Monte Carlo over tar values)
def front_door_estimate(tar_samples, df, model, p_smoke1):
    """Estimate P(Y | do(T=t)) via front-door."""
    # For each tar value m, compute sum_{t'} P(Y | T=t', M=m) * P(T=t')
    p_smoke0 = 1 - p_smoke1
    inner = np.array([
        p_smoke1 * model.predict(pd.DataFrame({'smokes': [1], 'tar': [m]})).values[0] +
        p_smoke0 * model.predict(pd.DataFrame({'smokes': [0], 'tar': [m]})).values[0]
        for m in tar_samples
    ])
    return inner.mean()

fd_do_smoke1 = front_door_estimate(tar_given_smoke1, df, model_cancer, p_smoke1)
fd_do_smoke0 = front_door_estimate(tar_given_smoke0, df, model_cancer, p_smoke1)
fd_ate = fd_do_smoke1 - fd_do_smoke0

print("=== Front-Door Identification ===")
print(f"True ATE (oracle):                   {true_ate:.4f}")
print(f"Naive estimate (biased):              {naive_ate:.4f}")
print(f"Front-door estimate:                  {fd_ate:.4f}")
print()
print("Front-door recovers the true effect without observing the genetic confounder.")
```

---

## Revisiting Previous Methods Through the DAG Lens

DAGs unify all causal inference methods by clarifying the identification conditions each one requires.

### Randomized Controlled Trials (RCTs)

An RCT **intervenes** on the DAG by removing all arrows into $T$. The mutilated graph has $T$ with no parents, so:
- There are no back-door paths (nothing points into $T$)
- The back-door criterion is trivially satisfied with $Z = \emptyset$
- $E[Y \mid do(T=t)] = E[Y \mid T=t]$ — simple comparison is valid

From the DAG perspective, randomization is the "gold standard" because it eliminates confounding structurally, not statistically.

### Observational Regression / Matching / IPW

All three methods share the same identification assumption:

**Conditional ignorability (unconfoundedness)**:

$$Y(t) \perp\!\!\!\perp T \mid X \quad \text{for all } t$$

In DAG terms: after conditioning on $X$, there are no open back-door paths from $T$ to $Y$. In other words, $X$ satisfies the back-door criterion.

- **Regression**: implements the adjustment formula parametrically (linear)
- **Matching**: implements it nonparametrically by finding similar units
- **IPW**: weights each observation by $1/P(T=t \mid X)$ to re-weight the sample to mimic randomization

All three are valid under the same DAG condition. They differ in:
- **Functional form assumptions**: regression assumes linearity; matching is nonparametric
- **Efficiency**: regression is efficient under linearity; matching/IPW can be more robust to misspecification
- **Overlap sensitivity**: IPW breaks down when propensity scores are near 0 or 1

### Instrumental Variables (Preview)

When there is an unmeasured confounder $U$ that blocks the back-door from observed variables, we need a different identification strategy. An **instrument** $Z$ is a variable that:
1. Affects $T$ (relevance): $Z \to T$
2. Has no direct effect on $Y$ except through $T$ (exclusion): no path $Z \to Y$ except through $T$
3. Is independent of the unmeasured confounder $U$: no back-door path from $Z$ to $Y$

In DAG terms:

```
Z → T → Y
      ↑↗
      U (unobserved)
```

IV doesn't satisfy the back-door criterion (because $U$ is unobserved). Instead, it exploits the front-door-like structure of $Z$ as an exogenous variation in $T$. The IV estimand is $\text{Cov}(Y, Z) / \text{Cov}(T, Z)$.

### Regression Discontinuity (Preview)

RDD exploits a threshold rule: units just above a threshold get treated, units just below don't. This creates a local "as-if randomized" region around the cutoff. In DAG terms, the assignment variable $S$ determines $T$ through a deterministic threshold; local to the cutoff, $S$ is approximately uncorrelated with unmeasured confounders — the back-door paths are locally negligible.

### Summary: Identification Conditions

| Method | DAG condition | Back-door satisfied? | Works with unmeasured U? |
|---|---|---|---|
| RCT | No arrows into T | Trivially | Yes (by design) |
| OLS with controls | Back-door criterion met by observed X | Yes | No |
| Matching / IPW | Same as OLS | Yes | No |
| IV | Exclusion + relevance + independence | No (bypasses it) | Yes |
| RDD | Local as-if randomization | Locally | Yes (locally) |
| Front-door | Front-door criterion met | Different path | Yes |

---

## The Do-Calculus

### Motivation

The back-door and front-door criteria cover many practical cases, but not all. Some causal effects are identifiable from observational data even when neither criterion applies — but it's not obvious how. **Do-calculus** is Pearl's complete system for answering this question.

### Three Rules

The do-calculus consists of three rules that transform interventional distributions $P(\cdot \mid do(\cdot))$ into observational distributions $P(\cdot \mid \cdot)$. You don't need to memorize them — the key insight is that they provide an algebra for manipulating causal expressions.

Let $G$ be a DAG and $G_{\overline{X}}$ denote the graph where all arrows into $X$ are removed (intervened), and $G_{\underline{X}}$ the graph where all arrows out of $X$ are removed.

**Rule 1 — Insertion/deletion of observations**:

$$P(y \mid do(t), z, w) = P(y \mid do(t), w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid T, W)_{G_{\overline{T}}}$$

If $Y$ and $Z$ are d-separated in the graph where $T$'s parents are removed, you can add or drop $Z$ from the conditioning set.

**Rule 2 — Action/observation exchange**:

$$P(y \mid do(t), do(z), w) = P(y \mid do(t), z, w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid T, W)_{G_{\overline{T}, \underline{Z}}}$$

If $Y$ and $Z$ are d-separated in the graph with $T$'s parents removed and $Z$'s children removed, you can replace the intervention $do(Z=z)$ with observation $Z=z$. This is how "back-door adjustment" is derived — the back-door criterion ensures this rule applies.

**Rule 3 — Deletion of actions**:

$$P(y \mid do(t), do(z), w) = P(y \mid do(t), w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid T, W)_{G_{\overline{T}, \overline{Z(W)}}}$$

If $Y$ and $Z$ are d-separated in the graph with certain interventions applied, the effect of $do(Z)$ can be ignored.

### The ID Algorithm

The **ID algorithm** (Shpitser and Pearl, 2006) is a complete algorithm that:
1. Takes a DAG and a causal query $P(Y \mid do(T))$
2. Either returns a formula expressing the query in terms of observational distributions, or declares the query **non-identifiable**

Non-identifiability means: even with infinite data, you cannot determine $P(Y \mid do(T))$ from observational data and the assumed DAG. You need either a different design (experiment, IV) or additional structural assumptions.

### Practical Takeaway

For most applied work:
1. Try back-door criterion first — it covers the majority of cases
2. If back-door fails (unmeasured confounder on every adjustment set), try front-door
3. If both fail, use `dowhy` or `causal-learn` to check identifiability automatically
4. If non-identifiable, you need a designed experiment or an instrumental variable

### DoWhy Example: Automated Identification

```python
# Install: pip install dowhy
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# Simulate a 4-variable system
# W (confounder) -> T, W -> Y, T -> Y (true causal effect)
W = np.random.normal(0, 1, n)
T = 0.5 * W + np.random.normal(0, 1, n)
T_binary = (T > 0).astype(float)
Y = 3.0 * T_binary + 1.5 * W + np.random.normal(0, 1, n)

df = pd.DataFrame({'Y': Y, 'T': T_binary, 'W': W})

try:
    import dowhy
    from dowhy import CausalModel

    # Define the causal graph in DOT notation
    model = CausalModel(
        data=df,
        treatment='T',
        outcome='Y',
        graph="""
        digraph {
            W -> T;
            W -> Y;
            T -> Y;
        }
        """
    )

    # Step 1: Identify the causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("=== DoWhy: Identified Estimand ===")
    print(identified_estimand)

    # Step 2: Estimate using back-door adjustment
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
    print(f"\nEstimated effect:  {estimate.value:.3f}")
    print(f"True effect:       3.000")

    # Step 3: Refutation test (placebo treatment)
    refutation = model.refute_estimate(
        identified_estimand, estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=20
    )
    print(f"\nPlacebo refutation: {refutation}")

except ImportError:
    # Fallback: show the concept without dowhy
    import statsmodels.formula.api as smf

    print("=== Manual back-door adjustment (dowhy not installed) ===")
    m = smf.ols('Y ~ T_binary + W', data=df.rename(columns={'T': 'T_binary'})).fit()
    df2 = df.copy()
    df2.columns = ['Y', 'T_binary', 'W']
    m = smf.ols('Y ~ T_binary + W', data=df2).fit()
    print(f"Estimated effect: {m.params['T_binary']:.3f}")
    print(f"True effect:      3.000")
    print()
    print("DoWhy automates: identification, estimation, and refutation tests.")
    print("It checks back-door, front-door, and do-calculus identifiability.")
```

### Identifiability Decision Tree

```
Is there a set Z of observed variables satisfying the back-door criterion?
├── YES → Use back-door adjustment (regression / matching / IPW)
└── NO
    └── Is there a set M satisfying the front-door criterion?
        ├── YES → Use front-door formula
        └── NO
            └── Run the ID algorithm (use dowhy)
                ├── Identifiable → Returns a formula
                └── Not identifiable → Need an experiment or IV
```

---

## Interview Questions

### Technical Q&A

**Q1: What is d-separation, and how do you use it to read conditional independences from a DAG?**

A: D-separation is an algorithm for reading statistical independence from a DAG. Two nodes $A$ and $B$ are d-separated by a set $Z$ if every path between them is blocked given $Z$. A path is blocked if it contains (a) a non-collider that is in $Z$ — conditioning on a non-collider blocks the path — or (b) a collider whose neither itself nor its descendants are in $Z$ — an unconditioned collider naturally blocks its path. If $A$ and $B$ are d-separated by $Z$, then in any distribution faithful to the DAG, $A \perp\!\!\!\perp B \mid Z$. This tells you which variables to condition on to make comparisons valid, and which variables not to condition on (colliders).

**Q2: State the back-door criterion. Why must we exclude descendants of the treatment from the adjustment set?**

A: A set $Z$ satisfies the back-door criterion relative to $(T, Y)$ if: (i) no member of $Z$ is a descendant of $T$, and (ii) $Z$ blocks every path from $T$ to $Y$ that has an arrow into $T$ (the "back-door paths"). Condition (i) is crucial because descendants of $T$ are post-treatment variables — they could be mediators (controlling blocks the causal path), or colliders caused by $T$ (conditioning opens spurious paths). For example, if $M$ is caused by $T$ and we include $M$ in $Z$, we block $T \to M \to Y$ and underestimate the total effect. Condition (ii) ensures all confounding is removed. Together, they guarantee that conditioning on $Z$ eliminates all non-causal associations without distorting the causal ones.

**Q3: Explain M-bias with an example. Why is it counterintuitive?**

A: M-bias occurs when you control for a collider on a back-door path that is already blocked. Example: $X_1 \to T$, $X_2 \to Y$, $X_1 \to C \leftarrow X_2$, with no direct path from $T$ to $Y$. The path $T \leftarrow X_1 \to C \leftarrow X_2 \to Y$ is blocked at the collider $C$ — no confounding exists. If you add $C$ to your model (perhaps because it's correlated with both $T$ and $Y$ through $X_1$ and $X_2$), you open this path and create bias where none existed. It's counterintuitive because we're taught to add correlated variables as controls. But $C$'s correlation with $T$ and $Y$ is spurious — it goes through $X_1$ and $X_2$, not through a direct causal mechanism. Controlling for it makes things worse, not better.

**Q4: When does the front-door criterion apply? What are the three conditions?**

A: The front-door criterion applies when the back-door criterion cannot be satisfied due to unmeasured confounders, but there's a measured mediator that intercepts the full causal effect. The three conditions for a set $M$ to satisfy the front-door criterion relative to $(T, Y)$: (1) All directed paths from $T$ to $Y$ go through $M$ — $M$ fully mediates the effect; (2) There are no unblocked back-door paths from $T$ to $M$ — the unmeasured confounder doesn't affect $M$ directly; (3) All back-door paths from $M$ to $Y$ are blocked by $T$ — you can adjust for the $M$-$Y$ confounding by conditioning on $T$. Classic example: smoking → tar deposits → cancer, with an unmeasured genetic confounder affecting smoking and cancer but not tar formation.

**Q5: What is the do-calculus, and what does "non-identifiability" mean in practice?**

A: The do-calculus is a complete set of three algebraic rules (Pearl, 1995) for transforming interventional probability expressions $P(Y \mid do(T))$ into observational expressions $P(Y \mid T)$. The rules tell you when you can replace an intervention with an observation, drop conditioning variables, or ignore interventions — based on d-separation in modified graphs. The ID algorithm uses these rules exhaustively to either identify a causal effect or prove it's non-identifiable. Non-identifiability means: given the assumed DAG, no function of the observed data can uniquely determine $P(Y \mid do(T))$, regardless of sample size. The only fixes are: (a) an experiment (randomize $T$), (b) an instrument (IV), (c) additional structural assumptions, or (d) revising the DAG (different set of measured variables).

### Case Study Questions

**Case 1**: You're measuring the effect of a price discount ($T$) on customer lifetime value ($Y$). You have access to: customer demographics, past purchase history, whether the customer redeemed the discount ($M$), and whether the customer complained about a product ($C$, which is caused by both redeeming a discount and having a bad experience, which also correlates with churn). You're considering controlling for $M$ and $C$. Walk through the DAG reasoning for each.

*Key analysis*: Redemption $M$ is a mediator ($T \to M \to Y$) — the discount's effect on LTV is partly through increased engagement from redemption. Controlling for $M$ strips out the primary channel. Only control for $M$ if you specifically want "does discounting affect LTV through channels other than redemption?" Complaint $C$ is likely a collider: it's caused by discount redemption (which may cause both more purchases and more product exposure) AND by bad product experience (which causes churn → lower LTV). Conditioning on $C$ would open the path between discount and LTV through the product quality channel — introducing bias. Leave $C$ out. Safe controls: pre-treatment demographics and past purchase history (true confounders).

**Case 2**: A social media platform wants to estimate the effect of showing users a "trending" badge on a post ($T$) on engagement ($Y$). The platform suspects there's an unmeasured confounder: content quality ($U$, not directly measurable). However, the platform can measure "shares" ($M$, which is caused by the badge and is the primary mechanism driving additional engagement). Under what conditions can you use the front-door criterion here, and what must you verify?

*Key analysis*: Front-door requires: (1) all of badge's effect on engagement goes through shares — verify this means no direct "badge → engagement" path other than via shares; (2) no unmeasured confounder directly affects shares (only affects badge assignment and engagement) — verify that content quality $U$ affects badge assignment and engagement but not shares independently; (3) the confounder-shares back-door path ($M \leftarrow T \leftarrow U \to Y$) is blocked by conditioning on $T$. If these hold, you can identify the effect. Practically, condition (1) is suspect — the badge may directly increase engagement via social proof independent of shares. Verify with domain knowledge and sensitivity analysis.

**Case 3**: You work at an e-commerce company. Marketing claims that users who see a promotional email ($T$) are more likely to purchase ($Y$). But you notice that email open rates are highest among users who are already highly engaged. What DAG structure describes this, what variable(s) should you control for, and what variable(s) must you not control for?

*DAG*: User engagement $U$ (or measurable proxy like past purchase frequency) → email open ($T$) and → purchase ($Y$). This is a confounding fork. You must control for past purchase frequency / engagement level to close the back-door. However: if "email open" is what they call $T$, then "clicked link in email" is a mediator — don't control for it. If their data includes "was assigned to email group" vs "opened the email," note that opening is post-treatment self-selection (the treatment is assignment, not opening). In that case, opening is a mediator and controlling for it underestimates the ITT effect. The correct estimand is the ITT (assigned to email → purchase), and the correct adjustment set is the pre-treatment engagement variables that predict assignment.

**Case 4**: An analyst controls for "number of prior support tickets" when estimating the effect of a new onboarding flow ($T$) on 90-day retention ($Y$). Their reasoning: "users with more problems have fewer tickets after the new flow, and those with fewer tickets retain better, so I should control for it." Is this reasoning valid? What's the potential issue?

*The issue*: "Number of prior support tickets" is ambiguous. If measured *before* the new onboarding flow, it's a valid pre-treatment confounder — users with more historical issues may both adopt new features differently and have different retention patterns. Control for it. But if measured *after* the new flow starts, it's a post-treatment variable. The new onboarding flow might directly affect how many support tickets users submit. Controlling for post-treatment ticket count would partially block the causal path $T \to \text{tickets} \to \text{retention}$, introducing mediator bias. Always verify the timing of every covariate relative to treatment. Pre-treatment: control freely. Post-treatment: generally don't control (unless you specifically want a direct effect estimate).
