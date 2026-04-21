# Fundamentals of Causal Inference

These notes follow the O'Reilly causal inference curriculum. Each chapter covers core concepts, key formulas, real-world examples, Python code, and interview questions.

## Chapters

| # | Chapter | Key Topics |
|---|---------|-----------|
| 1 | [Introducing Causality](chapter1) | Causality vs. correlation, do-operator, Pearl's ladder, experimental vs. observational studies |
| 2 | [Causal Models and the Adjustment Formula](chapter2) | Simpson's paradox, SCMs, DAGs, adjustment formula, positivity |
| 3 | [Applying Causal Inference](chapter3) | End-to-end workflow, DoWhy, counterfactual prediction, causal vs. ML models |
| 4 | [Matching Methods](chapter4) | Exact matching, Mahalanobis distance, nearest-neighbor, balance (SMD), ATT |
| 5 | [Propensity Score Methods](chapter5) | Propensity score estimation, PSM, IPW, doubly robust (AIPW) |
| 6 | [Linear Regression for Causal Inference](chapter6) | FWL theorem, partialling out, bad controls, clustered SEs, heterogeneous effects |
| 7 | [Advanced DAGs — Identification and Do-Calculus](chapter7) | Back-door criterion, front-door criterion, d-separation, do-calculus |
| 8 | [Double Machine Learning](chapter8) | DML algorithm, cross-fitting, EconML, CATE with CausalForest |
| 9 | [Instrumental Variables](chapter9) | Wald estimator, 2SLS, LATE, weak instruments, classic IV examples |
| 10 | [Difference-in-Differences](chapter10) | 2×2 DiD, parallel trends, event study, TWFE, staggered adoption |
| 11 | [Synthetic Control](chapter11) | Weighted counterfactual, California smoking example, permutation inference |

## Methods at a Glance

| Situation | Recommended Method | Chapter |
|-----------|-------------------|---------|
| Randomized experiment | Difference in means / regression | 1, 6 |
| Observational, all confounders measured, low-dim | Regression adjustment | 6 |
| Observational, many confounders | Propensity score (IPW/AIPW) | 5 |
| Observational, very high-dimensional confounders | Double Machine Learning | 8 |
| Observational, want comparable units | Matching | 4 |
| Complex causal graph, need to identify estimand | DAG analysis | 2, 7 |
| Unmeasured confounders, valid instrument exists | Instrumental Variables | 9 |
| Panel data, policy change | Difference-in-Differences | 10 |
| Single treated unit (country/state/market) | Synthetic Control | 11 |
