# Causal Inference Notes

Study notes following the O'Reilly causal inference curriculum — built for a data scientist / product analyst ramping up on causal methods.

Each chapter covers:
- Core concepts with intuition-first explanations
- Key formulas in math notation
- Real-world examples (tech, healthcare, economics, policy)
- Self-contained Python code examples
- Interview questions — technical Q&As and case studies

## Why Causal Inference?

Predictive ML answers: *"What will happen?"*
Causal inference answers: *"What will happen **if we intervene**?"*

The difference matters every time you make a product decision, evaluate a policy, or ask whether a feature *caused* a metric to move — not just correlated with it.

## Roadmap

```
Part 1 — Foundations
  Ch 1 · Introducing Causality
  Ch 2 · Causal Models and the Adjustment Formula
  Ch 3 · Applying Causal Inference

Part 2 — Observational Methods (confounders measured)
  Ch 4 · Matching Methods
  Ch 5 · Propensity Score Methods
  Ch 6 · Linear Regression for Causal Inference
  Ch 7 · Advanced DAGs — Identification and Do-Calculus

Part 3 — Advanced Methods (confounders unmeasured / panel data)
  Ch 8  · Double Machine Learning
  Ch 9  · Instrumental Variables
  Ch 10 · Difference-in-Differences
  Ch 11 · Synthetic Control
```

## Python Libraries Used

| Library | Chapters | Install |
|---------|----------|---------|
| `numpy`, `pandas`, `scipy`, `sklearn` | All | `pip install numpy pandas scipy scikit-learn` |
| `statsmodels` | 6, 9, 10 | `pip install statsmodels` |
| `matplotlib` | 4, 5, 8, 10, 11 | `pip install matplotlib` |
| `dowhy` | 3, 7 | `pip install dowhy` |
| `econml` | 5, 8 | `pip install econml` |
| `linearmodels` | 9, 10 | `pip install linearmodels` |
| `networkx` | 7 | `pip install networkx` |

---

*Notes by Annie Wang · O'Reilly Causal Inference course*
