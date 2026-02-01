# Causality Basics

## Correlation vs. Causation

One of the fundamental challenges in data science and statistics is distinguishing between correlation and causation.

**Correlation** means that two variables tend to change together, but this doesn't necessarily mean that one causes the other.

**Causation** means that changes in one variable directly produce changes in another.

## Common Pitfalls

### Confounding Variables

A confounding variable is a third variable that influences both the treatment and the outcome, creating a spurious association.

**Example**: Ice cream sales and drowning deaths are correlated, but this doesn't mean ice cream causes drowning. The confounding variable is temperature (summer weather increases both ice cream consumption and swimming).

### Reverse Causality

Sometimes the direction of causality is unclear. For example:
- Does poverty cause poor health, or does poor health cause poverty?
- The answer might be both!

## The Fundamental Problem of Causal Inference

We can never observe the same unit in both treated and untreated states at the same time. This is called the **fundamental problem of causal inference**.

For a person who received a treatment, we observe their outcome with treatment, but we can never observe what their outcome would have been without treatment (the counterfactual).

## Next Steps

In the following sections, we'll explore frameworks and methods for addressing these challenges:
- Potential outcomes framework
- Randomized controlled trials
- Observational methods for causal inference
