## Structural Approach

$$P(R = 1|\text{do}(T = A)) = \sum_s P(R = 1|S = s, T = A)P(S = s)$$

$$E[R|\text{do}(T = A)] = \sum_s E[R|T = A, S = s]P(S = s)$$

$$\sum_k r_k P(R = r_k|\text{do}(T = A)) = \sum_k r_k \sum_s P(R = r_k|T = A, S = s)P(S = s)$$

$$\sum_s \sum_k r_k P(R = r_k|T = A, S = s)P(S = s) = \sum_s E[R|T = A, S = s]P(S = s)$$

### Average Treatment Effect

$$\text{ATE} = E[R|\text{do}(T = A)] - E[R|\text{do}(T = B)]$$

- If $\text{ATE} > 0$: pick treatment A
- Otherwise: pick treatment B

### Assumptions for Adjustment Formula

The adjustment formula implicitly requires an important assumption that we haven't talked about yet. Imagine an extreme case where treatment A has only been tested with large stones. This is unfortunate because we don't know how it will work with small stones. If something has not been tried, we don't have any information about it. Mathematically speaking, we cannot use the term that measures the efficacy of treatment A in small stones $P(R = 1|T = A, S = \text{small})$ because the event $(T = A, S = \text{small})$ has never happened: that is, $P(T = A|S = \text{small}) = 0$.

$$0 < P_0(T = t|S = s) < 1$$

$$0 < P_0(T = t|S_1 = s_1, \ldots, S_p = s_p) < 1$$

information. This may happen when a treatment is purposely not given to a particular subset of patients. For instance, physicians may decide not to give a treatment to older people because it entails a higher risk. Whenever the problem is the design of the assignment policy, increasing the sample size will not solve it: there will always be a combination of confounders $S = s$ and treatment $T = t$ that will never happen, which is mathematically expressed in terms of the data-generation distribution as $P_0(T = t|S = s) = 0$.

In practice, depending on the type of problem you are dealing with, you will pay more or less attention to this assumption. For instance, in healthcare, when comparing different treatments, not having data about one treatment in a particular subgroup of the population is a problem that has to be dealt with (and we will talk about it in chapter 5).

### Interventions and RCTs (Randomized Controlled Tests)

Treatment assignment is randomized.

![Figure 2.7 Graph depicting an RCT. In an RCT, various factors may influence the outcome, but none influence the treatment, as the treatment is assigned randomly. This means there are no confounders.](image-7.png)

### Structural Approach

If you have a small number of confounding variables, you can apply the adjustment formula. No other factors matter here.

![Treatment → Recovery](image-8.png)

The variable $U$ represents a set of unknown factors unique to each patient—including things like age—that can affect the patient's recovery but are unrelated to the treatment. If we excluded the variable $U$ and wrote Recovery = f(Treatment), we would essentially be claiming that Recovery solely depends on Treatment. This would imply that all patients receiving the same treatment would have identical responses, which is not the case.

We use the symbol $:=$, known as the *assignment* or *walrus operator* (due to its resemblance to a walrus), much like in programming. When we write $Y := 2X$, it means once you know the value of $X$, you set the value of $Y$ to $2 \times X$. However, if you change the value of $Y$, the value of $X$ remains the same. In contrast, in a mathematical equation like $Y = 2X$, a change in the value of $X$ also changes the value of $Y$, and vice versa. This symbol may appear to be a minor detail, but it inherits the causal direction of $Y$.

$$T := U_0$$

![Figure 2.9 Graph modeling the dynamics of the kidney stone data](image-13.png)

$$S := U_S$$
$$T := g(S, U_T)$$
$$R := f(S, T, U_R)$$

Recovery rate:

|              | Treatment A       | Treatment B    |
|--------------|-------------------|----------------|
| Small stones | **93%** (81/87)   | 87% (234/270)  |
| Large stones | **73%** (192/263) | 63% (50/80)    |

*Table 2.3 Recovery rates by treatment and size*

|               | Treatment A     | Treatment B       |
|---------------|-----------------|-------------------|
| Recovery rate | 78% (273/350)   | **81%** (284/350) |

![Figure 2.21 Five steps describing the typical process in causal inference](image-17.png)

![Figure 2.22 This diagram outlines the approach for parts 1 and 2 of the book, focusing on applying the adjustment formula across various scenarios using different techniques.](image-18.png)

On the other hand, if your treatment variable is continuous, the adjustment formula explained in this chapter cannot be applied directly. In this case, you have two options. If you want a preliminary result that uses methods you can understand and that can be used as a baseline before further analysis, you should use linear methods (chapter 6). If instead you require an accurate method capable of handling nonlinear relationships, you should use double machine learning (chapter 8).
