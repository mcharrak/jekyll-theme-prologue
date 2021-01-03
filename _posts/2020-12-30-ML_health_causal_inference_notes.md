---
title: Causal Inference in Healthcare, Lecture Notes David Sontag
author: Amine M'Charrak
layout: post
date: 2020-12-30 4:00:00
---

### Notes Machine Learning for Causal inference in Healthcare
###### (resource/video/slides: [Lecture ML for Healthcare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/)).

---

*"Correlation does not imply causation."*

---

#### PART 1

#### Motivation:
Health system does not want to know how to **predict** diabetes - they want to know how to **PREVENT** it.
People respond differently to the same treatment - which is called **heterogeneity in treatment repsonse**.

* Example question: *"Does smoking causa lung cancer?"*
* Problem: A randomized control trial (RCT) is unethical.
* Idea: Using observational data to answer this question. Answering such questions from observational data is challenging because of **confounding** factors, that do both, cause people to be smokers but also cause them to receive lung cancer.

---

In causal inference for healthcare we have 3 variables, which as a triplet make up the dataset.

1. patient variables $X$, which includes all confounding factors
2. outcome variable $Y$, which could be the onset of a disease or the progression of it
3. interventions $T$, such as medication or treatment procedure (e.g., radiotherapy vs chemotherapy)

An intervention T (a.k.a treatment-plan) can be
* binary, i.e. choice between two treatments
* continuous, e.g, dosage
* a sequence of treatments over a time-interval

The causal graph is known. The only unknowns are (1) the **effect/strength** of the treatment $T$ on the outcome $Y$ and (2) for what patients $i$ does the treatment work the most/best.

All factors that influence the treatment decision are **confounding factors** but not all confounding factors are observed - there can be **hidden confounding factors**.

---

### Potential Outcomes Framework (a.k.a Rubin-Neyman Causal Model)

For each unit (individual) $x_i$ there are 2 outcomes:

1. "control outcome" $Y_0(x_i)$, had the unit not been treated.
2. "treated outcome" $Y_1(x_i)$, had the unit been treated

The Conditional Average Treatment Effect (CATE) for unit $x_i$ is defined as:

$$
CATE(x_i) =
\mathbb{E}_{Y_1 \sim p(Y_1|x_i)}[Y_1|x_i] -
\mathbb{E}_{Y_0 \sim p(Y_0|x_i)}[Y_0|x_i]
$$

and thus the Average Treatment Effect (ATE) is defined as:

$$
ATE := \mathbb{E}[Y_1 - Y_0] = \mathbb{E}_{x \sim p(x)}[CATE(x)]
$$

The fundamental problem of causal inference is that we only ever observe one of the multiple (i.e. two for binary treatment) outcomes.

If there is a hidden variable $h$ (i.e., confounder), which is not observed and which affects both, the treatment $T$ (which the individual receives) and the potential outcomes $Y_i$.

Confounding factor $h$:
A confounding factor is a variable that

* is unobserved in the observational dataset
* is affecting both, the treatment $T$ and the potential outcomes $Y$ (i.e., for the potential-outcomes-graph, there would be an edge from $h$ to $T$ and an edge from $h$ to each potential outcome $Y_i$).

**Typical assumptions**:

1. *Ignorability*: There are no unobserved confounding factors. Mathematically it means, that the potential outcomes $Y_0$ and $Y_1$ are conditionally independent of the treatment T, conditioned on covariates X:

$$(Y_0, Y_1) \perp \!\!\! \perp \ T \: | \; X$$

Ignorability can not be tested, based on the observed data.

2. *Overlap (common support)*: There always must be some stochasticity in the treatment decisions. Thus, we assume that the propensity score is always bounded between 0 and 1:

$$
p(T = t, X = x) > 0 \; \forall t,x
$$

Overlap can be assessed from the observed data.


The propensity score is the probability of receiving some treatment (for each individual).

---

#### PART 2

There are 2 common approaches for counterfactual Inference

1. Covariate adjustment, where we explicitly model the relationship between treatment $T$, confounders $X$, and outcome $Y$ as a function $f(X,T) \approx Y$.
2. Propensity scores.

**Reminder**: IGNORABILITY, is the assumption of no hidden confounding.

**Matching**:

The key idea of matching is to use each individual's *"twin"*, to get some intuition about what their potential (counterfactual) outcome might have been. This method works well in a large-sample-setting because where we are more likely to observe a counterfactual for every unit/individual in the data.

Visual example: In a 2D-plane with two groups of points, say red and blue points, we match each point with it's nearest neighbour from the opposite color (i.e., for a blue point we select the nearest red point, and vice versa). Then we can compare the outcomes for each pair to get an intuition about the potential/counterfactual outcome.

Mathematical example - 1-NN (nearest neighbour) Matching:

Let $d(\cdot,\cdot)$ be a metric between the covariate $X$ of two units/individuals. For each individual $i$, we define $j(i) = \underset{j \; s.t. \; t_j \neq t_i}{argmin} \; d(X_j,X_i)$. Then, $j(i)$ is the nearest counterfactual neighbour of $i$.
With this definition we can define the estimate of the conditional average treatment effect for any individual $X_i$ as follows:

* if $t_i=1$, unit $i$ is treated:
 $\widehat{CATE}(X_i) = y_i - y_{j(i)}$
* if $t_i=0$, unit $i$ is control:
 $\widehat{CATE}(X_i) = y_{j(i)} - y_i$

 We can combine the two cases to get the conditional average treatment effect for the binary treatment $t_i$ as:

 $$
 \widehat{CATE}(X_i) = (2t_i-1)(y_i - y_{j(i)})
 $$

 and the estimated average treatment effect ATE as:

 $$
\widehat{ATE} = \frac{1}{n}\sum_{i=1}^n \widehat{CATE(X_i)}
 $$

The advantages and disadvantages of matching are:

\+ interpretability, especially in small-sample regime.
\+ nonparametric, does not rely on any assumption about the parametric form of the potential outcomes $Y_i(X)$.
\- heavily reliant on the underlying metric $d(\cdot, \cdot)$.
\- could be misled by features which don't affect the outcome.

**Propensity scores**:

The key idea of propensity score methods, is to turn an observational study into something that looks like a randomised control trial (RCT) via/by re-weighting samples (i.e., data points).

The key challenge when working with data from an observational study is that there might be bias with regards to who receives treatment 0 ($t=0$) and who receives treatment 1 ($t=1$). This means that the probability of receiving a treatment is not uniform for all individuals i. For the case of binary treatments we have: $$p(X|t=0) \neq p(X|t=1)$$ which means, that the conditional distribution of $X$ given treatment $T=t$, $p(X|T=t)$, varies with T.

The goal of propensity score methods is to weight the conditional distributions such that the difference disappears. Meaning that we need to find the weights $w_{T}(x)$ for each treatment T such that we get the property:

$$
p(X|t=0) \cdot w_0(X) \approx p(X|t=1) \cdot w_1(X)
$$

The *propensity score* for a binary treatment is
$p(T=1|X)$, which is independent of the outcome $Y$. For example, in this case the propensity score could be computed with logistic regression.

**Propensity scores - algorithm**:

To compute the ATE from the dataset, $(x_1,t_1,y_1), \dots, (x_n,t_n,y_n)$, using propensity scores, we have to perform the following 2 steps:

1. Use any ML method to estimate the probability of a treatment $T=t$ given $x$: $\hat{p}(T=t|x)$ for every data point $x$
2. Then $$ \hat{ATE} = \frac{1}{n} \sum_{i \: s.t. \: t_i = 1} \frac{y_i}{\hat{p}(t_i = 1|x_i)} - \frac{1}{n} \sum_{i \: s.t. \: t_i = 0} \frac{y_i}{\hat{p}(t_i = 0|x_i)}$$ The inverse of the propensity score is the *weighting* that we referred to and denoted earlier as $w_T(x_i)$.

The derivation of this ATE estimate works as follows:

Under the assumption that the potential outcomes, $Y_1,Y_0$, are deterministic, they are no longer random variables, so we can define the ATE as:

$$
ATE = \mathbb{E}_{x \sim p(x)} [Y_1(x)] - \mathbb{E}_{x \sim p(x)} [Y_0(x)]
$$

where the expectation is over **all** individuals $x$ (because for the expectation we sample from $p(x)$), and not only the individuals that received the specific treatment.

So, we need to show that the first in $\hat{ATE}$ is an estimator of the first term in $ATE$ ( analogous for the second term).

From Bayes' rule, we know that

$$
p(x|T=1) = \frac{p(T=1|x) \cdot p(x)}{p(T=1)}
$$

so we can write

$$
p(x) = p(x|T=1) \cdot  \frac{p(T=1)}{p(T=1|x)}
$$

Therefore, we can re-write the first term of $ATE$ as follows

$$
\mathbb{E}_{x \sim p(x)} [Y_1(x)] = \mathbb{E}_{x \sim p(x|T=1)} \left[Y_1(x) \frac{p(T=1)}{p(T=1|x)} \right]
$$

because we know that the expectations can be written as


$$
\mathbb{E}_{x \sim p(x)} [Y_1(x)] = \int_X p(x) \cdot Y_1(x) \; dx = \int_{x|T=1} p(x|T=1) \cdot \frac{p(T=1)}{p(T=1|x)} \cdot Y_1(x) \; dx = \mathbb{E}_{x \sim p(x|T=1)} \left[ \frac{p(T=1)}{p(T=1|x)} \cdot Y_1(x) \right]
$$

Empirically, this expression can be approximated by

$$
\mathbb{E}_{x \sim p(x|T=1)} \left[ \frac{p(T=1)}{p(T=1|x)} \cdot Y_1(x) \right] \approx \frac{1}{n_1} \sum_{i \; s.t. \; t_i = 1} \left[ \frac{n_1/n}{\hat{p}(t_i=1|x_i)} \cdot y_i \right]
$$

where $n_1/n$ is the proportion of individuals that received treatment 1 and where $\hat{p}(t_i=1|x_i)$ is the propensity score of individual i for treatment 1. The second term for treatment 0 can be derived analogously.

The disadvantage of inverse propensity weighting (IPW) happens when the data lacks overlap (meaning that some treatments are very unlikely for a data point). As a consequence, these samples/data points will have propensity scores close to 0, which causes large variance (due to the inverse weighting) and errors in the estimation of the ATE. Moreover, with reduced overlap, the propensity scores become less informative.

**Conclusion**:

There are 2 methods to employ ML for causal inference:

1. **Covariate adjustment method**: Learn a model $f(X,T)$ that predicts the outcome $Y$ given covariate $X$ and treatment $T$; then use $f(X,T)$ to impute counterfactuals

2. **Propensity score method**: Learn the propensity scores (i.e. predict the probability of a treatment $T$ given covariate $X$). For example, for binary treatment we can use logistic regression to get the predictions. Re-weight outcomes or stratify the

Both methods give consistent estimates only if we have/use:

* correct causal graph - meaning no unobserved confounding
* overlap between treatment groups/classes - meaning treatments are balanced among groups
* nonparamemtric regression model (or if the specified model correctly models the outcome $Y$)

You can find the lectures for these notes here: [Lecture ML for Healthcare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/).
