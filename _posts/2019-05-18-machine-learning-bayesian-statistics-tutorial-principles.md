---
title: Machine Learning Tutorial on Bayesian Statistics - Concepts and Terms
author: Amine
layout: post
math : true
---
#### Bayesian Optimization (BO)

A strategy for global optimization of a black-box function (i.e. unknown). BO helps to find a best model among many. BO does not require derivatives because the objective function is unknown. Use a prior which captures our beliefs about the behavior of the objective function. Use function evaluations in order to update the prior to form the posterior distribution over the objective function. BO is suited for functions that are very expensive to evaluate. BO is applied within automatic machine learning toolboxes and architecture configuration in deep learning.

#### Gaussian Process (GP)

A GP defines a distribution over functions $p(f)$ which can be used for Bayesian regression. f is a function which maps from x to the set of real numbers R. A GP uses a *kernel function* to measure the similarity between points $x$, $x'$ to predict the value $y'$ for the unseen point $x'$ from the training data. The prediction $y'$ is not only an estimate for that point $x'$ but also contains information about uncertainty. The distribution of a GP is the joint distribution. Its marginal distribution is a multivariate Gaussian distribution. GPs are parameterized by a mean function $\mu(x)$ and a covariance function (i.e. kernel) $K(x,x')$. GP can be used for nonlinear regression or prediction/classification tasks.

#### Ensemble learning

Ensemble learning models combine the hypotheses of multiple models in order to improve the overall performance on a given task. *Ensemble* means that we generate multiple hypotheses using the same base learner (ML model) with different random initializations. For example, Bayesian model averaging (BMA) is an ensemble technique. Fast algorithm classes such as decision trees (e.g. Random Forest, AdaBoost) are commonly used as ensemble learning methods.

#### Empirical Bayes method

Empirical Bayes is a method for statistical inference. In contrast to standard Bayesian methods where the prior distribution is fixed before any data is observed, in Empirical Bayes methods the prior is estimated from the data. In a nutshell, we use the data to estimate the Bayesian prior.

#### Categorical Distribution and its conjugate prior the Dirichlet distribution

The Dirichlet distribution is a family of continuous multivariate probability distributions. It is a multivariate generalization of the **beta distribution**, hence its alternative name **multivariate beta distribution (MBD)**. Dirichlet distributions are frequently used as prior distributions in Bayesian statistics.

The Dirichlet distribution is the conjugate prior distribution of the categorical distribution (and the related multinomial distribution). Thus, if the data has either a categorical or multinomial distribution and the prior of the distribution's parameter (often $\theta$) is a Dirichlet distribution, then the posterior distribution (product of likelihood and prior) of the parameter follows a Dirichlet distribution. The Dirichlet distribution is commonly used as the prior distribution for the categorical random variables of a model.

#### Categorical distribution vs. multinomial distribution

The categorical distribution is a generalization of the Bernoulli distribution ($K=2$) because the number of possible outcomes $K>2$. The categorical distribution (sometimes called **multinoulli distribution**) describes the probability of a random variable (RV) that can take on one of $K$ different outcomes and relates to a single trial ($N=1$). In contrast, the multinomial distribution describes the probability after $N$ trials for a sequence of i.i.d RVs $X_1, ..., X_n$ each with a categorical distribution. In a nutshell the relationship between these distributions is as follows

Bernoulli ($N=1$) $\rightarrow$ Binomial ($N>1$)
Categorical ($N=1$) $\rightarrow$ Multinomial ($N>1$)

where $N$ is the number of trials.

#### Dirichlet-multinomial distribution

A family of of discrete multivariate probability distributions on a finite set of non-negative integers. First a probability vector $\bm{p}$ id drawn from a Dirichlet distribution with parameter $\bm{\alpha}$, and the categorical-valued observation is drawn from a multinomial distribution with probability vector $\bm{p}$ and number of trials $N$. The marginal joint distribution of the observations $P(x_1, ..., x_N$) (with the prior parameter $\theta$ marginalized out) is such a Dirichlet-multinomial distribution. For $N=1$ it reduces to the vanilla categorical distribution.

#### Conjugate prior

In Bayesian probability theory, if the posterior distributions $p(\theta|x)$ is from the same probability distribution family as the prior distribution $p(\theta)$, then the two distributions are conjugate distributions. The prior is called conjugate prior for the likelihood function $p(x|\theta)$. The conjugate prior may give intuition, by showing how a likelihood function (i.e. observed data) updates the prior distribution.

For example, the Gaussian distribution is conjugate to itself (*self-conjugate*) with respect to the Gaussian likelihood function. Thus, if the likelihood function is Gaussian, choosing a Gaussian prior will ensure that the posterior distribution is also Gaussian.

Below an overview of distributions and their conjugate prior.

**Dicrete distributions**

| Distribution  | Conjugate prior |
| ------------  | --------------- |
| Bernoulli     | Beta            |
| Binomial      | Beta            |
| Geometric     | Beta            |
| Categorical   | Dirichlet       |
| Multinomial   | Dirichlet       |

**Continuous distributions**

| Distribution  | Conjugate prior |
| ------------  | --------------- |
| Normal (with known variance $\sigma^{2}$) | Beta            |
| Multivariate normal (with known covariance matrix $\bm{\Sigma}$       | Multivariate Normal            |
| Uniform      | Pareto            |
| Log-normal   | Same as for normal distribution after exponentiating the data       |
| Exponential   | Gamma       |


#### Probability calibration of classifiers

When doing classification we want to predict a sample's class label as well as the probabilities that the sample belongs to any of the classes. This serves as a measure of uncertainty of our prediction. Often a separate calibration of the output probabilities is performed as a post-processing step. Ideally, the goal is to assign a high probability for a sample's ground truth class and small probability for all remaining class candidates.

#### Longitudinal data

A dataset is longitudinal if it tracks the same type of information on the *same subject* at multiple points in time. Thus, longitudinal data is a collection of repeated observations of the same subject/sample over a period of time. It is useful for measuring change. Longitudinal data differs from cross-sectional data because it tracks the same subject/sample over a period of time, while cross-sectional data refers to observations over a period of time but for *different* subjects/samples at each point in time.

#### Survival analysis

Branch of statistics for determining the expected duration of time until one or more events happen (*time-to-event*). For example, death of a human through an event such as heart attack or organ failure or defect of a mechanical machine. Questions we try to answer with survival analysis

* What proportion of a population will survive past a certain time $t_q$?

* Of those that survive, at what rate will they die or fail?

* Which particular circumstances increase or decrease the survival probability?

Survival analysis aims at modeling time-to-event data. Thus, death, failure, disease occurrence or recovery are such "events".

#### Bayesian Model Averaging (BMA)

The selection of a single prediction model may lead to overconfident inferences which leads to riskier decision making.
Bayesian Model Averaging is an application of Bayesian inference to the problems of model selection, combined estimation and prediction. Here, we combine multiple models in order to reduce prediction uncertainty. BMA does not only model parameter uncertainty through the prior distribution $p(\theta)$ but BMA also accounts for the model uncertainty.

#### Meta-Learning

A subfield of machine learning (ML) (aka *learning to learn*) where automatic learning algorithms are applied on metadata from previous ML experiments. Metadata is data, that provides information about other data such as statistical meta-features of a dataset (e.g. number of features, size of data, class imbalance, etc.).  The goal is to use metadata to make automatic learning more flexible and thus improving the  performance of a learning algorithm.

#### Nonparametric estimation

A statistical method that is not based solely on parameterized families of probability distributions (e.g. mean $\mu$ and variance $\sigma^{2}$ in case of a normal distribution).

The two types of nonparametric techniques architecture
* artificial neural networks (ANN)
* kernel estimation

In general we want to learn a function $f$ that maps input $X$ to output $Y$

$$Y = f(X)$$

ML methods that simplify the function $f$ to a known form (fixed number of parameters regardless of the size of the dataset X) are called *parametric* ML algorithms (e.g. logistic regression, perceptron, Naïve Bayes). Such an algorithm involves two steps:

1. Select a form for the function $f(X;\theta)$
2. Learn the coefficients $\theta$ for the function from the training data X.

In contrast to that, nonparametric machine learning do not make such strong assumptions about the form of the function $f(X)$. Thus, they are great to use when we have a lot of data and no prior knowledge about $f(X)$. An example is k-Nearest Neighbors which makes predictions for a new data sample based on the k most similar samples from the training data. Other popular nonparametric algorithms are Decision Trees and SVMs.

To make a long story short, nonparametric models differ from parametric models in that the model structure (referred to as $f$ above) is not specified *a priori* but is instead determined from data.

#### Terminology: "Being exponential in n"

Given a function $f=a^{2}be^{n}$. Considering this expression as a function of a single variable (either $a$, $b$ or $n$) we say that
* $f$ is linear in $c$,

* polynomial (i.e. quadratic) in $b$,

* and exponential in $n$.

#### Posterior probability distribution

The posterior probability  distribution is a conditional probability distribution that represents our updated belief about the unknown parameter $\theta$ which is treated as a RV, after having observed data (aka evidence). It is *always* the case that the argument of a distribution (or probability function) is a function of an unknown. The posterior probability is just the conditional probability that is outputted by the Bayes theorem. There is nothing special about it as it does not differ from any other conditional probability; it just has  its own name.

#### Gibbs sampler/sampling

Gibbs sampler is a Markov chain Monte Carlo (MCMC) algorithm for obtaining a sequence of observations $x_1, ..., x_n$  when direct sampling from the multivariate probability distribution is difficult. Subsequently, this sequence of observations can be used

* to approximate the joint distribution  (e.g. generating a histogram of the distribution)
* to approximate the marginal distribution of one of the random variables $X_i$, or some subset ${X_i}$ (e.g. the unknown parameters or latent variables)

* to compute an integral (e.g. the expected value of one of the random variables $\mathop{\mathbb{E}}[X_i]$)

Gibbs sampling is used when the joint distribution is not known explicitly or is hard to sample from directly **but!** the conditional distributions of each variable $X_i$ is known and easier to sample from.

Gibbs sampling generates an instance $x_1, ..., x_n$ from the distribution of each variable $X_i$ in turn, conditional on the current values of the other variables $x_1, ..., x_{i-1}, x_{i-1}, ..., x_n$. The stationary distribution of the Markov chain $\pi(\bm{x})$ is the sought-after joint distribution.

Gibbs sampling is a randomized algorithm *(i.e. makes use of random numbers)* and it is an alternative to deterministic algorithms such as the expectation-maximization algorithm (EM). Gibbs sampling generates a Markov chain of samples, each of which will be correlated to nearby (in time) samples.

Gibbs sampling is commonly applied for sampling from the posterior distribution of a Bayesian network (BN). This is because a BN is typically specified as a collection of conditional probabilities $P(X_i|X_{-i}) \; \forall \in \{1,...,n\}$  


#### Gumbel-Max trick

The Gumbel-Max trick is a method to draw samples $z$ from a categorical (discrete) distribution $\bm{Cat}(\alpha_1, ..., \alpha_K)$ with category probabilities $\bm{\alpha}$. In effect, where sampling from category $k$ has the probability $\alpha_k$ given $K$ different categories.

Let $Z$ be a discrete random variable (RV) with $P(Z=k) \propto \alpha_k $. The recipe for sampling from a categorical distribution is

1. For each category $k$ draw i.i.d. sample of Gumbel noise from the Gumbel distribution Gumbel(0,1) by transforming a uniform sample $u$
over $g=-log(-log(u))$. Here we sample $u$ from the Uniform distribution $u \sim Uniform(0,1)$ .
2. Add the sampled Gumbel noise to $log(\alpha_k)$
3. Take the index $k$ that maximizes the term $log(\alpha_k) + g_k$ as sample value for the categorical distribution

$$z = \underset{x}{\arg\max} [log(\alpha_k) + g_i] $$

The **Gumbel-Softmax** trick is a method that allows to draw samples from a categorical distribution but uses the $softmax$ function instead of the $\arg \max$ function as differentiable approximation to $\arg \max$.

#### Marginal likelihood

A likelihood function in which some parameter variables have been marginalized out. Thus is may also be referred to as *evidence*. Given a set of i.i.d. data points $\bm{X} = (x_1, ..., x_n)$, where $x_i \sim p(x_i|\theta)$ (a probability distribution parameterized by $\theta$ ), and where $\theta$ itself is a RV (i.e. has the distribution $\theta \sim p(\theta|\alpha)$), the marginal likelihood is the probability $p(\bm{X}|\alpha)$. Where the parameter variable $\theta$ has been marginalized over (integrated out) s.t.

$$p(\bm{X}|\alpha) = \int_{\theta} p(\bm{X}|\theta,\alpha) p(\theta|\alpha)d\theta$$
