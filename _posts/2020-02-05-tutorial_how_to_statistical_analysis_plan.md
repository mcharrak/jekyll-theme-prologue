---
title: How to write a statistical analysis plan for research projects
author: Amine M'Charrak
layout: post
date: 2020-02-05 4:00:00
---

# How to write a statistical analysis plan for projects and research papers when working with a complex, real dataset using statistical modeling and machine learning methods?

If you have a complex dataset (e.g. observational study or a randomized controlled trial (RCT) from experiments) and a few interesting research questions, problems, and ideas; it becomes important to write a statistical analysis plan before you start fitting and testing your machine learning and statistical models. This is because it is difficult to work on a research project from start to end, if the research scope, aims, and potential pitfalls are not clear or undefined.

Below, I have composed a short recipe for how to write a concise statistical analysis plan, which will be helpful towards starting a research project or writing a scientific research paper.

Statistical analysis structure:

1. What are the aims of your research project?

    + For each aim, provide a set of hypotheses (which you will later on investigate with statistical methods)
2. Why is the research topic significant?

    + Highlight the importance, value, and relevancy using motivating problem cases and scenarios.
3. What is your analysis approach?

    + How do you get your data, what is the study design, and most of all what are the characteristics of your dataset population (e.g. which samples are included, and which sample are being excluded due to variable constraints). A tree structure is a good way to describe how your final dataset is constructed from the raw dataset, visually and level by level you can explain what variables are being filtered and how your dataset size shrinks by adjusting to variable constraints.
    + Describe your independent variables (IVs) (i.e. *features*). Which ones do you consider, how do you collect and measure them?
    + Describe your dependent variables (DVs) (i.e. *labels*). Which ones do you consider, how do you collect and measure them?
    + Identify and describe all possible confounding variables (i.e. *confounders*). Which ones to choose can be difficult but often it is helpful to review related literature to determine which confounding variables have been already studied and confirmed to have an influence on your independent outcome variables.
4. What is analysis plan?
    + Depending on the variable types (continuous, categorical (ordinal, nominal)), specify what statistical method you will employ for your research hypothesis (e.g. ANOVA, t-test). Below you can find a quick reference overview 2x2-table
    + Before using your statistical models, make sure that your selected variables fulfill the modeling assumptions (e.g. independent samples, data normality, homogeneity of variance)
    + Define follow up experiments to investigate how stable and robust your interpretations and associations are; for instance you can try to add another variable and see if your interpretation holds or if earlier found associations vanish
    + Clearly define the reference group for your comparisons between groups with different characteristics
    + Explain for which variables, that you know might have an effect on your outcome variable, you are adjusting in order to make sure that you draw correct associations and conclusion between variables from your dataset
    + How big is your sample size, and given that sample size, what is the statistical power of your method and at which significance level (α)
5. What are the strengths and limitations of your research plan?
    + if your research contains new ideas, explain how it fits into the current literature and what novelty it contributes to the literature
    + if your data is not perfect (highly likely) or if your measuring methods do not allow to correctly represent the truth, then point out such flaws and other too optimistic assumptions

| Independent Variable (X)        | Dependent Variable (Y)           | Model |
| :------------- |:-------------| :-----|
| **continuous**      | continuous | linear regression |
| **continuous**      | categorical | logistic regression |
| | | |
| **categorical**      | continuous | 2 samples: t-test (compare two means) <br> ≤ 3 samples: ANOVA (compare multiple means)|
| **categorical**      | categorical | Chi-squared (χ2) test |


That's it, now you are to start exploring your research ideas and run statistical experiments in a streamlined fashion. I hope this will help you to keep focused, while struggling with the data and unexpected results.
