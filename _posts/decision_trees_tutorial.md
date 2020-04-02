---
title: Decision Trees Tutorial
author: Amine M'Charrak
layout: post
date: 2020-03-10 4:00:00
---

# Decision Trees Tutorial

The top of the tree is called the "root node" as it is the root of the decision tree. The last layer of the tree consists of "leaf nodes" as they are the leaves of the tree. All nodes between root node and leaf nodes are called internal nodes
(or just simply nodes).

When we construct a decision tree, we have to compute the "impurity" of each independent predictor variable (IV) to determine which separation sequence within the tree is the best. The IV with the lowest impurity is chosen as next separation variable within the decision tree. A common tool to measure impurity is the **Gini** impurity. We repeat the Gini calculation at each node in our tree until we have included all IVs on the path from root to leaf node. If the Gini impurity of a child node is larger than the Gini impurity of its parent node, then the tree stops "splitting/growing" at this child node, and the child node becomes a leaf node.

Therefore, at every new newly expanded node, we have to determine if separating the data would result in an improvement (i.e. lower Gini value) or not.

In cases where our indepdent variable $X$ is a numeric variable we have to first sort our data according to the values of $X$. Then, we compute then mean $\bar{x}_{(i,i+1)}$ between two neighboring values $x_i$ and $x_{i+1}$. Finally, we use all computed means as thresholds to calculate the Gini impurity according to each $\bar{x}_{(i,i+1)}$. We will choose the threshold which leads to the smallest Gini score; then we divide our data into two groups, those with a value smaller than the threshold and those larger.

Depending on the type of outcome variable, there are two main types of decision trees:

1) *Classification tree*
2) *Regression tree*

A famous ensemble learning algorithm for classification and regression is the **Random Forest**, which constructs multiple decision trees to fit the data. The final outcome prediction is thus achieved by finding either
* the mean of all decision trees (in case of regression)
* the mode of all decision trees (most common predicted category) among all categories

## Random Forests

In order to build a Random Forest, we have to execute the following two steps multiple times (i.e. once for every bootstrapped dataset that we created):

1. create bootstrap dataset $\bar{D_{i}}$ of size $n$ from our original dataset $\mathcal{D}$ of size $N$. *Bootstrapping* relies on random sampling **with replacement**, which means our bootstrapped datasets $\bar{D_{i}}$ are allowed to contain duplicate observations from the original dataset $\mathcal{D}$.
2. for each bootstrapped datasets $\bar{D_{i}}$, we create a decision tree. At each node of the tree, when determining which IV we use to separate our data, we will only consider a random subset of size $k$ from all our remaining IVs (instead of considering all remaining IVs; as we did above for regular decision tree building).

Now that have created multiple, let's say $L$, different decision trees, we can use them to make more accurate and robust predictions than if we were to only build a single decision tree based on the full dataset $\mathcal{D}$. Therefore, to conclude our final prediction using the random forest of size $L$, we have to predict/infer the outcome variable for each decision tree. Now, in the case of a categorical outcome variable, all we have to do is to check which category received the most votes, and choose this majority vote as the final prediction of our random forest.

To sum up, above we have done two specific things:
1) we **b**-ootstrapped multiple datasets $\bar{D_{i}}$ to build decision trees
2) we **agg**-regated all predictions to retrieve a final prediction

Therefore, people refer to these two steps using the term **bagging**.

To measure how accurate the random forest is, we compute the predictions for each decision tree $DT_{i}$ on its "Out-Of-Bag" samples (i.e. the samples that did not make it into the bootstrapped dataset $\bar{D_{i}}$ which we used to construct each $DT_{i}$ in the first place). Then, the proportion of "Out-Of-Bag" samples which were *incorrectly* classified corresponds to the **"Out-Of-Bag-Error"** rate of the random forest.

Above, in step 2. of building the random forest, we chose to only look at a subset of size $k$ of our remaining independent variables (IVs). This parameter $k$, i.e. number of IVs used per step/node, determines the accuracy (or Out-Of-Bag-Error) of the Random Forest. Therefore, we can search for a "good" value of $k$ by constructing multiple Random Forests with different $k$ values and pick the Random Forest which yields the best performance.

Many statistical and machine learning libraries choose k to be the  square root of the number of IVs in our dataset $D\in \R^{N \times M}$, thus

$$ k = \sqrt{M}$$

Therefore, when building a random forest, there are two primary tuning parameters:

1. Number of Decision Trees (here: $L$)
2. Size of Random Set of Variables considered for each node (here: $k$)
