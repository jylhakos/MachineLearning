# Machine Learning

Machine learning algorithms aim to learn and improve their accuracy as they process more datasets.

## Supervised Learning

Supervised learning uses algorithms to train a model to find patterns in a dataset with labels and features and then uses the trained model to predict the labels on a new dataset’s features.

A large amount of labeled training datasets are provided which provide examples of the data that the computer will be processing.

Supervised learning tasks can be categorized as classification or regression problems.

## Ensemble learning

Ensemble learning algorithms combine multiple machine learning algorithms to obtain a better model.

## Decision trees

Decision tree learning is a machine learning approach that processes inputs using a series of classifications or regressions which lead to an answer or a continuous value.

Decision trees create a model that predicts the label by evaluating a tree of if-then-else true/false feature questions, and estimating the minimum number of questions needed to assess the probability of making a correct decision. 

Decision trees can be used for classification to predict a category, or regression to predict a continuous numeric value.

## Gradient Boosting

The term Gradient Boosting comes from the idea of boosting or improving a single weak model by combining it with a number of other weak models in order to generate a collectively strong model.

Gradient Boosting is a functional gradient algorithm that repeatedly selects a function that leads in the direction of a weak hypothesis so that it can minimize a loss function. 

Gradient Boosting can be used for classification and regression problems.

Gradient Boosting classifier combines several weak learning models to produce a predicting model.

Gradient Boosting approach trains learners based upon minimising the loss function of a learner.

The learners have equal weights in the Gradient Boosting. 

Early stopping support in Gradient Boosting enables us to find the least number of iterations which is sufficient to build a model that generalizes well to unseen data.

###  Classification

***Loss Function***

The loss function's purpose is to calculate how well the model predicts, given the available data.

***Weak Learner***

A weak learner classifies the data, but it makes a lot of mistakes.

***Additive Model***

The predictions are combined in an additive manner, where the addition of each base model improves (or boosts) the overall model. 

This is how the trees are added incrementally, iteratively, and sequentially. 

## XGBoost

Extreme Gradient Boosting (XGBoost) is implementation of gradient boosting.

With XGBoost, trees are built in parallel, instead of sequentially like Gradient Boosted Decision Trees (GBDT). 