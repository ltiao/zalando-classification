# Zalando Coding Exercise

## Problem Description

We’re given a dataset with the following properties:

* **Size:** 1,000,000 (1 million) datapoints
* **Dimensionality:** 100
* **Classes:** 2 (binary classification)
* **Balance:** Perfectly balanced (50% positive, 50% negative)

### Hints

1. We’re told that the optimal classifier is around 90% accurate. This is possible to know if one has access to the underlying data generating process and can compute the densities `P(x | y=1)` and `P(x | y=0)`. The Bayes optimal classifier is then given by `P(y=1|x) = P(x | y=1) / [P(x | y=1) + P(x | y=0)]` (see https://tiao.io/post/density-ratio-estimation-for-kl-divergence-minimization-between-implicit-distributions/ for an example derivation).
2. We’re also told that the features are highly-dependent.

## Approach

Before diving into any sophisticated techniques, it’s always a good idea to try out some of the most naive methods one can think of to use as a baseline, like **Naive Bayes (NB)** or **k-Nearest Neighbors (k-NN)**. Thereafter, it’s reasonable to attack the problem with popular methods that have consistently dominated in data science competitions like Kaggle. Some of these include ensemble methods like **Random Forests **and** Gradient Boosting (XGBoost)**. Some other promising approaches include **Multi-layer Perceptrons (MLPs)**, **Support Vector Classification (SVC) **and **Gaussian Process Classification (GPC)**. 

### Preliminary results

For model evaluation, we first created a hold-out test set  consisting of 20% of the datapoints we were given. With the naive baselines (**NB, k-NN**), we hovered around 50% test accuracy.

In preliminary analysis (without making significant effort to tune the model hyperparameters), we consistently obtained at best around 55% test accuracy with **Random Forests**, **Gradient Boosting** and **SVC**.

In the last model (**GPC**), exact posterior inference is not tractable, especially given with size of the dataset. While approximate methods are available, it is left outside the scope of this solution set. I felt it deserved a mention nonetheless since it is of particular interest for this problem — it is capable of capturing compex dependencies between features (which is important according to Hint 2), and comes with a principled approach feature selection when Automatic Relevance Determination (ARD) kernels are used.

### **Multi-layer Perceptron**

Multi-layer Perception with two hidden layers with a “rectangular-shape” (number of hidden units are constant across all hidden layers).

*k*-fold cross-validation with *k=3*

since each model takes approximately 20 mins to train for 50 epochs, and we need to train a separate model for each fold, we limit k to a modest value k=3

the *solid lines* denotes the mean accuracy across folds while the *shaded error bands* denotes the standard deviation.

[Results](results-2.hires.png)

### Preprocessing

* Whitening

### **Feature Selection**

## Metrics

Cross-validation

* Accuracy
* ROC Curve
* Confusion matrix

## Credits

The boilerplace for this package was created with [Cookiecutter] and the 
[audreyr/cookiecutter-pypackage] project template.

[Cookiecutter]: https://github.com/audreyr/cookiecutter
[audreyr/cookiecutter-pypackage]: https://github.com/audreyr/cookiecutter-pypackage
