# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 15:46:02 2014

@author: Giacomo
"""

import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# test set. 40 features, 9000 samples
X_test = pd.read_csv("data/test.csv", sep=',', header=None).as_matrix()
# training set. 40 features, 1000 samples
X_train = pd.read_csv("data/train.csv", sep=',', header=None).as_matrix()
# training labels
y = pd.read_csv("data/trainLabels.csv", sep=',', header=None)[0].as_matrix()

### scatter plot of training set
from pandas.tools.plotting import scatter_matrix
# alpha is the transparency level
df1 = pd.DataFrame(X_train[:,0:5])
scatter_matrix(df1, alpha=0.3, figsize=(8, 8), diagonal='hist')

#df2 = pd.DataFrame(X_train[:,5:10])
#scatter_matrix(df2, alpha=0.3, figsize=(8, 8), diagonal='hist')
#
#df3 = pd.DataFrame(X_train[:,10:15])
#scatter_matrix(df3, alpha=0.3, figsize=(8, 8), diagonal='hist')
#
#df4 = pd.DataFrame(X_train[:,15:20])
#scatter_matrix(df4, alpha=0.3, figsize=(8, 8), diagonal='hist')
#
#df5 = pd.DataFrame(X_train[:,20:25])
#scatter_matrix(df5, alpha=0.3, figsize=(8, 8), diagonal='hist')
#
#df6 = pd.DataFrame(X_train[:,25:30])
#scatter_matrix(df6, alpha=0.3, figsize=(8, 8), diagonal='hist')
#
#df7 = pd.DataFrame(X_train[:,30:35])
#scatter_matrix(df7, alpha=0.3, figsize=(8, 8), diagonal='hist')
#
#df8 = pd.DataFrame(X_train[:,35:40])
#scatter_matrix(df8, alpha=0.3, figsize=(8, 8), diagonal='hist')

#  from the scatterplots it appears that these 40 variables are mostly uncorrelated,
# so a PCA will not add much value to the analysis. Maybe it is better to perform
# an ICA

# the correlation between 2 variables is very small
np.corrcoef(X_train[:,0], X_train[:,4])

# dimensionality reduction using PCA (with whitening)
# variance explained by all the 40 features (with whitening)
pca40 = PCA(n_components=40, whiten=True) 
pca40.fit(X_train)
print(pca40.explained_variance_ratio_)

# plot all the principal components with their relative explained variance
features = [x for x in range(1,41)]
plt.figure(1)
# percentage of variance explained by each of the selected components.
# The sum of explained variances is equal to 1.0
plt.plot(features, pca40.explained_variance_ratio_, 'g--', marker='o')
plt.axis([1, 40, 0, 0.3])
plt.grid(True)
plt.xlabel("principal components"), plt.ylabel("variance explained")
plt.title("scree plot")

# pick the first 12 principal components

# let's have a look at their distributions

# here, most of the variance is explained by the first 2 principal components
pca2 = PCA(n_components=2, whiten=True)

# since there are 9000 samples in the test set and only 1000 in the training set,
# it makes sense to use both in the unsupervised learning phases(PCA and GMM)
# NOTE: I should find confirmation of the legitimacy of this action
# concatenates along the first axis and fit the model with the stacked result
pca2.fit(np.r_[X_train, X_test])

# apply dimensionality reduction to the training set
X_pca = pca2.transform(X_train)

# find all objects of class 0 in trainLabels
i0 = np.argwhere(y == 0)[:, 0]
# find all objects of class 1 in trainLabels
i1 = np.argwhere(y == 1)[:, 0]

# save the subset of class 0 objects in a new variable
X0 = X_pca[i0, :]
# save the subset of class 1 objects in a new variable
X1 = X_pca[i1, :]
# plot the 2 subsets (class 0 objects in red, class 1 objects in blue)
plt.figure(2) 
plt.plot(X0[:, 0], X0[:, 1], 'ro')
plt.plot(X1[:, 0], X1[:, 1], 'b*')
# scatter plot of principal components
plt.scatter(X1[:, 0], X1[:, 1], marker="o")
plt.xlabel("PC1"), plt.ylabel("PC2")
plt.title("scatter plot of the variables in the training set")
# NOTE: as we can see from the plot, the 2 classes are NOT much separable

X_reduced = pca2.fit_transform(X_train)
X_reduced.shape

# Kernel Density Plot
def kde_plot(x):
        from scipy.stats.kde import gaussian_kde
        kde = gaussian_kde(x)
        positions = np.linspace(x.min(), x.max())
        smoothed = kde(positions)
        plt.figure()
        plt.plot(positions, smoothed)

# qq plot, to see if this variable follows a gaussian distribution
def qq_plot(x):
    from scipy.stats import probplot
    plt.figure()
    probplot(x, dist='norm', plot=plt)
    
X_all = pca40.fit_transform(np.r_[X_train, X_test])
kde_plot(X_all[:, 0])
qq_plot(X_all[:,0])


# Train a linear SVM
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(X_train, y)
predictions = clf.predict(X_test)
np.savetxt("linearSVMSubmission.csv", predictions.astype(int), fmt='%d', delimiter=",")

# I tried different models, but this one with c=10 and gamma=.01 gives
# gives the SVM benchmark score.
clf=svm.SVC(C=10.0,gamma=.01,kernel='rbf',probability=True)
clf.fit(X_train,y)
print clf.n_support_
y_pred1=clf.predict(X_test)

from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn import grid_search

# Scale data: center to the mean and component wise scale to unit variance
train = pp.scale(X_train)
test = pp.scale(X_test)

### Cross Validation
# Stratified K-fold CV: all the folds have size trunc(n_samples / n_folds)
# Provides train/test indices to split data in train test sets
skf = cv.StratifiedKFold(y, n_folds=3)

### Hyperparameter optimization using grid-search
# find C and gamma with a "grid-search" using cross validation
# C is a regularization parameter
# large C makes constraints hard to ignore -> narrow separation margin
C_range = 10.0 ** np.arange(7, 9) # 1, 10
# gamma is th width of the Radial Basis Function (RBF) kernel.
# gamma controls the shape of the separating hyperplane
gamma_range = 10.0 ** np.arange(-5, 0) # -5, 0
# Dictionary with parameters names as keys and lists of parameter settings to try as values
params = dict(gamma = gamma_range, C = C_range)

classifier = SVC()
# GridSearchCV parameters of the classifier used to predict is optimized by cross-validation
clf = grid_search.GridSearchCV(classifier, param_grid = params, cv = skf)
clf.fit(train, y)
print("The best classifier is: ", clf.best_estimator_)

# Estimate score
scores = cv.cross_val_score(clf.best_estimator_, train, y, cv=30)
print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

# Predict and save
result = clf.best_estimator_.predict(test)
np.savetxt("result.csv", result, fmt="%d")
