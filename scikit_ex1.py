# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 15:46:02 2014

@author: Giacomo
"""

import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
from sklearn import svm
from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn import grid_search
import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, show

### import the data
# test set. 40 features, 9000 samples
X_test = pd.read_csv("data/test.csv", sep=',', header=None).as_matrix()
# training set. 40 features, 1000 samples
X_train = pd.read_csv("data/train.csv", sep=',', header=None).as_matrix()
# training labels
y = pd.read_csv("data/trainLabels.csv", sep=',', header=None)[0].as_matrix()

### scatter plot of training set
# alpha is the transparency level
df = pd.DataFrame(X_train[:, 0:5]) # example with 5 of the 40 variables
pd.tools.plotting.scatter_matrix(df, alpha=0.3, figsize=(8, 8), diagonal='hist')

#  from the scatterplots it appears that these 40 variables are mostly uncorrelated,
# so a PCA will not add much value to the analysis. However, classifiers perform worse
# with high dimensionality problems, so a PCA is still very useful. See here:
# http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/
# variables have almost the same variance, so there is no need of standardizing them before PCA.

# correlation matrix (correlation between 2 variables is very small)
R = np.corrcoef(X_train.transpose())
plt.figure(2)
pcolor(R)
colorbar()
plt.title("correlation matrix")
plt.savefig("correlation_matrix.png")
show()

### center the variables before performing PCA
# center to the mean, but DO NOT component wise scale to unit variance
# by centering the variables, principal components remain the same,
# by standardizing the variables, principal components change
X_train = pp.scale(X_train, with_mean=True, with_std=False)
X_test = pp.scale(X_test, with_mean=True, with_std=False)

### dimensionality reduction using PCA
# since data is uncorrelated and with variance almost equal to 1,
# whitening is not necessary
pca40 = PCA(n_components=40, whiten=False) 
pca40.fit(X_train)
print(pca40.explained_variance_ratio_)

# plot all the principal components with their relative explained variance
features = [x for x in range(1,41)]
plt.figure(3)
# percentage of variance explained by each of the selected components.
# The sum of explained variances is equal to 1.0
plt.plot(features, pca40.explained_variance_ratio_, 'g--', marker='o')
plt.axis([1, 40, 0, 0.3])
plt.grid(True)
plt.xlabel("principal components"), plt.ylabel("variance explained")
plt.title("scree plot")
plt.savefig("scree_plot.png")

# from the scree plot we choose to pick the first 12 principal components
pca12 = PCA(n_components=12, whiten=True) 
pca12.fit(X_train)

# apply dimensionality reduction to the training set and the test set
X_pca_train = pca12.transform(X_train)
X_pca_test = pca12.transform(X_test)

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
    probplot(x, dist="norm", plot=plt)

# kernel density plot and qq plot of the first principal component    
kde_plot(X_pca_train[:, 0])
qq_plot(X_pca_train[:,0])

### Cross Validation
# Stratified K-fold CV: all the folds have size trunc(n_samples / n_folds).
# Each fold contains roughly the same proportions of the two types of class labels.
# Provides train/test indices to split data in train test sets
skf = cv.StratifiedKFold(y, n_folds=3)

### Hyperparameter optimization using grid-search
# find C and gamma with a "grid-search" using cross validation
# Grid Search, first pass: coarse tuning of the parameters C and gamma
# C is a regularization parameter
# large C makes constraints hard to ignore -> narrow separation margin
C_range = 10.0 ** np.arange(6, 8) # 5, 10 is too low
# gamma is th width of the Radial Basis Function (RBF) kernel.
# gamma controls the shape of the separating hyperplane
gamma_range = 10.0 ** np.arange(-3, 0) # -5, 0 is too low
# Dictionary with parameters names as keys and lists of parameter settings to try as values
params = dict(gamma = gamma_range, C = C_range)
classifier = svm.SVC(kernel='rbf')
# SVM classifier optimized through cross validation
clf = grid_search.GridSearchCV(classifier, param_grid = params, cv = skf)
# fit the model after the reduced dimensionality performed by PCA
clf.fit(X_pca_train, y)
print("The best classifier is: ", clf.best_estimator_)

# Grid Search, second pass: fine tuning of the parameters C and gamma
C_range = 10.0 ** np.arange(6.5, 7.5, 0.1) 
gamma_range = 10.0 ** np.arange(-1.5, 0.5, 0.1) 
params = dict(gamma = gamma_range, C = C_range)
classifier = svm.SVC(kernel='rbf')
clf = grid_search.GridSearchCV(classifier, param_grid = params, cv = skf)
clf.fit(X_pca_train, y)
print("The best classifier is: ", clf.best_estimator_)

### Estimate score of the classifier
scores = cv.cross_val_score(clf.best_estimator_, X_pca_train, y, cv=10)
print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

### Predict and save
result = clf.best_estimator_.predict(X_pca_test)

f = open('result.csv','w')
# headers in the CSV file
f.write('Id,Solution\n') 
id = 1
for x in result:
    # id, label are numerical values (%d)
    f.write('%d,%d\n' % (id,x))
    id += 1
f.close()
