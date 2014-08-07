# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 15:46:02 2014

@author: Giacomo
"""

import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

# test set. 40 features, 9000 samples
X_test = pd.read_csv("data/test.csv", sep=',', header=None).as_matrix()
# training set. 40 features, 1000 samples
X_train = pd.read_csv("data/train.csv", sep=',', header=None).as_matrix()
# training labels
y = pd.read_csv("data/trainLabels.csv", sep=',', header=None)[0].as_matrix()

# dimensionality reduction using PCA (with whitening)
# variance explained by all the 40 features (with whitening)
pca40 = PCA(n_components=40, whiten=True) 
pca40.fit(X_train)
print(pca40.explained_variance_ratio_)

# plot all the principal components with their relative explained variance
features = [x for x in range(1,41)]
plt.figure(1) 
plt.plot(features, pca40.explained_variance_ratio_, 'g--', marker='o')
# maybe it's better to dicplay both a linear and a logaritmic scale
plt.axis([1, 40, 0, 0.3])
plt.grid(True)
plt.title('Principal Components')
plt.xlabel('principal components'), plt.ylabel('explained variance')

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
# NOTE: as we can see from the plot, the 2 classes are NOT much separable