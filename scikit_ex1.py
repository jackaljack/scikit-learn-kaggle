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

# here, most of the variance is explained by the first 2 principal components
pca2 = PCA(n_components=2, whiten=True) 
# concatenates along the first axis and fit the model with the stacked result
pca2.fit(np.r_[X_train, X_test])
X_pca = pca2.transform(X_train)
i0 = np.argwhere(y == 0)[:, 0]
i1 = np.argwhere(y == 1)[:, 0]
X0 = X_pca[i0, :]
X1 = X_pca[i1, :]
plt.plot(X0[:, 0], X0[:, 1], 'ro')
plt.plot(X1[:, 0], X1[:, 1], 'b*')