#!/usr/bin/env python3

import numpy as np
from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC



def standardize_3D(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)



X = np.array([[[-1, -1], [1,1]], 
              [[-2, -1], [-1, 2]], 
              [[1, 1], [2, 1]]
              ])
y = np.array([1, 1, 2])

X = standardize_3D(X)



clf_1 = LinearSVC()
clf_1.fit(X, y)


print(clf_1.coef_)

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])


# clf_2 = make_pipeline(StandardScaler(), SVC(kernel="linear"))
# clf_2.fit(X, y)