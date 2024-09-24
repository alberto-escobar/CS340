"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q1"""
        d = euclidean_dist_squared(X_hat, self.X)
        y_hat = []
        for row in d:
            nn = np.argsort(row)[0:self.k]
            y = np.array(self.y)
            y = y_hat.append(utils.mode(y[nn]))
        return y_hat
            
            



