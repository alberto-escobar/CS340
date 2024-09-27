from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """

    def __init__(self, num_trees, max_depth):
        self.trees = []
        self.num_trees = num_trees
        self.max_depth = max_depth
        for i in range(num_trees):
            self.trees.append(RandomTree(max_depth=max_depth))


    def fit(self, X, y):
        for randomTree in self.trees:
            randomTree.fit(X, y)


    def predict(self, X_pred):
        tree_y_hat = []
        for randomTree in self.trees:
            tree_y_hat.append(randomTree.predict(X_pred))
        tree_y_hat = np.array(tree_y_hat)
        y_hat = []
        for i in range(len(tree_y_hat[0])):
            y_hat.append(utils.mode(tree_y_hat[:,i]))
        return y_hat

