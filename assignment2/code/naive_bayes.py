import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        p_xy = 0.5 * np.ones((d, k)) # an array of d arrays where each array is k length

        for c in range(k):
            indicies_c = np.array(y == c) # basically an array that is length of y but has 1 where the element equals c and zero otherwise
            X_c = X[indicies_c] # a matrix where all the examples are the examples where the label equals c
            p_xy[:, c] = np.mean(X_c, axis=0) 
            # this mean is basically finding the mean of an entire column in x_c 
            # (hence axis = 0), so this returns an array of length d
            # the array is inserted into p_xy as a column at index c.

        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        """YOUR CODE FOR Q3.4"""
        n, d = X.shape
        k = self.num_classes

        counts = np.bincount(y)
        p_y = (counts + self.beta) / (n + 2*self.beta)

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        p_xy = 0.5 * np.ones((d, k)) # an array of d arrays where each array is k length

        for c in range(k):
            indicies_c = np.array(y == c) # basically an array that is length of y but has 1 where the element equals c and zero otherwise
            X_c = X[indicies_c] # a matrix where all the examples are the examples where the label equals c
            p_xy[:, c] = (np.sum(X_c, axis=0) + self.beta) / (len(indicies_c) + 2*self.beta)
            # this mean is basically finding the mean of an entire column in x_c 
            # (hence axis = 0), so this returns an array of length d
            # the array is inserted into p_xy as a column at index c.


        self.p_y = p_y
        self.p_xy = p_xy
