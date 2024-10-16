import numpy as np
from scipy.optimize.optimize import approx_fprime

from utils import ensure_1d

"""
Implementation of function objects.
Function objects encapsulate the behaviour of an objective function that we optimize.
Simply put, implement evaluate(w, X, y) to get the numerical values corresponding to:
f, the function value (scalar) and
g, the gradient (vector).

Function objects are used with optimizers to navigate the parameter space and
to find the optimal parameters (vector). See optimizers.py.
"""


class FunObj:
    """
    Function object for encapsulating evaluations of functions and gradients
    """

    def evaluate(self, w, X, y):
        """
        Evaluates the function AND its gradient w.r.t. w.
        Returns the numerical values based on the input.
        """
        raise NotImplementedError("This is a base class, don't call this")

    def check_correctness(self, w, X, y):
        n, d = X.shape
        w = ensure_1d(w)
        y = ensure_1d(y)

        estimated_gradient = approx_fprime(
            w,
            lambda w: self.evaluate(w, X, y)[0],
            epsilon=1e-6,
        )
        implemented_gradient = self.evaluate(w, X, y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print(
                "User and numerical derivatives differ: %s vs. %s"
                % (estimated_gradient, implemented_gradient)
            )
        else:
            print("User and numerical derivatives agree.")


class LeastSquaresLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is the sum of squared residuals.
        """
        # help avoid mistakes (as described in the assignment) by
        # potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        # Prediction is linear combination
        y_hat = X @ w
        # Residual is difference between ground truth and prediction
        # ("what's left" after your prediction)
        # These are "minus residuals"; slightly more convenient here.
        m_residuals = y_hat - y

        # Loss is sum of squared residuals
        f = 0.5 * np.sum(m_residuals ** 2)

        # The gradient, derived mathematically then implemented here
        g = X.T @ m_residuals  # X^T X w - X^T y

        return f, g


class RobustRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """
        # help avoid mistakes (as described in the assignment) by
        # potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        n, d = X.shape
        f = 0
        partial_fg = np.zeros(n)
        for i in range(n):
            residual = w.T@X[i,:] - y[i]
            f += np.log(np.exp(residual) + np.exp(-residual))
            partial_fg[i] = (np.exp(residual) - np.exp(-residual)) / (np.exp(residual) + np.exp(-residual))
            
        g = X.T@partial_fg
        return f, g
        
        
