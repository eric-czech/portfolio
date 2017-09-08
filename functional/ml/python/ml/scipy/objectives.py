
import numpy as np
from py_utils.math import sigmoid

class ScipyObjective(object):

    def __init__(self, evaluate_fn, jacobian_fn):
        self.evaluate_fn = evaluate_fn
        self.jacobian_fn = jacobian_fn


def objective_mse_evaulate(p, X, y):
    return np.mean(np.square(y - np.dot(X, p)))


def objective_mse_jacobian(p, X, y):
    # Reference: http://theory.stanford.edu/~tim/s15/l/l15.pdf
    # See: "3.3 The Gradient of the MSE Function"
    # return (-2 / y.shape[0]) * np.squeeze(np.dot(X.T, y - np.dot(X, p)))
    return (-2. / y.shape[0]) * np.dot(X.T, y - np.dot(X, p))


# Mean squared error objective
# Notes:
# - See [1] for derivation of gradient ("3.3 The Gradient of the MSE Function")
#     [1] - http://theory.stanford.edu/~tim/s15/l/l15.pdf
OBJECTIVE_MSE = ScipyObjective(objective_mse_evaulate, objective_mse_jacobian)


def objective_logloss_evaulate(p, X, y):
    """ Mean log probability for logistic outcome """
    pl = np.dot(X, p)
    # Negative for minimization
    return -1. * np.mean(y * pl - np.log(1 + np.exp(pl)))


def objective_logloss_jacobian(p, X, y):
    """ Mean log probability gradient for logistic outcome """
    yp = sigmoid(np.dot(X, p))
    return (-1. / y.shape[0]) * np.dot(X.T, y - yp)

# Mean log-loss objective
# Notes:
# - See [1] and [2] for identical definitions of logloss gradient and objective.
#     [1] - https://courses.cs.washington.edu/courses/cse446/13sp/slides/logistic-regression-gradient.pdf
#     [2] - http://www.cs.cmu.edu/~tom/10601_fall2012/slides/GenDiscr_LR_9-20-2012.pdf
# - The mean log probability is used insted of the sum to avoid issues with "directional line search" errors
#   from SLSQP, which generally does not like large objective function values (and using the mean instead of sum
#   helps do that).
OBJECTIVE_MLL = ScipyObjective(objective_logloss_evaulate, objective_logloss_jacobian)
