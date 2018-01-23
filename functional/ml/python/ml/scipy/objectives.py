
import numpy as np
from py_utils.math import sigmoid


class ScipyObjective(object):

    def __init__(self, evaluate_fn, jacobian_fn):
        self.evaluate_fn = evaluate_fn
        self.jacobian_fn = jacobian_fn


def objective_mse_evaulate(p, X, y, w):
    if w is None:
        return np.mean(np.square(y - np.dot(X, p)))
    else:
        return np.mean(w * np.square(y - np.dot(X, p)))


def objective_mse_jacobian(p, X, y, w):
    if w is None:
        return (-2. / y.shape[0]) * np.dot(X.T, y - np.dot(X, p))
    else:
        return (-2. / y.shape[0]) * np.dot(X.T * w, y - np.dot(X, p))


# Mean squared error objective
# Notes:
# - See [1] for derivation of gradient ("3.3 The Gradient of the MSE Function")
# - See [2] and [3] and [4] for least squares objective function with weights (especially 4)
# - See [5] for jacobian of weighted least squares problem
#     [1] - http://theory.stanford.edu/~tim/s15/l/l15.pdf
#     [2] - http://www.itl.nist.gov/div898/handbook/pmd/section4/pmd432.htm
#     [3] - https://arxiv.org/pdf/1310.5715.pdf
#     [4] - https://www.mathworks.com/help/curvefit/least-squares-fitting.html#bq_5kr9-3
#     [5] - https://stats.stackexchange.com/questions/237235/how-do-you-derive-the-gradient-for-weighted-least-squares
OBJECTIVE_MSE = ScipyObjective(objective_mse_evaulate, objective_mse_jacobian)


def objective_logloss_evaulate(p, X, y, w):
    """ Mean log probability for logistic outcome """
    pl = np.dot(X, p)
    ov = y * pl - np.log(1 + np.exp(pl))

    # Negative for minimization
    if w is None:
        return -1. * np.mean(ov)
    else:
        return -1. * np.mean(w * ov)


def objective_logloss_jacobian(p, X, y, w):
    """ Mean log probability gradient for logistic outcome """
    yp = sigmoid(np.dot(X, p))

    if w is None:
        return (-1. / y.shape[0]) * np.dot(X.T, y - yp)
    else:
        return (-1. / y.shape[0]) * np.dot(X.T * w, y - yp)

# Mean log-loss objective
# Notes:
# - See [1] and [2] for identical definitions of logloss gradient and objective.
#     [1] - https://courses.cs.washington.edu/courses/cse446/13sp/slides/logistic-regression-gradient.pdf
#     [2] - http://www.cs.cmu.edu/~tom/10601_fall2012/slides/GenDiscr_LR_9-20-2012.pdf
# - The mean log probability is used insted of the sum to avoid issues with "directional line search" errors
#   from SLSQP, which generally does not like large objective function values (and using the mean instead of sum
#   helps do that).
#
# Notes on how to do this with weights:
# - http://article.sciencepublishinggroup.com/html/10.11648.j.sjams.20150306.13.html#paper-content-1-2

OBJECTIVE_MLL = ScipyObjective(objective_logloss_evaulate, objective_logloss_jacobian)


######################
# Poisson Objectives #
######################

def objective_poisson_evaulate(p, X, y, w, max_link=512):
    """ Mean log probability for poisson outcome """
    if w is not None:
        raise ValueError('Sample weights not supported for this model')
    from scipy.stats import poisson
    yp = np.exp(np.clip(np.dot(X, p), -max_link, max_link))
    return -np.mean(poisson.logpmf(y, yp))

    # from scipy.special import gammaln
    # yp = np.clip(np.dot(X, p), -np.inf, max_link)
    # return -np.mean(-np.exp(yp) + y * yp - gammaln(y + 1))


def objective_poisson_jacobian(p, X, y, w, max_link=512):
    """ Mean log probability gradient for poisson outcome """
    if w is not None:
        raise ValueError('Sample weights not supported for this model')
    yp = np.exp(np.clip(np.dot(X, p), -max_link, max_link))
    return -(1./y.shape[0]) * np.dot(y - yp, X)


# Mean poisson log probability objective
# Notes:
# - See https://www.cs.princeton.edu/~bee/courses/lec/lec_jan29.pdf
# - https://github.com/statsmodels/statsmodels/blob/master/statsmodels/discrete/discrete_model.py
OBJECTIVE_PML = ScipyObjective(objective_poisson_evaulate, objective_poisson_jacobian)


######################
# Ordinal Objectives #
######################

# def objective_oml_evaluate_old(n_classes, p_out, p_lin, X, y, epsilon=1e-128):
#     """
#     Mean log probability for ordinal classification task
#
#     References:
#         - http://fa.bianp.net/blog/2013/logistic-ordinal-regression/
#         - https://github.com/fabianp/minirank/blob/master/minirank/logistic.py
#
#     :param n_classes: Number of outcome classes
#     :param p_out: Outcome intercept parameters [n_classes - 1]
#     :param p_lin: Linear parameters
#     :param X: Covariates [n_samples, n_features]
#     :param y: Ordinal categorical response ([n_samples] of n_classes unique values)
#     :param epsilon: Small value below which differences of predicted probabilities are considered to be 0
#     :return:
#     """
#     from py_utils import math
#     assert np.all(np.in1d(y, np.arange(1, n_classes + 1)))
#
#     n = len(y)
#
#     if np.any(np.diff(p_out) < 0):
#         return np.inf
#
#     lp = 0
#     for i in range(n):
#         k = y[i]
#         Xi = X[i]
#
#         s0 = 1 if k == n_classes else math.sigmoid(np.dot(Xi, p_lin) + p_out[k-1], clip=True)
#         s1 = 0 if k == 1 else math.sigmoid(np.dot(Xi, p_lin) + p_out[k-2], clip=True)
#         assert 0 <= s0 <= 1
#         assert 0 <= s1 <= 1
#         p = s0 - s1
#         assert p >= 0
#         if p < epsilon:
#             return np.inf
#         else:
#             lp += np.log(p)
#
#     # print('params: ', pv)
#     # print('logprob: ', -lp)
#     return -lp / n
#
#
# def objective_oml_jacobian_old(n_classes, p_out, p_lin, X, y, epsilon=1e-128):
#     """
#     Mean log probability gradient for ordinal classification task
#
#     References:
#         - http://fa.bianp.net/blog/2013/logistic-ordinal-regression/
#         - https://github.com/fabianp/minirank/blob/master/minirank/logistic.py
#
#     :param n_classes: Number of outcome classes
#     :param p_out: Outcome intercept parameters [n_classes - 1]
#     :param p_lin: Linear parameters
#     :param X: Covariates [n_samples, n_features]
#     :param y: Ordinal categorical response ([n_samples] of n_classes unique values)
#     :param epsilon: Small value below which differences of predicted probabilities are considered to be 0
#     :return:
#     """
#     from py_utils import math
#     assert np.all(np.in1d(y, np.arange(1, n_classes + 1)))
#
#     n = len(y)
#
#     # Initialize gradient vectors to accmulate in loop
#     gb = np.zeros(X.shape[1])
#     ga = np.zeros(n_classes - 1)
#
#     gnan = np.repeat(np.nan, len(gb) + len(ga))
#
#     for i in range(n):
#         k = y[i]
#         Xi = X[i]
#
#         # Helper values for gradient calculation
#         s0 = 1 if k == n_classes else math.sigmoid(np.dot(Xi, p_lin) + p_out[k-1], clip=True)
#         s1 = 0 if k == 1 else math.sigmoid(np.dot(Xi, p_lin) + p_out[k-2], clip=True)
#         spq0 = s0 * (1 - s0)
#         spq1 = s1 * (1 - s1)
#
#         if (s0 - s1) < epsilon:
#             return gnan
#
#         # Accumulate gradient components
#         if k < n_classes:
#             ga[k-1] += spq0 / (s0 - s1)
#         if k > 1:
#             ga[k-2] -= spq1 / (s0 - s1)
#         gb += (Xi * (spq0 - spq1)) / (s0 - s1)
#
#     j = np.concatenate((gb, ga)) / n
#     # print('jac', j)
#     return -j
#
#
# OBJECTIVE_OML_OLD = ScipyObjective(objective_oml_evaluate_old, objective_oml_jacobian_old)


def objective_oml_evaluate(n_classes, p_out, p_lin, X, y, epsilon=1e-128):
    """
    Mean log probability for ordinal classification task

    References:
        - http://fa.bianp.net/blog/2013/logistic-ordinal-regression/
        - https://github.com/fabianp/minirank/blob/master/minirank/logistic.py

    :param n_classes: Number of outcome classes
    :param p_out: Outcome intercept parameters [n_classes - 1]
    :param p_lin: Linear parameters
    :param X: Covariates [n_samples, n_features]
    :param y: Ordinal categorical response ([n_samples] of n_classes unique values)
    :param epsilon: Small value below which differences of predicted probabilities are considered to be 0
    :return:
    """
    from py_utils import math

    n = len(y)

    if np.any(np.diff(p_out) < 0):
        return np.inf

    v_int = np.tile(p_out, (n, 1))
    v_lin = np.dot(X, p_lin)[:, np.newaxis]

    # Compute cumulative probability of each outcome
    y_proba = math.sigmoid(v_int + v_lin, clip=True)

    # Difference cumulative probabilities to get probability for each outcome
    y_proba = np.hstack((np.zeros((n, 1)), y_proba, np.ones((n, 1))))
    y_proba = np.diff(y_proba, axis=1)

    # Select probability associated with each observed outcome
    # * In other words, select one value in each row of the y_proba matrix corresponding to outcome index (in columns)
    y_proba = y_proba[np.arange(n), y - 1]
    assert y_proba.ndim == 1

    # Return infinite objective function value if any observations are so unlikely that
    # taking the log of the probability will be problematic
    if np.any(y_proba < epsilon):
        return np.inf

    # Return negative mean log probability of each observation
    return -np.mean(np.log(y_proba))


def objective_oml_jacobian(n_classes, p_out, p_lin, X, y, epsilon=1e-128):
    """
    Mean log probability gradient for ordinal classification task

    References:
        - http://fa.bianp.net/blog/2013/logistic-ordinal-regression/
        - https://github.com/fabianp/minirank/blob/master/minirank/logistic.py

    :param n_classes: Number of outcome classes
    :param p_out: Outcome intercept parameters [n_classes - 1]
    :param p_lin: Linear parameters
    :param X: Covariates [n_samples, n_features]
    :param y: Ordinal categorical response ([n_samples] of n_classes unique values)
    :param epsilon: Small value below which differences of predicted probabilities are considered to be 0
    :return:
    """
    from py_utils import math
    assert np.all(np.in1d(y, np.arange(1, n_classes + 1)))

    n = len(y)
    idx = np.arange(n)

    # Compute cumulative probability of each outcome by repeating outcome intercepts and adding them
    # to the linear predictor (dot product of ind. vars and linear parameters)
    v_out = np.tile(p_out, (n, 1))
    v_lin = np.dot(X, p_lin)[:, np.newaxis]
    y_proba = math.sigmoid(v_out + v_lin, clip=True)

    # Add 0 and 1 vectors to cumulative probabilities
    y_proba = np.hstack((np.zeros((n, 1)), y_proba, np.ones((n, 1))))

    # Select left and right cumulative probabilities active for each observation (based on actual outcome)
    ypr = y_proba[idx, y]
    ypl = y_proba[idx, y-1]
    ypd = ypr - ypl
    assert ypr.ndim == ypl.ndim == 1

    # If any differences between cumulative probabilities are sufficiently close to 0, return
    # a nan gradient instead of hitting division by zero errors/overflow below
    if np.any(ypd < epsilon):
        return np.repeat(np.nan, len(p_lin) + len(p_out))

    # Calculate gradient for linear parameters
    g_lin = np.sum(X * (1 - ypr - ypl)[:, np.newaxis], axis=0)

    # Compute gradient components for ordinal outcome intercepts noting that if there c classes
    # then there are c - 1 parameters being optimized but c + 1 cumulative probabilities computed (probabilities
    # for parameters surrounded by 0 and 1) -- this means that y, which contains integers in [1, c] are ok
    # to use directly as an indexer, as is y - 1 (but the gradient components for the 0 and 1 columns should be
    # dropped in the result)
    g_out = np.zeros_like(y_proba)
    g_out[idx, y] = (ypr * (1 - ypr)) / ypd
    g_out[idx, y - 1] = -(ypl * (1 - ypl)) / ypd
    g_out = np.sum(g_out, axis=0)[1:-1]  # Ignore first and last gradient components

    return -np.concatenate((g_lin, g_out)) / n


OBJECTIVE_OML = ScipyObjective(objective_oml_evaluate, objective_oml_jacobian)
