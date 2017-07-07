
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import BaseEstimator
from scipy import optimize
import warnings


# def mapped_con_fn(fn):
#     def con_fn(w, m):
#         w = {k: w[i] for k, i in m.items()}
#         return fn(w)
#     return con_fn

def mapped_con_fn(fn):
    def con_fn(pv, pi):
        # `pi` contains name to index
        # `pv` contains value by index (as array)
        # `pm` will contain name to value
        pm = {k: pv[i] for k, i in pi.items()}
        return fn(pm, pi)
    return con_fn


def constraint_gte(p_hi, p_lo):
    """
    Create a scipy.optimize constraint where parameter named `p_hi` >= `p_lo`
    """
    def con_val(pm, pi):
        return pm[p_hi] - pm[p_lo]

    def con_jac(pm, pi):
        j = np.zeros(len(pi))
        j[pi[p_hi]] = 1.0
        j[pi[p_lo]] = -1.0
        return j

    return {'type': 'ineq', 'fun': mapped_con_fn(con_val), 'jac': mapped_con_fn(con_jac)}


def constraint_gtez(p):
    """
    Create a scipy.optimize constraint where parameter named `p` >= 0
    """
    def con_val(pm, pi):
        return pm[p]

    def con_jac(pm, pi):
        j = np.zeros(len(pi))
        j[pi[p]] = 1.0
        return j

    return {'type': 'ineq', 'fun': mapped_con_fn(con_val), 'jac': mapped_con_fn(con_jac)}


def constraint_ltez(p):
    """
    Create a scipy.optimize constraint where parameter named `p` <= 0
    """
    def con_val(pm, pi):
        return -1*pm[p]

    def con_jac(pm, pi):
        j = np.zeros(len(pi))
        j[pi[p]] = -1.0
        return j

    return {'type': 'ineq', 'fun': mapped_con_fn(con_val), 'jac': mapped_con_fn(con_jac)}


class RLSRegressor(BaseEstimator):
    """ Restricted least-squares regressor

    Example:

    from ml.sklearn import constrained_linear_model as clm

    # Fit a 3 variable LS model (w/ intercept) where each variable is gte to the ones before
    def gte(v1, v2):
        # Below, `pm` is parameter name to value; `pi` is parameter name to index
        return clm.mapped_con_fn(lambda pm, pi: pm[v1] - pm[v2])

    def gtz(v):
        return clm.mapped_con_fn(lambda pm, pi: pm[v1] - pm[v2])

    cons = [
        # Constrain X2 >= X1
        {'type': 'ineq', 'fun': mapped_con_fn(lambda w: w['X2'] - w['X1'])},

        # Constrain X1 >= X0
        {'type': 'ineq', 'fun': mapped_con_fn(lambda w: w['X1'] - w['X0'])},

        # Constrain X0 >= 0
        {'type': 'ineq', 'fun': mapped_con_fn(lambda w: w['X0'])}
    ]
    est = RLSRegressor(constraints=cons, fit_intercept=True, use_jacobian=False).fit(X)

    Resources:
        - https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp

    Parameters
    ----------
    constraints: List of constraints for scipy.optimize
    feature_names: List of feature names (optional) that can be used to refer to parameter values by name;
        All constraint functions should have the signature fn(w, m) where:
            w: Coefficient vector
            m: Dict of feature name to coefficient index
        Note that the helper method `mapped_con_fn` can be used to wrap a constraint so that the only input is w, a
        dictionary keyed by coefficient name and mapped directly to coefficient values (for convenience)
    fit_intercept: Whether or not to fit intercept term

    Attributes
    ----------
    optimize_result_ : Result of SLSQP optimization
    coef_ : array-like, shape = [n_features]
    intercept_ : Scalar intercept if `fit_intercept=True`

    """
    def __init__(self, constraints, feature_names=None, fit_intercept=True,
                 use_jacobian=True, raise_failure=False, bounds=None, start=None):
        self.constraints = constraints
        self.feature_names = feature_names
        self.fit_intercept = fit_intercept
        self.use_jacobian = use_jacobian
        self.raise_failure = raise_failure
        self.bounds = bounds
        self.start = start

        if feature_names is not None and 'Intercept' in feature_names:
            raise ValueError('Feature name list cannot contain reserved name "Intercept"')

        if self.bounds is not None and not isinstance(self.bounds, list):
            raise ValueError('Boundaries must be given as a list')

        if use_jacobian and (bounds is None or start is None):
            warnings.warn(
                'When using an analytical Jacobian, also giving bounds and a starting '
                'point for optimization is highly recommended'
            )

        for con in constraints:
            if 'args' in con:
                raise ValueError('Constraint cannot include "args" key')
            if use_jacobian and 'jac' not in con:
                raise ValueError('Jacobian for constraints must be supplied when using `use_jacobian=True`')

    def _objective(self, pv, X, y):
        return np.sum(np.square(y - np.dot(X, pv)))

    def _jacobian(self, pv, X, y):
        # Reference: http://global.oup.com/booksites/content/0199268010/samplesec3
        return -2 * np.squeeze(np.dot(X.T, y - np.dot(X, pv)))

    def _prep(self, X):
        if not self.fit_intercept:
            return X
        bias = np.ones((len(X), 1))
        return np.hstack((bias, X))

    def _constraints(self, pi):
        cons = []
        for con in self.constraints:
            c = dict(con)
            c['args'] = (pi,)
            if not self.use_jacobian:
                del c['jac']
            cons.append(c)
        return cons

    def fit(self, X, y):
        """
        Fit RLS Model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, accept_sparse=None, y_numeric=True, multi_output=False)

        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = ['X{}'.format(i) for i in range(X.shape[1])]

        X = self._prep(X)
        n_params = X.shape[1]

        if self.start is not None:
            if self.fit_intercept:
                pv0 = np.concatenate([np.array([0]), np.array(self.start)])
            else:
                pv0 = np.array(self.start)
        else:
            pv0 = np.zeros(n_params)
        if len(pv0) != n_params:
            raise ValueError(
                'Number of starting points ({}) must equal number of parameters ({})'
                .format(len(pv0), n_params)
            )

        if self.bounds is not None and self.fit_intercept:
            bounds = [(None, None)] + self.bounds
        else:
            bounds = self.bounds
        if bounds is not None and len(bounds) != n_params:
            raise ValueError(
                'Number of bounds ({}) must equal number of parameters ({})'
                .format(len(bounds), n_params)
            )

        # Initialize parameter name -> index map
        pi = {'Intercept': 0} if self.fit_intercept else {}
        offset = 1 if self.fit_intercept else 0
        for f in feature_names:
            pi[f] = feature_names.index(f) + offset

        # Run constrained optimization
        self.optimize_result_ = optimize.minimize(
            self._objective,
            pv0,
            jac=self._jacobian if self.use_jacobian else None,
            args=(X, y),
            bounds=bounds,
            method='SLSQP',
            constraints=self._constraints(pi)
        )

        if self.fit_intercept:
            self.coef_ = self.optimize_result_.x[1:]
            self.intercept_ = self.optimize_result_.x[0]
        else:
            self.coef_ = self.optimize_result_.x

        if not self.optimize_result_.success:
            msg = '[RLSRegressor] Optimization did not converge (status code = {}, message = {})'\
                .format(self.optimize_result_.status, self.optimize_result_.message)
            if self.raise_failure:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        return self

    def predict(self, X):
        """
        Predict new values

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        check_is_fitted(self, 'optimize_result_')
        X = check_array(X)
        X = self._prep(X)
        return np.dot(X, self.optimize_result_.x)
