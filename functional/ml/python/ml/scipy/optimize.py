# Notes:
# From http://www.scipy-lectures.org/advanced/mathematical_optimization/#a-review-of-the-different-optimizers
# - With no constraints:
#     - With or without knowledge of gradient, prefer BFGS or L-BFGS
# - With constraints:
#     - SLSQP - inequality and equality constraints
#     - COBYLA - inequality constraints only
#
# Examples of jacobian calculations for constraints:
# - https://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html


from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import BaseEstimator
from scipy import optimize
from collections import OrderedDict
from ml.scipy.objectives import *
from ml.scipy.constraints import *
import warnings


# class ScipyConstraint(object):
#
#     def get_lagrangian(self, param_value_map, param_index_map):
#         raise NotImplementedError('Constraint value in object function (lagrangian) not implemented')
#
#     def get_jacobian(self, param_value_map, param_index_map):
#         raise NotImplementedError('Constraint jacobian not implemented')


class ScipyParameters(object):

    def __init__(self, names, starts=None, bounds=None):

        if not isinstance(names, list):
            raise ValueError('Parameter names must be given as a list (given type = "{}")'.format(type(names)))
        param_val, param_cts = np.unique(names, return_counts=True)
        if len(param_cts) > 0 and param_cts.max() > 1:
            dupe_params = param_val[param_cts > 1]
            raise ValueError('Parameter name list contains duplicates (duplicate values = "{}")'.format(dupe_params))
        self.names = names
        self.p = len(names)

        if starts is not None and not isinstance(starts, list):
            raise ValueError('Parameter starts must be given as a list (given type = "{}")'.format(type(starts)))
        self.starts = [0. for _ in range(self.p)] if starts is None else starts
        if self.starts is not None:
            if len(self.starts) != self.p:
                raise ValueError(
                    'Starting values for parameters must have same length as parameter '
                    'names (n params = {}, n starting positions given = {})'
                    .format(self.p, len(self.starts))
                )

        if bounds is not None and not isinstance(bounds, list):
            raise ValueError('Parameter bounds must be given as a list (given type = "{}")'.format(type(bounds)))
        self.bounds = [(None, None) for _ in range(self.p)] if bounds is None else bounds
        if self.bounds is not None:
            if len(self.bounds) != self.p:
                raise ValueError(
                    'Bounds for parameters must have same length as parameter '
                    'names (n params = {}, n bounds given = {})'
                    .format(self.p, len(self.bounds))
                )

    def merge(self, other):
        return ScipyParameters(
            self.names + other.names,
            starts=self.starts + other.starts,
            bounds=self.bounds + other.bounds,
        )

    def get_parameter_index(self):
        return OrderedDict([(p, i) for i, p in enumerate(self.names)])

    def get_initial_values(self):
        return np.array(self.starts)

    def empty(self):
        return self.p == 0

    def index_of(self, parameter):
        if parameter not in self.names:
            raise ValueError('Parameter "{}" not found in parameter name list'.format(parameter))
        return self.names.index(parameter)

EMPTY_PARAMS = ScipyParameters([], starts=[], bounds=[])


class ScipyModel(object):

    def get_parameter_index(self):
        raise NotImplementedError('Method not yet implemented')

    def prepare_training_data(self, X, y):
        raise NotImplementedError('Method not yet implemented')

    def get_parameter_starts(self):
        raise NotImplementedError('Method not yet implemented')

    def get_parameter_bounds(self):
        raise NotImplementedError('Method not yet implemented')

    def evaulate_objective_fn(self, pv, pi, X, y):
        raise NotImplementedError('Method not yet implemented')

    def evaulate_jacobian_fn(self, pv, pi, X, y):
        raise NotImplementedError('Method not yet implemented')

    def evaluate_fit(self, fit):
        raise NotImplementedError('Method not yet implemented')

    def predict(self, fit, X):
        raise NotImplementedError('Method not yet implemented')


class ScipyLinearModel(ScipyModel):

    def __init__(self, builder):

        if not isinstance(builder.objective, ScipyObjective):
            raise ValueError('Objective must be of type "ScipyObjective" (type given = "{}")'
                             .format(type(builder.objective)))
        self.objective = builder.objective

        self.linear_params = builder.linear_params
        self.intercept_params = builder.intercept_params
        self.fit_params = builder.fit_params

        self.coef_params = self.intercept_params.merge(self.linear_params)
        self.all_params = self.coef_params.merge(self.fit_params)

    def get_parameter_index(self):
        return self.all_params.get_parameter_index()

    def get_parameter_starts(self):
        return self.all_params.starts

    def get_parameter_bounds(self):
        return self.all_params.bounds

    def prepare_training_data(self, X, _):
        if self.intercept_params.empty():
            return X
        bias = np.ones((len(X), 1))
        return np.hstack((bias, X))

    def predict(self, fit, X):
        X = self.prepare_training_data(X, None)
        return np.dot(X, fit.x)

    def evaulate_objective_fn(self, pv, pi, X, y):
        i = [pi[p] for p in self.coef_params.names]
        return self.objective.evaluate_fn(pv[i], X, y)

    def evaulate_jacobian_fn(self, pv, pi, X, y):
        i = [pi[p] for p in self.coef_params.names]
        return self.objective.jacobian_fn(pv[i], X, y)

    def evaluate_fit(self, fit):
        raise NotImplementedError('Method not yet implemented')


class ScipyLinearModelBuilder(object):

    def __init__(self):
        self.objective = None
        self.linear_params = None
        self.intercept_params = EMPTY_PARAMS
        self.fit_params = EMPTY_PARAMS

    def set_objective(self, objective):
        if not isinstance(objective, ScipyObjective):
            raise ValueError('Objective must be of type "ScipyObjective" (type given = "{}")'.format(type(objective)))
        self.objective = objective
        return self

    def add_intercept(self, start=None, bound=None):
        self.intercept_params = ScipyParameters(
            names=['Intercept'],
            starts=None if start is None else [start],
            bounds=None if bound is None else [bound]
        )
        return self

    def add_linear_params(self, names, starts=None, bounds=None):
        self.linear_params = ScipyParameters(
            names=names,
            starts=starts,
            bounds=bounds
        )
        return self

    def add_fit_params(self, names, starts=None, bounds=None):
        self.fit_params = ScipyParameters(
            names=names,
            starts=starts,
            bounds=bounds
        )
        return self

    def build(self):
        if self.objective is None:
            raise ValueError('Objective for linear model must be set')
        if self.linear_params is None:
            raise ValueError('Linear parameters must be specified before building full parameter set')
        if self.intercept_params is None:
            raise ValueError('Intercept params must be set (to EMPTY_PARAMS if not desired)')
        if self.fit_params is None:
            raise ValueError('Fit params must be set (to EMPTY_PARAMS if not desired)')
        return ScipyLinearModel(self)


def get_fit_summary(fit):
    if fit.success:
        preamble = 'Optimization converged successfully:'
    else:
        preamble = 'Optimization failed to converge:'
    summary = """
    Success: {}
    Status Code: {}
    Message: {}
    Number of iterations: {}
    Number of function evaluations: {}
    Objective Function Value: {}
    """.format(fit.success, fit.status, fit.message, fit.nit, fit.nfev, fit.fun)
    return preamble + '\n' + summary


class ScipyRegressor(BaseEstimator):
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
    def __init__(self, model, constraints=None, analytical_gradients=True, raise_on_failure=True):
        self.model = model
        self.constraints = constraints
        self.analytical_gradients = analytical_gradients
        self.raise_on_failure = raise_on_failure
        self.parameter_index = self.model.get_parameter_index()

    def _constraints(self):
        if self.constraints is None:
            return None

        cons = []
        for con in self.constraints.get_constraints():
            if 'args' in con:
                raise ValueError('Constraint cannot include "args" key')
            if self.analytical_gradients and 'jac' not in con:
                raise ValueError('Jacobian for constraints must be supplied when using `analytical_gradients=True`')
            if not self.analytical_gradients and 'jac' in con:
                del con['jac']
            con['args'] = (self.parameter_index,)
            cons.append(con)
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

        if y.ndim != 1:
            raise ValueError('At the moment, only 1-d outcome values are supported '
                             '(this assumption is built into objective function definitions)')
        X = self.model.prepare_training_data(X, y)

        # Run constrained optimization
        self.fit_ = optimize.minimize(
            fun=self.model.evaulate_objective_fn,
            x0=self.model.get_parameter_starts(),
            jac=self.model.evaulate_jacobian_fn if self.analytical_gradients else None,
            args=(self.parameter_index, X, y),
            bounds=self.model.get_parameter_bounds(),
            method='SLSQP',
            constraints=self._constraints()
        )

        if not self.fit_.success:
            msg = get_fit_summary(self.fit_)
            if self.raise_on_failure:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        return self

    def get_fit_summary(self):
        return get_fit_summary(self.fit_)

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
        y : array-like, shape = [n_samples]
            Predicted values
        """
        check_is_fitted(self, 'fit_')
        X = check_array(X)
        return self.model.predict(self.fit_, X)



