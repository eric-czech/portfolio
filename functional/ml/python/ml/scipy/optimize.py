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
from ml.scipy.constraints import *
import warnings


# class ScipyConstraint(object):
#
#     def get_lagrangian(self, param_value_map, param_index_map):
#         raise NotImplementedError('Constraint value in object function (lagrangian) not implemented')
#
#     def get_jacobian(self, param_value_map, param_index_map):
#         raise NotImplementedError('Constraint jacobian not implemented')


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
    def __init__(self, model, analytical_gradients=True, monitor_gradient=False,
                 raise_on_failure=True, seed=None, fit_kwargs=None,
                 gradient_diff_thresh=3):
        self.model = model
        self.analytical_gradients = analytical_gradients
        self.raise_on_failure = raise_on_failure
        self.monitor_gradient = monitor_gradient
        self.gradient_diff_thresh = gradient_diff_thresh
        self.constraints = self.model.get_parameter_constraints()
        self.seed = seed
        self.fit_kwargs = fit_kwargs

    def _constraints(self):
        if self.constraints is None:
            return []

        cons = []
        parameter_index = self.model.get_parameter_index()
        for constraint in self.constraints.get_constraints():
            # Copy to avoid side effects on mutation
            con = dict(constraint)
            if 'args' in con:
                raise ValueError('Constraint cannot include "args" key')
            if self.analytical_gradients and 'jac' not in con:
                raise ValueError('Jacobian for constraints must be supplied when using `analytical_gradients=True`')
            if not self.analytical_gradients and 'jac' in con:
                del con['jac']
            con['args'] = (parameter_index,)
            cons.append(con)
        return cons

    def fit(self, X, y, sample_weight=None, raise_on_failure=None, **kwargs):
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
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=False)

        if y.ndim != 1:
            raise ValueError('At the moment, only 1-d outcome values are supported '
                             '(this assumption is built into objective function definitions)')
        self.model.validate(X, y)
        X = self.model.prepare_training_data(X, y)

        # Run constrained optimization
        np.random.seed(self.seed)

        self.gradient_err_ = []

        # Initialize function for checking analytical gradients vs approximate gradients
        def callback(pv):
            if not self.monitor_gradient:
                return
            err = optimize.check_grad(
                self.model.evaluate_objective_fn,
                self.model.evaluate_jacobian_fn,
                pv, X, y, sample_weight
            )
            self.gradient_err_.append(err)

        fit_kwargs = kwargs if self.fit_kwargs is None else {**self.fit_kwargs, **kwargs}
        self.fit_ = optimize.minimize(
            fun=self.model.evaluate_objective_fn,
            x0=self.model.get_parameter_starts(),
            jac=self.model.evaluate_jacobian_fn if self.analytical_gradients else None,
            args=(X, y, sample_weight),
            bounds=self.model.get_parameter_bounds(),
            method='SLSQP',
            constraints=self._constraints(),
            callback=callback,
            **fit_kwargs
        )

        # Default raise on failure to class variable, but override with kwargs if set
        raise_on_failure = self.raise_on_failure if raise_on_failure is None else raise_on_failure

        # Trigger appropriate action on lack of convergence (or other optimization error)
        if not self.fit_.success:
            msg = get_fit_summary(self.fit_)
            if raise_on_failure:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        # Trigger appropriate action on gradient calculation discrepancy, if necessary
        if self._gradient_error():
            msg = self._gradient_error_message()
            if raise_on_failure:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        return self

    def _gradient_error(self):
        """
        Determine whether or not analytical gradient calculations contain errors

        This is only available when analytical_gradients=True and monitor_gradient=True (gradient monitoring is
        very expensive computationally).
        :return: Flag indicating whether or not there was a big enough discrepancy between gradient calculation methods
        """
        if not self.monitor_gradient or not self.analytical_gradients:
            return False
        err = np.array(self.gradient_err_)

        # Assume a gradient calculation error occurred if gradients were all nan/inf or if more than the
        # threshold number of iterations had an MSE greater than 1
        if np.all(~np.isfinite(err)) or np.sum(err[np.isfinite(err)] > 1.) > self.gradient_diff_thresh:
            return True
        return False

    def _gradient_error_message(self):
        """
        Generate informative message with context around gradient monitoring failures
        """
        err = np.array(self.gradient_err_)
        err_n = None if not np.any(np.isfinite(err)) else np.sum(err[np.isfinite(err)] > 1)
        err_max = None if not np.any(np.isfinite(err)) else err[np.isfinite(err)].max()
        return 'Gradient monitoring operation determined that more than {} iteration(s) had an analytical gradient ' \
               'that differed significantly in MSE (> 1) between gradient components ' \
               'found via finite differencing (or that all differences were non-finite/nan).  Gradient MSE history ' \
               'across all iterations:\n{}\nMax MSE = {}\nN over 1 = {}' \
            .format(self.gradient_diff_thresh, err, err_max, err_n)

    def get_gradient_error_history(self):
        """
        Fetch history of MSE discrepancies between analytical gradients and those approximated via finite differences

        This is only available after fitting and when both analytical_gradients=True and monitor_gradient=True
        :return: Array of MSE between gradients for each iteration
        """
        check_is_fitted(self, 'fit_')
        return np.array(self.gradient_err_)

    def get_fit_summary(self):
        return get_fit_summary(self.fit_)

    def inference(self):
        return self.model.inference(self.fit_)

    def predict(self, X, key='values'):
        """

        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        :param key: string or None
            Name/type of prediction to return result for; common choices are:
                - None - Returns all available prediction results as a dictionary
                - values - Typically these are scalar values equal to numeric outcome or categorical class
                - probabilities - Typically a 2-D n_samples x n_classes probability matrix
            Other values are possible but depend on the underlying implementation
        :return: Predicted result (structure varies)
        """
        check_is_fitted(self, 'fit_')
        X = check_array(X)
        pred = self.model.predict(self.fit_, X)
        if key is None:
            return pred
        else:
            return pred[key]




