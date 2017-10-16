
from collections import OrderedDict
from ml.scipy.objectives import *
from ml.scipy.common import *
from ml.scipy.constraints import ScipyConstraints


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

    def size(self):
        return self.p

    def get_parameter_names(self):
        return self.names

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


class ScipyModelInference(object):
    """
    Wrapper for late binding of pandas dependencies in interpretation of model results
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        import pandas as pd
        return pd.Series(self.data[item])

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)


class ScipyModel(object):

    def get_parameter_index(self):
        raise NotImplementedError('Method not yet implemented')

    def prepare_training_data(self, X, y):
        raise NotImplementedError('Method not yet implemented')

    def get_parameter_starts(self):
        raise NotImplementedError('Method not yet implemented')

    def get_parameter_bounds(self):
        raise NotImplementedError('Method not yet implemented')

    def get_parameter_constraints(self):
        raise NotImplementedError('Method not yet implemented')

    def evaluate_objective_fn(self, pv, X, y):
        raise NotImplementedError('Method not yet implemented')

    def evaluate_jacobian_fn(self, pv, X, y):
        raise NotImplementedError('Method not yet implemented')

    def validate(self, X, y):
        pass

    def predict(self, fit, X):
        raise NotImplementedError('Method not yet implemented')

    def inference(self, fit):
        raise NotImplementedError('Method not yet implemented')

    def _extract_params(self, parameter_values, parameter_set):
        """
        Extract ordered dict of parameter values keyed by name

        :param parameter_values: ndarray containing all parameter values
            (usually much larger than size of parameter set)
        :param parameter_set: Subset of parameters to extract
        :return: Dictionary containing parameter values
        """
        if parameter_set.empty():
            return OrderedDict()
        params = parameter_set.get_parameter_names()
        parameter_index = self.get_parameter_index()
        idx = [parameter_index[p] for p in params]
        return OrderedDict(zip(params, parameter_values[idx]))


class ScipyLinearModel(ScipyModel):

    def __init__(self, builder):
        self.objective = builder.objective

        self.linear_params = builder.linear_params
        self.intercept_params = builder.intercept_params
        self.fit_params = builder.fit_params

        self.coef_params = self.intercept_params.merge(self.linear_params)
        self.all_params = self.coef_params.merge(self.fit_params)
        self.parameter_index = self.all_params.get_parameter_index()

        self.constraints = builder.constraints

    def get_parameter_index(self):
        return self.parameter_index

    def get_parameter_starts(self):
        return self.all_params.starts

    def get_parameter_bounds(self):
        return self.all_params.bounds

    def get_parameter_constraints(self):
        return self.constraints

    def prepare_training_data(self, X, _):
        if self.intercept_params.empty():
            return X
        bias = np.ones((len(X), 1))
        return np.hstack((bias, X))

    def evaluate_objective_fn(self, pv, X, y):
        i = [self.parameter_index[p] for p in self.coef_params.names]
        return self.objective.evaluate_fn(pv[i], X, y)

    def evaluate_jacobian_fn(self, pv, X, y):
        i = [self.parameter_index[p] for p in self.coef_params.names]
        return self.objective.jacobian_fn(pv[i], X, y)

    def inference(self, fit):
        return ScipyModelInference({
            'intercept': self._extract_params(fit.x, self.intercept_params),
            'linear': self._extract_params(fit.x, self.linear_params),
            'fit': self._extract_params(fit.x, self.fit_params)
        })


class ScipyLinearRegressionModel(ScipyLinearModel):

    def __init__(self, builder):
        super(ScipyLinearRegressionModel, self).__init__(builder)

    def predict(self, fit, X):
        X = self.prepare_training_data(X, None)
        return {'values': np.dot(X, fit.x)}


class ScipyLogisticRegressionModel(ScipyLinearModel):

    def __init__(self, builder):
        super(ScipyLogisticRegressionModel, self).__init__(builder)

    def predict(self, fit, X):
        from py_utils import math
        X = self.prepare_training_data(X, None)
        y_proba = math.sigmoid(np.dot(X, fit.x), clip=True)
        y_proba = np.hstack((1. - y_proba, y_proba))
        y_pred = np.argmax(y_proba, axis=1)
        return {
            PRED_VALUES: y_pred,
            PRED_PROBAS: y_proba
        }


class ScipyLinearModelBuilder(object):

    def __init__(self):
        self.objective = None
        self.linear_params = None
        self.intercept_params = EMPTY_PARAMS
        self.fit_params = EMPTY_PARAMS
        self.constraints = None

    # Not necessary until something other than MSE is needed for scalar, continuous outcome regression
    # def set_objective(self, objective):
    #     if not isinstance(objective, ScipyObjective):
    #         raise ValueError('Objective must be of type "ScipyObjective" (type given = "{}")'.format(type(objective)))
    #     self.objective = objective
    #     return self

    def add_intercept(self, start=None, bound=None):
        self.intercept_params = ScipyParameters(
            names=['intercept'],
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

    def add_constraints(self, constraints):
        if self.constraints is None:
            self.constraints = []
        self.constraints.extend(constraints)
        return self

    def _validate(self):
        if self.objective is None:
            raise ValueError('Objective for linear model must be set')
        if self.linear_params is None:
            raise ValueError('Linear parameters must be specified before building full parameter set')
        if self.intercept_params is None:
            raise ValueError('Intercept params must be set (to EMPTY_PARAMS if not desired)')
        if self.fit_params is None:
            raise ValueError('Fit params must be set (to EMPTY_PARAMS if not desired)')

    def _build(self):
        raise NotImplementedError('Not implemented')

    def _objective(self):
        raise NotImplementedError('Not implemented')

    def build(self):
        self.objective = self._objective()
        self._validate()
        return self._build()


class ScipyLinearRegressionModelBuilder(ScipyLinearModelBuilder):

    def __init__(self):
        super(ScipyLinearRegressionModelBuilder, self).__init__()

    def _objective(self):
        return OBJECTIVE_MSE

    def _build(self):
        return ScipyLinearRegressionModel(self)


class ScipyLogisticRegressionModelBuilder(ScipyLinearModelBuilder):

    def __init__(self):
        super(ScipyLogisticRegressionModelBuilder, self).__init__()

    def _objective(self):
        return OBJECTIVE_MLL

    def _build(self):
        return ScipyLogisticRegressionModel(self)


class ScipyOrdinalRegressionModel(ScipyModel):

    def __init__(self, builder):
        self.objective = builder.objective

        self.linear_params = builder.linear_params
        self.fit_params = builder.fit_params
        self.outcome_params = builder.outcome_params

        # Number of outcome parameters is equal to number of classes - 1
        self.n_classes = self.outcome_params.size() + 1

        self.all_params = self.linear_params\
            .merge(self.outcome_params)\
            .merge(self.fit_params)
        self.parameter_index = self.all_params.get_parameter_index()

        outcome_constraints = ScipyConstraints()
        for i in range(1, self.outcome_params.p):
            outcome_constraints.add_gte(self.outcome_params.names[i], self.outcome_params.names[i - 1])

        self.constraints = builder.constraints.merge(outcome_constraints)

    def get_parameter_index(self):
        return self.parameter_index

    def get_parameter_starts(self):
        return self.all_params.starts

    def get_parameter_bounds(self):
        return self.all_params.bounds

    def get_parameter_constraints(self):
        return self.constraints

    def validate(self, X, y):
        y_range = np.arange(1, self.n_classes + 1)
        if not np.all(np.in1d(y, y_range)):
            raise ValueError('Outcome values must be one of {}'.format(y_range))

    def prepare_training_data(self, X, _):
        return X

    def _split_params(self, pv):
        """ Split parameters into linear and outcome intercepts
        :param pv: Parameter vector (1-d)
        :return: tuple of (outcome intercept parameters, linear parameters)
        """
        i_out = [self.parameter_index[p] for p in self.outcome_params.names]
        i_lin = [self.parameter_index[p] for p in self.linear_params.names]
        return pv[i_out], pv[i_lin]

    def _predict_proba(self, fit, X):
        from py_utils import math
        X = self.prepare_training_data(X, None)
        n = len(X)
        p_out, p_lin = self._split_params(fit.x)
        yb = np.dot(X, p_lin)[:, np.newaxis]
        ya = np.tile(p_out, (n, 1))
        p = math.sigmoid(ya + yb, clip=True)
        assert p.ndim > 1
        p = np.hstack((np.zeros((n, 1)), p, np.ones((n, 1))))
        return np.diff(p, axis=1)

    def predict(self, fit, X):
        y_proba = self._predict_proba(fit, X)
        y_pred = np.argmax(y_proba, axis=1) + 1
        return {
            PRED_VALUES: y_pred,
            PRED_PROBAS: y_proba
        }

    def evaluate_objective_fn(self, pv, X, y):
        p_out, p_lin = self._split_params(pv)
        return self.objective.evaluate_fn(self.n_classes, p_out, p_lin, X, y)

    def evaluate_jacobian_fn(self, pv, X, y):
        p_out, p_lin = self._split_params(pv)
        return self.objective.jacobian_fn(self.n_classes, p_out, p_lin, X, y)

    def inference(self, fit):
        return ScipyModelInference({
            'intercepts': self._extract_params(fit.x, self.outcome_params),
            'linear': self._extract_params(fit.x, self.linear_params),
            'fit': self._extract_params(fit.x, self.fit_params)
        })


class ScipyOrdinalRegressionModelBuilder(object):

    def __init__(self):
        self.objective = OBJECTIVE_OML
        self.linear_params = None
        self.outcome_params = None
        self.fit_params = EMPTY_PARAMS
        self.constraints = ScipyConstraints()

    def add_linear_params(self, names, starts=None, bounds=None):
        self.linear_params = ScipyParameters(
            names=names,
            starts=starts,
            bounds=bounds
        )
        return self

    def add_outcome_intercepts(self, n_outcome, start_range=.1, names=None):
        if names is not None:
            if n_outcome != len(names):
                raise ValueError('Number of outcomes must equal number of outcome parameter names')
        else:
            names = ['Outcome:{}'.format(i+1) for i in range(n_outcome)]

        if n_outcome <= 2:
            raise ValueError('Number of outcomes must be greater than 2 for ordinal regression models')

        # Start outcome intercepts as small numbers near zero, but in sorted order
        # (since ordinal models will assume that these parameters are sorted)
        starts = list(np.linspace(-start_range, start_range, n_outcome))

        # Ignore parameter for final outcome
        self.outcome_params = ScipyParameters(
            names=names[:-1],
            starts=starts[:-1],
            bounds=None
        )
        return self

    def add_fit_params(self, names, starts=None, bounds=None):
        self.fit_params = ScipyParameters(
            names=names,
            starts=starts,
            bounds=bounds
        )
        return self

    def add_constraints(self, constraints):
        self.constraints = self.constraints.merge(constraints)
        return self

    def build(self):
        if self.objective is None:
            raise ValueError('Objective for ordinal model must be set')
        if self.linear_params is None:
            raise ValueError('Linear parameters must be specified before building full parameter set')
        if self.outcome_params is None:
            raise ValueError('Outcome intercept parameters must be specified before building full parameter set')
        if self.fit_params is None:
            raise ValueError('Fit params must be set (to EMPTY_PARAMS if not desired)')
        return ScipyOrdinalRegressionModel(self)
