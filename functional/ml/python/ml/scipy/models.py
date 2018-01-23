
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

    def evaluate_objective_fn(self, pv, X, y, w):
        raise NotImplementedError('Method not yet implemented')

    def evaluate_jacobian_fn(self, pv, X, y, w):
        raise NotImplementedError('Method not yet implemented')

    def get_classes(self, y):
        """ Get all possible class values for classification objectives """
        return None

    def validate(self, X, y):
        pass

    def predict(self, fit, X):
        raise NotImplementedError('Method not yet implemented')

    def inference(self, fit, **kwargs):
        raise NotImplementedError('Method not yet implemented')

    def influence(self, inference, observation, **kwargs):
        raise NotImplementedError('Method not yet implemented')

    def _extract_params(self, parameter_values, parameter_set, transform=None):
        """
        Extract ordered dict of parameter values keyed by name

        :param parameter_values: ndarray containing all parameter values
            (usually much larger than size of parameter set)
        :param parameter_set: Subset of parameters to extract
        :param transform: Function used to transform parameter values;
            signature: fn(parameter_array) -> parameter_array
        :return: Dictionary containing parameter values
        """
        if parameter_set.empty():
            return OrderedDict()
        params = parameter_set.get_parameter_names()
        parameter_index = self.get_parameter_index()
        idx = [parameter_index[p] for p in params]
        vals = parameter_values[idx]
        if transform is not None:
            vals = transform(vals)
        return OrderedDict(zip(params, vals))


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

    def prepare_training_data(self, X, y):
        if self.intercept_params.empty():
            return X, y
        bias = np.ones((len(X), 1))
        return np.hstack((bias, X)), y

    def evaluate_objective_fn(self, pv, X, y, w):
        i = [self.parameter_index[p] for p in self.coef_params.names]
        return self.objective.evaluate_fn(pv[i], X, y, w)

    def evaluate_jacobian_fn(self, pv, X, y, w):
        i = [self.parameter_index[p] for p in self.coef_params.names]
        return self.objective.jacobian_fn(pv[i], X, y, w)

    def inference(self, fit, **kwargs):
        return ScipyModelInference({
            'intercept': self._extract_params(fit.x, self.intercept_params),
            'linear': self._extract_params(fit.x, self.linear_params),
            'fit': self._extract_params(fit.x, self.fit_params)
        })

    def influence(self, inference, observation, **kwargs):
        import pandas as pd
        if not isinstance(observation, pd.Series):
            raise ValueError(
                'Observation must be a Series to obtain influence (type given = {})'
                .format(type(observation))
            )
        d_inf = inference['linear']
        if d_inf.isnull().any():
            raise ValueError('Linear inference contains null values')
        if observation.isnull().any():
            raise ValueError('Observation contains null values')
        if len(d_inf) != len(observation):
            raise ValueError(
                'Linear inference shape ({}) does not match observation shape ({})'
                .format(d_inf.shape, observation.shape)
            )
        if not d_inf.index.sort_values().equals(observation.index.sort_values()):
            raise ValueError(
                'Linear inference index does not equal observation index\n'
                'Observation Index = {}\nInference Index = {}'
                .format(observation.index.sort_values(), d_inf.index.sort_values())
            )
        # Return element-wise product of inference (ie parameters) and observation
        return d_inf * observation


class ScipyLinearRegressionModel(ScipyLinearModel):

    def __init__(self, builder):
        super(ScipyLinearRegressionModel, self).__init__(builder)
        self.y_scaler = builder.y_scaler

    def prepare_training_data(self, X, y):
        if not self.intercept_params.empty():
            bias = np.ones((len(X), 1))
            X = np.hstack((bias, X))
        if y is not None and self.y_scaler is not None:
            self.y_scaler = self.y_scaler.fit(y)
            y = self.y_scaler.transform(y)
        return X, y

    def _invert(self, v):
        if self.y_scaler is None:
            return v
        return self.y_scaler.inverse_transform(v)

    def inference(self, fit, invert=True, **kwargs):
        transform = self._invert if invert else None
        return ScipyModelInference({
            'intercept': self._extract_params(fit.x, self.intercept_params, transform=transform),
            'linear': self._extract_params(fit.x, self.linear_params, transform=transform),
            'fit': self._extract_params(fit.x, self.fit_params)
        })

    def predict(self, fit, X):
        X, _ = self.prepare_training_data(X, None)
        y = np.dot(X, fit.x)
        return {PRED_VALUES: self._invert(y)}


class ScipyLogisticRegressionModel(ScipyLinearModel):

    def __init__(self, builder):
        super(ScipyLogisticRegressionModel, self).__init__(builder)
        self.is_binomial = builder.is_binomial

    def predict(self, fit, X):
        from py_utils import math
        X, _ = self.prepare_training_data(X, None)
        y_proba = math.sigmoid(np.dot(X, fit.x), clip=True)[:, np.newaxis]
        y_proba = np.hstack((1. - y_proba, y_proba))

        # In "binomial" mode rather than "bernoulli", predicted values should
        # be fractions rather than classes
        y_pred = y_proba[:, 1] if self.is_binomial else np.argmax(y_proba, axis=1)

        return {
            PRED_VALUES: y_pred,
            PRED_PROBAS: y_proba
        }

    def validate(self, X, y):
        # Verify that all outcomes in [0, 1] (including fractions)
        mask = (y >= 0) & (y <= 1)
        if not np.all(mask):
            raise ValueError(
                'All outcome (i.e. y) values must be in [0, 1]; Invalid values found = {}'
                .format(np.unique(y[~mask]))
            )

    def get_classes(self, y):
        if self.is_binomial:
            return None
        return [0, 1]


class ScipyPoissonRegressionModel(ScipyLinearModel):

    def __init__(self, builder):
        super(ScipyPoissonRegressionModel, self).__init__(builder)

    def predict(self, fit, X):
        X, _ = self.prepare_training_data(X, None)
        return {PRED_VALUES: np.exp(np.dot(X, fit.x))}


class ScipyLinearModelBuilder(object):

    def __init__(self):
        self.objective = None
        self.linear_params = None
        self.intercept_params = EMPTY_PARAMS
        self.fit_params = EMPTY_PARAMS
        self.constraints = ScipyConstraints()

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
        self.constraints = self.constraints.merge(constraints)
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
        self.y_scaler = None

    def set_y_scaler(self, scaler):
        self.y_scaler = scaler
        return self

    def _objective(self):
        return OBJECTIVE_MSE

    def _build(self):
        return ScipyLinearRegressionModel(self)


class ScipyLogisticRegressionModelBuilder(ScipyLinearModelBuilder):

    def __init__(self):
        super(ScipyLogisticRegressionModelBuilder, self).__init__()
        self.is_binomial = False

    def enable_binomial_outcome(self):
        self.is_binomial = True
        return self

    def enable_bernoulli_outcome(self):
        self.is_binomial = False
        return self

    def _objective(self):
        return OBJECTIVE_MLL

    def _build(self):
        return ScipyLogisticRegressionModel(self)


class ScipyPoissonRegressionModelBuilder(ScipyLinearModelBuilder):

    def __init__(self):
        super(ScipyPoissonRegressionModelBuilder, self).__init__()

    def _objective(self):
        return OBJECTIVE_PML

    def _build(self):
        return ScipyPoissonRegressionModel(self)


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
            raise ValueError('Outcome (i.e. y) values must be one of {}'.format(y_range))

    def get_classes(self, y):
        return np.arange(1, self.n_classes + 1)

    def prepare_training_data(self, X, y):
        return X, y

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
        X, _ = self.prepare_training_data(X, None)
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

    def evaluate_objective_fn(self, pv, X, y, w):
        if w is not None:
            raise ValueError('Sample weights not supported for this model')
        p_out, p_lin = self._split_params(pv)
        return self.objective.evaluate_fn(self.n_classes, p_out, p_lin, X, y)

    def evaluate_jacobian_fn(self, pv, X, y, w):
        if w is not None:
            raise ValueError('Sample weights not supported for this model')
        p_out, p_lin = self._split_params(pv)
        return self.objective.jacobian_fn(self.n_classes, p_out, p_lin, X, y)

    def inference(self, fit, **kwargs):
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
