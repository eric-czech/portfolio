
import tensorflow.contrib.learn.python.learn as learn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from py_utils import arg_utils
import uuid
import os
import tempfile


class LearnEstimator(BaseEstimator):

    def __init__(self, model_fn, model_dir=tempfile.mkdtemp(), model_params=None, fit_params=None,
                 model_config=None):
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.model_params = model_params
        self.fit_params = fit_params
        self.model_config = model_config
        self.model_id = None
        self.monitors_ = None

        if not self.model_dir:
            raise ValueError('Model directory must be specified')
        if not os.path.exists(self.model_dir):
            raise ValueError('Directory specified "{}" does not exist.'.format(self.model_dir))

    def __getstate__(self):
        # Return all variables and settings except for any ending with "_" (like 'monitors_')
        state = {k: v for k, v in self.__dict__.items() if not k.endswith('_')}
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def get_estimator(self):
        return self._estimator()

    def _estimator(self, new_model_id=False):
        if new_model_id:
            self.model_id = str(uuid.uuid4()).replace('-', '')

        if self.model_id is None:
            raise ValueError('Retrieving estimator is only possible after calling .fit first')

        model_dir = os.path.join(self.model_dir, self.model_id)

        # If model function takes a single "model_dir" argument, assume it
        # is a factory method for tflearn models taking nothing other than a model directory
        fn_args = self.model_fn.__code__.co_varnames
        if len(fn_args) >= 1 and fn_args[0] == 'model_dir':
            return learn.SKCompat(self.model_fn(model_dir))
        # Otherwise, assume the function is a custom model graph to be wrapped in a new
        # estimator
        else:
            return learn.SKCompat(learn.Estimator(
                    model_fn=self.model_fn, model_dir=model_dir,
                    params=self.model_params, config=self.model_config))

    def fit(self, X, Y, **kwargs):
        # Merge pre-defined fit params into actual fit params
        if self.fit_params is not None:
            for k in self.fit_params:
                assert k not in kwargs, \
                    'Fit parameter given that was already defined on construction: {}'.format(k)
            kwargs.update(self.fit_params)

        # Validate monitors argument
        assert 'monitors' not in kwargs, \
            'Monitors should not be passed directly to the fit function, '\
            'instead a monitor_fn argument should be supplied that takes '\
            'as input the X and y values to fit on and returns a map of '\
            'monitors keyed by name for later reference'
        self.monitors_ = None
        monitor_kwargs, kwargs = arg_utils.parse_kwargs(kwargs, 'monitor_')
        if monitor_kwargs:
            assert 'fn' in monitor_kwargs, \
                'Monitor arguments must contain "fn" argument.  Arguments given = {}'.format(monitor_kwargs)
            monitor_fn = monitor_kwargs['fn']
            del monitor_kwargs['fn']
            self.monitors_ = monitor_fn(X, Y, **monitor_kwargs)
            assert isinstance(self.monitors_, dict), 'Monitor map must be a dictionary'

        monitor_list = None if self.monitors_ is None else list(self.monitors_.values())
        self._estimator(new_model_id=True).fit(X, Y, monitors=monitor_list, **kwargs)

        return self


class LearnClassifier(LearnEstimator, ClassifierMixin):

    def __init__(self, model_fn, model_params=None, fit_params=None, model_config=None,
                 model_dir=None, class_output='classes', proba_output='probabilities'):
        super(LearnClassifier, self).__init__(
            model_fn,
            model_params=model_params,
            fit_params=fit_params,
            model_config=model_config,
            model_dir=model_dir
        )
        self.class_output = class_output
        self.proba_output = proba_output

    def predict(self, X, output=None):
        o = output or self.class_output
        return self._estimator().predict(X, outputs=[o])[o]

    def predict_proba(self, X, output=None):
        o = output or self.proba_output
        return self._estimator().predict(X, outputs=[o])[o]


class LearnRegressor(LearnEstimator, RegressorMixin):

    def __init__(self, model_fn, model_params=None, fit_params=None, model_config=None,
                 model_dir=None, prediction_output='scores'):
        super(LearnRegressor, self).__init__(
            model_fn,
            model_params=model_params,
            fit_params=fit_params,
            model_config=model_config,
            model_dir=model_dir
        )
        self.prediction_output = prediction_output

    def predict(self, X, output=None):
        o = output or self.prediction_output
        return self._estimator().predict(X, outputs=[o])[o]
