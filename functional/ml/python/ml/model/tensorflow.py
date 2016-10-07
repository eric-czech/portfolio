
import numpy as np
import tensorflow.contrib.learn.python.learn as learn
from sklearn.base import BaseEstimator, ClassifierMixin
from ml.model.saveable import SaveableMixin
import uuid
import os


class LearnClassifier(BaseEstimator, ClassifierMixin, SaveableMixin):

    def __init__(self, model_fn, model_params=None, fit_params=None, model_config=None,
                 model_dir=None):
        self.model_fn = model_fn
        self.model_params = model_params
        self.fit_params = fit_params
        self.model_config = model_config
        self.model_dir = model_dir
        self.model_id = None
        self.monitors_ = None
        self.classifier_ = None

    def _init(self, new_model_id=True):
        if new_model_id:
            self.model_id = str(uuid.uuid4())
        model_dir = None
        if self.model_dir is not None:
            model_dir = os.path.join(self.model_dir, self.model_id)

        self.classifier_ = learn.Estimator(
                model_fn=self.model_fn, model_dir=model_dir,
                params=self.model_params, config=self.model_config)

    def _restore(self):
        assert self.model_id is not None, 'Model ID must have been set by ".fit" before calling restore'
        self._init(new_model_id=False)

    def fit(self, X, y, **kwargs):
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
        if 'monitor_fn' in kwargs:
            self.monitors_ = kwargs['monitor_fn'](X, y)
            assert isinstance(self.monitors_, dict), 'Monitor map must be a dictionary'
            del kwargs['monitor_fn']

        self._init()

        monitor_list = [] if self.monitors_ is None else list(self.monitors_.values())
        self.classifier_.fit(X, y, monitors=monitor_list, **kwargs)
        return self

    def predict(self, X):
        return np.argmax(self.classifier_.predict(X), axis=1)

    def predict_proba(self, X):
        return self.classifier_.predict(X)

    def prepare_for_save(self):
        self.classifier_ = None
        self.monitors_ = None

    def restore_from_save(self):
        self._restore()


