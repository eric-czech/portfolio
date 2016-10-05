
import numpy as np
import pandas as pd
import tensorflow.contrib.learn.python.learn as learn
from sklearn.base import BaseEstimator, ClassifierMixin
import uuid
import os


class LearnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_fn, model_params=None, fit_params=None, model_config=None, model_dir=None):
        self.model_fn = model_fn
        self.model_params = model_params
        self.fit_params = fit_params
        self.model_config = model_config
        self.model_dir = model_dir

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

        model_dir = None
        if self.model_dir is not None:
            model_dir = os.path.join(self.model_dir, str(uuid.uuid4()))

        self.classifier_ = learn.Estimator(
                model_fn=self.model_fn, model_dir=model_dir,
                params=self.model_params, config=self.model_config)

        monitor_list = [] if self.monitors_ is None else list(self.monitors_.values())
        self.classifier_.fit(X, y, monitors=monitor_list, **kwargs)
        return self

    def predict(self, X):
        return np.argmax(self.classifier_.predict(X), axis=1)

    def predict_proba(self, X):
        return self.classifier_.predict(X)
