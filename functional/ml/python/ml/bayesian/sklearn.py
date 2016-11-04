
from ml.bayesian.models import BayesianModel
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import uuid
import os


class BayesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model, model_config=None,
                 model_dir=None):
        assert isinstance(model, BayesianModel), 'Model must be an instance of "{}"'.format(BayesianModel.__name__)
        self.model = model
        self.model_config = model_config
        self.model_dir = model_dir
        self.model_id = None

    def _init(self, new_model_id=True):
        if new_model_id:
            self.model_id = str(uuid.uuid4()).replace('-', '')

        if self.model_dir is not None:
            model_dir = os.path.join(self.model_dir, self.model_id)
            self.model.set_log_dir(model_dir)

    def fit(self, X, y, **kwargs):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self._init()
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] > .5, 1, 0)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        p = self.model.predict(X)
        return np.stack([1.-p, p], axis=1)

