
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DummyBinaryFeatureClassifier(BaseEstimator, ClassifierMixin):
    """
    Makes predictions using a feature given in the dataset.

    This is useful when benchmarking against a known predictive quantity
    """

    def __init__(self, feature_index):
        assert isinstance(feature_index, int), 'Feature index must be an integer'
        self.feature_index = feature_index

    def fit(self, X, y, sample_weight=None):
        assert self.feature_index < X.shape[1]
        return self

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

    def predict_proba(self, X):
        X = np.array(X)
        p = X[:, self.feature_index]
        p = np.clip(p, 0., 1.)
        return np.transpose(np.vstack((1-p, p)))
