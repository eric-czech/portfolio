
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


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


class AverageRegressor(BaseEstimator, RegressorMixin):
    """
    Makes predictions using the average value of all given features

    Parameters
    ----------
    weights : {array-like}, [n_features]
        Weights to multiply features by before averaging
    """

    def __init__(self, weights=None):
        if weights:
            assert np.isclose(np.sum(weights), 1), \
                'Sum of weights must be 1 (sum given is {})'.format(np.sum(weights))
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        if self.weights is not None:
            assert len(self.weights) == X.shape[1], \
                'Weight vector size invalid (Expecting {}, Got {})'.format(X.shape[1], len(self.weights))
        return self

    def predict(self, X):
        w = np.ones(X.shape[1])/X.shape[1] if self.weights is None else self.weights
        return np.dot(X, w)

