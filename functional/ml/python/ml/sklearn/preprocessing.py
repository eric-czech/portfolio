
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import TransformerMixin
import numpy as np


class IdentityScaler(BaseEstimator, TransformerMixin):
    """ NOOP scaling useful for maintaining consistency in pipelines when no scaling is wanted"""

    def __init__(self):
        pass

    def fit(self, v):
        return self

    def transform(self, v):
        return v

    def inverse_transform(self, v):
        return v


class MeanScaler(BaseEstimator, TransformerMixin):
    """ Divide all values by mean

    This is useful for positive, right-skewed data (like costs)
    """

    def __init__(self):
        self.mean_ = None

    def fit(self, v):
        self.mean_ = np.mean(v, axis=0)
        return self

    def transform(self, v):
        if self.mean_ is None:
            raise ValueError('Must call fit prior to transform')
        return v / self.mean_

    def inverse_transform(self, v):
        return v * self.mean_
