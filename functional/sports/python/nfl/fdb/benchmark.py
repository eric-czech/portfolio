__author__ = 'eczech'


from sklearn.base import BaseEstimator
from sklearn.base import clone


class BenchmarkRegressor(BaseEstimator):

    def __init__(self, benchmark_col):
        self.benchmark_col = benchmark_col

    def get_params(self, deep=True):
        return {'benchmark_col' : self.benchmark_col}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[self.benchmark_col]


class FeatureSubsetRegressor(BaseEstimator):

    def __init__(self, regressor, features):
        self.features = features
        self.regressor = regressor

    def get_params(self, deep=True):
        return {'features': self.features, 'regressor': self.regressor}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X[self.features], y)
        return self

    def predict(self, X):
        return self.regressor_.predict(X[self.features])

