
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin


class TransformEstimator(BaseEstimator, SelectorMixin):
    """
    Wrapper for BaseEstimator instances that will apply an arbitrary transformation function to training data

    This is often useful for estimators that might do things like ignore records where a response is null.

    Note that this does NOT make sense to use on anything that requires features rather than responses -- instead
    this is only useful in a an actual estimation step within a pipeline (often as feature selection or model fitting)
    """

    def __init__(self, estimator=None, transformer=None):
        """
        Initialize transformation wrapper

        :param estimator: BaseEstimator instance to wrap
        :param transformer: Transformation function to apply in `fit`; should have signature (X, y) and return a tuple
        containing the transformed (X', y')
        """
        self.estimator = estimator
        self.transformer = transformer

    def fit(self, X, y):
        assert self.transformer is not None, 'Transformation function cannot be null'
        X, y = self.transformer(X, y)
        return self.estimator.fit(X, y)

    def predict(self, X, **kwargs):
        return self.estimator.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.estimator.predict_proba(X, **kwargs)

    def _get_support_mask(self):
        if not isinstance(self.estimator, SelectorMixin):
            raise ValueError(
                "Estimator in transformation wrapper must be of type 'SelectorMixin' in "
                "order for wrapper to be used as a selector (got '{}')".format(type(self.estimator))
            )
        return self.estimator._get_support_mask()


