

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, has_fit_parameter, check_array
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import parallel_helper
from sklearn.multioutput import _fit_estimator
from abc import ABCMeta
from sklearn.externals import six
import numpy as np


class MultiRegressor(six.with_metaclass(ABCMeta, BaseEstimator)):

    def __init__(self, estimator, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """ Fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
            Returns self.
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        # Ignore this because it chokes on nas
        # X, y = check_X_y(X, y, multi_output=True, accept_sparse=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi target regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying regressor does not support"
                             " sample weights.")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_estimator)(
            self.estimator, X, y[:, i], sample_weight) for i in range(y.shape[1]))
        return self

    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement a predict method")

        X = check_array(X, accept_sparse=True)

        y = Parallel(n_jobs=self.n_jobs)(delayed(parallel_helper)(e, 'predict', X)
                                         for e in self.estimators_)

        return np.asarray(y).T