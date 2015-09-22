__author__ = 'eczech'

from .common import *
import logging
import multiprocessing

#LOGGER = multiprocessing.log_to_stderr()
#LOGGER.setLevel(logging.IN)
from sklearn.grid_search import BaseSearchCV


def get_classifier_fi(clf, columns):
    res = None

    # If model is grid search, fetch underlying, best estimator
    if isinstance(clf, BaseSearchCV):
        clf = clf.best_estimator_

    # SVC - univariate importance exists only with linear kernel
    # and when number of classes == 2
    if isinstance(clf, SVC) \
        and clf.get_params()['kernel'] == 'linear' \
            and clf.coef_.shape[0] == 1:
        res = clf.coef_[0]

    # Logreg - univariate importance exists only with 2 classes
    if is_instance_of(clf, LOGREG_CLASSIFIERS) and clf.coef_.shape[0] == 1:
        res = clf.coef_[0]

    # Tree classifiers - univariate importance always exists
    if is_instance_of(clf, TREE_CLASSIFIERS):
        res = clf.feature_importances_

    return pd.Series(res, index=columns) if res is not None else res


def get_regressor_fi(clf, columns):
    res = None

    # If model is grid search, fetch underlying, best estimator
    if isinstance(clf, BaseSearchCV):
        clf = clf.best_estimator_

    # SVR - univariate importance is only present with linear kernel
    if isinstance(clf, SVR) \
            and clf.get_params()['kernel'] == 'linear':
        res = clf.coef_[0]

    # Linear regressors - coefficients are always one-dimensional
    if is_instance_of(clf, LINEAR_REGRESSORS):
        res = clf.coef_

    # Tree classifiers - univariate importance always exists
    if is_instance_of(clf, TREE_REGRESSORS):
        res = clf.feature_importances_

    return pd.Series(res, index=columns) if res is not None else res