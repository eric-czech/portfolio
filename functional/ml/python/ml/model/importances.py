__author__ = 'eczech'

from .common import *
from sklearn.grid_search import BaseSearchCV


def resolve_clf(clf):
    # If model is grid search, fetch underlying, best estimator
    if isinstance(clf, BaseSearchCV):
        return clf.best_estimator_

    # If model is pipeline, resolved contained estimator
    pipe_clf = resolve_estimator_from_pipeline(clf)
    if pipe_clf is not None:
        return resolve_clf(pipe_clf)

    # Otherwise, return as is
    return clf


def get_classifier_fi(clf, columns):
    res = None
    clf = resolve_clf(clf)

    # SVC - univariate importance exists only with linear kernel
    # and when number of classes == 2
    if isinstance(clf, SVC) \
        and clf.get_params()['kernel'] == 'linear' \
            and clf.coef_.shape[0] == 1:
        res = np.abs(clf.coef_[0])

    # Logreg - univariate importance exists only with 2 classes
    if is_instance_of(clf, LOGREG_CLASSIFIERS) and clf.coef_.shape[0] == 1:
        res = np.abs(clf.coef_[0])

    # Tree classifiers - univariate importance always exists
    if is_instance_of(clf, TREE_CLASSIFIERS):
        res = clf.feature_importances_

    # Return nothing if feature importance vector does not
    # match shape of column list (can happen with decomposition in pipeline)
    if res is not None and len(res) != len(columns):
        return None
    return pd.Series(res, index=columns) if res is not None else res


def get_regressor_fi(clf, columns):
    res = None
    clf = resolve_clf(clf)

    # SVR - univariate importance is only present with linear kernel
    if isinstance(clf, SVR) \
            and clf.get_params()['kernel'] == 'linear':
        res = np.abs(clf.coef_[0])

    # Linear regressors - coefficients are always one-dimensional
    if is_instance_of(clf, LINEAR_REGRESSORS):
        res = np.abs(clf.coef_)

    # Tree classifiers - univariate importance always exists
    if is_instance_of(clf, TREE_REGRESSORS):
        res = clf.feature_importances_

    # Return nothing if feature importance vector does not
    # match shape of column list (can happen with decomposition in pipeline)
    if res is not None and len(res) != len(columns):
        return None
    return pd.Series(res, index=columns) if res is not None else res