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
    if is_linear_svc_1d(clf):
        res = np.abs(clf.coef_[0])

    # Logreg - univariate importance exists only with 2 classes
    print(is_logreg_classifier_1d(clf))
    if is_logreg_classifier_1d(clf):
        res = np.abs(clf.coef_[0])

    # Tree classifiers - univariate importance always exists
    if is_tree_classifier(clf):
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
    if is_linear_svr(clf):
        res = np.abs(clf.coef_[0])

    # Linear regressors - coefficients are always one-dimensional
    if is_linear_regressor(clf):
        res = np.abs(clf.coef_)

    # Tree classifiers - univariate importance always exists
    if is_tree_regressor(clf):
        res = clf.feature_importances_

    # Return nothing if feature importance vector does not
    # match shape of column list (can happen with decomposition in pipeline)
    if res is not None and len(res) != len(columns):
        return None
    return pd.Series(res, index=columns) if res is not None else res


def get_xgb_feature_importance_calculator(feature_names):
    def xgb_imp(clf):
        imp_vals = clf.booster().get_fscore()
        r = pd.Series({f: float(imp_vals.get(f, 0.)) for f in feature_names})
        return pd.Series(r/r.sum(), index=feature_names)
    return xgb_imp
