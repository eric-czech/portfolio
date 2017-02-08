__author__ = 'eczech'

from .common import *

def get_classifier_fi(clf, columns):
    res = None
    clf = resolve_clf(clf)

    # SVC - univariate importance exists only with linear kernel
    # and when number of classes == 2
    if is_linear_svc_1d(clf):
        res = np.abs(clf.coef_[0])

    # Logreg - univariate importance exists only with 2 classes
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
        clf = resolve_clf(clf)
        imp_vals = clf.booster().get_fscore()
        r = pd.Series({f: float(imp_vals.get(f, 0.)) for f in feature_names})
        return pd.Series(r/r.sum(), index=feature_names)
    return xgb_imp


def get_logistic_feature_importance_calculator(feature_names, class_index):
    def logit_imp(clf):
        clf = resolve_clf(clf)
        imp_vals = clf.coef_[class_index]
        r = pd.Series(imp_vals, index=feature_names).abs()
        return r/r.sum()
    return logit_imp


def get_classifier_feat_imp_calculator(feature_names):
    def get_imp(clf):
        feat_imp = get_classifier_fi(clf, feature_names)
        return feat_imp/feat_imp.sum()
    return get_imp
