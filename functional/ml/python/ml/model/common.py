__author__ = 'eczech'

import numpy as np
import pandas as pd


# Classifier Imports
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# Regressor Imports
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, BayesianRidge, Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.pipeline import Pipeline
from sklearn.grid_search import BaseSearchCV

TREE_CLASSIFIERS = [GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier]
# TREE_CLASSIFIER_EXT = ['XGBClassifier']
TREE_CLASSIFIER_EXT = []  # XGBClassifier has no feature_importances_

LOGREG_CLASSIFIERS = [LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression]
TREE_REGRESSORS = [GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor]
# TREE_REGRESSORS_EXT = ['XGBRegressor']
TREE_REGRESSORS_EXT = []  # XGBRegressor has no feature_importances_
LINEAR_REGRESSORS = [
    ElasticNet, ElasticNetCV, LinearRegression, BayesianRidge, Ridge, RidgeCV, Lasso, LassoCV
]
LINEAR_REGRESSORS_MV = [
    MultiTaskLassoCV, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso
]

FOLD_COL_PREFIX = 'fold:'
CLF_IMPL = 'value'
CLF_NAME = 'name'

MODE_CLASSIFIER = 'classifier'
MODE_REGRESSOR = 'regressor'


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

def is_linear_regressor(clf):
    return is_instance_of(clf, LINEAR_REGRESSORS)


def is_linear_svr(clf):
    return isinstance(clf, SVR) and clf.get_params()['kernel'] == 'linear'


def is_linear_svc_1d(clf):
    return isinstance(clf, SVC) \
        and clf.get_params()['kernel'] == 'linear' \
            and clf.coef_.shape[0] == 1


def is_tree_regressor(clf):
    return is_instance_of(clf, TREE_REGRESSORS) or clf.__class__.__name__ in TREE_REGRESSORS_EXT


def is_tree_classifier(clf):
    return is_instance_of(clf, TREE_CLASSIFIERS) or clf.__class__.__name__ in TREE_CLASSIFIER_EXT


def is_logreg_classifier_1d(clf):
    return is_instance_of(clf, LOGREG_CLASSIFIERS) and clf.coef_.shape[0] == 1


def validate_mode(mode):
    modes = [MODE_CLASSIFIER, MODE_REGRESSOR]
    if mode not in modes:
        raise ValueError('Given mode "{}" is not valid.  Must be one of: {}'.format(mode, modes))


def is_instance_of(obj, classes):
    return np.any([isinstance(obj, c) for c in classes])


def parse_kwargs(kwargs, prefix):
    keys = [k for k in kwargs if k.startswith(prefix)]
    new_kwargs = {k.replace(prefix, '', 1): kwargs[k] for k in keys}
    old_kwargs = {k: kwargs[k] for k in kwargs if k not in keys}
    return new_kwargs, old_kwargs


def resolve_estimator_from_pipeline(clf):
    """
    Returns the estimator within a pipeline named as 'est' or 'clf'
    :param clf: The estimator instance (possibly a pipeline) to resolve
    :return: The estimator found or None if the given estimator was not a pipeline
        or a step named 'est' or 'clf' as not found.
    """
    if isinstance(clf, Pipeline):
        steps = clf.named_steps
        for name in ['clf', 'est']:
            if name in steps:
                return clf.named_steps[name]
    return None
