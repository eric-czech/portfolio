__author__ = 'eczech'

import numpy as np
import pandas as pd


# Classifier Imports
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

# Regressor Imports
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, BayesianRidge, Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV

import logging
import multiprocessing

LOGGER = multiprocessing.log_to_stderr(level=logging.WARN)

TREE_CLASSIFIERS = [GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier]
LOGREG_CLASSIFIERS = [LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression]
TREE_REGRESSORS = [GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor]
LINEAR_REGRESSORS = [ElasticNet, ElasticNetCV, LinearRegression, BayesianRidge, Ridge, RidgeCV, Lasso, LassoCV]

FOLD_COL_PREFIX = 'fold:'
CLF_IMPL = 'value'
CLF_NAME = 'name'

MODE_CLASSIFIER = 'classifier'
MODE_REGRESSOR = 'regressor'


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
