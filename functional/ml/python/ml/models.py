__author__ = 'eczech'

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression
from sklearn.grid_search import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve
from sklearn.grid_search import GridSearchCV
from sklearn import base
from sklearn.svm import SVC
import functools
import multiprocessing
import logging

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

TREE_CLASSIFIERS = [GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier]
LOGREG_CLASSIFIERS = [LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression]
MODE_CLASSIFIER = 'classifier'
MODE_REGRESSOR = 'regressor'
FOLD_COL_PREFIX = 'fold:'
CLF_IMPL = 'value'
CLF_NAME = 'name'


def is_instance_of(obj, classes):
    return np.any([isinstance(obj, c) for c in classes])


def _get_classifier_fi(clf, columns):
    res = None

    # If model is grid search, fetch underlying, best estimator
    if isinstance(clf, GridSearchCV):
        logger.info(clf.best_params_)
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


def get_feature_importance(clf, columns, mode):
    if mode == MODE_CLASSIFIER:
        return _get_classifier_fi(clf, columns)
    return None


def run_fold(args):
    i, train, test, clfs, mode, X, y, kwargs = args
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    res = []
    for clf in clfs:
        model = base.clone(clf[CLF_IMPL])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        feat_imp = get_feature_importance(model, X_test.columns.tolist(), mode)

        fold_res = {
            'fold': i + 1,
            'model': {CLF_NAME: clf[CLF_NAME], CLF_IMPL: model},
            'y_pred': y_pred,
            'y_test': y_test.values,
            'feat_imp': feat_imp
        }
        if kwargs.get('keep_X', False):
            fold_res['X_test'] = X_test
        if mode == MODE_CLASSIFIER and kwargs.get('predict_proba', False):
            fold_res['y_proba'] = model.predict_proba(X_test)

        res.append(fold_res)
    return res


def run_classifiers(X, y, clfs, cv, **kwargs):
    return run_models(X, y, clfs, cv, MODE_CLASSIFIER, **kwargs)


def parse_kwargs(kwargs, prefix):
    keys = [k for k in kwargs if k.startswith(prefix)]
    new_kwargs = {k.replace(prefix, '', 1): kwargs[k] for k in keys}
    old_kwargs = {k: kwargs[k] for k in kwargs if k not in keys}
    return new_kwargs, old_kwargs


def run_models(X, y, clfs, cv, mode, **kwargs):
    """ Fit multiple classification or regression models concurrently

    :param X: Feature data set (must be a pandas DataFrame)
    :param y: Response to predict (must be a pandas Series)
    :param clfs: Estimators to fit (must be a dictionary of 'name': 'value' pairs).  For Example:
        {'logreg': LogisticRegression(),
        'gbr': GridSearchCV(GradientBoostingClassifier(), {'n_estimators': [10, 25, 50, 100]})}
    :param cv: Cross validation definition
    :param mode: Either 'classifier' (#MODE_CLASSIFIER) or 'regressor' (#MODE_REGRESSOR)
    :param kwargs:
        Joblib kwargs:
            par_*: Arguments, prefixed by 'par_', to be passed to parallel executor (e.g. par_n_jobs, par_backend)
        Other:
            keep_X: Boolean indicating whether or not test datasets should be preserved in results
            predict_proba: Boolean indicating whether or not class probability predictions should be made for classifiers;
                defaults to false
    :return: A list of lists where the outer list contains as many entries as there are folds and the inner lists
        contain per-fold results for each estimator given
    """
    clfs = [{CLF_NAME: k, CLF_IMPL: v} for k, v in clfs.items()]

    par_kwargs, kwargs = parse_kwargs(kwargs, 'par_')

    args_list = [(i, train, test, clfs, mode, X, y, kwargs) for i, (train, test) in enumerate(cv)]
    res = Parallel(**par_kwargs)(delayed(run_fold)(args) for args in args_list)

    return res


def summarize_scores(res, score_func, use_proba=False):
    score_res = []
    for fold_res in res:
        for clf_res in fold_res:
            clf = clf_res['model'][CLF_IMPL]
            clf_name = clf_res['model'][CLF_NAME]
            fold_id = clf_res['fold']
            y_test = clf_res['y_test']
            if use_proba:
                if 'y_proba' not in clf_res:
                    raise ValueError(
                        'Cannot compute score when flag to use probability predictions for scoring is ' +
                        'true but probability predictions do not exist in model results [model name = "{}"]'
                        .format(clf_name)
                    )
                y_pred = clf_res['y_proba']
            else:
                y_pred = clf_res['y_pred']
            score = score_func(clf, y_test, y_pred)
            score_res.append((fold_id, clf_name, score))
    return pd.DataFrame(score_res, columns=['fold_id', 'model_name', 'score'])


def summarize_curve(res):
    preds = summarize_predictions(res)

    def get_curve(x):
        if 'y_proba_2' in x.columns:
            raise ValueError('Curve summarizations are not available for multinomial classifiers')
        fpr, tpr, _ = roc_curve(x['y_true'], x['y_proba_1'])
        return pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    return preds.groupby(['model_name', 'fold_id']).apply(get_curve)\
        .reset_index()[['model_name', 'fold_id', 'fpr', 'tpr']]


def summarize_predictions(res):
    pred_res = []
    for fold_res in res:
        for clf_res in fold_res:

            # Create resulting fold predictions data frame using the test set
            # features for the fold if present; otherwise create an empty frame
            if 'X_test' in clf_res:
                pred = clf_res['X_test'].copy()
                pref = FOLD_COL_PREFIX
            else:
                pref = ''
                pred = pd.DataFrame()

            # Add fold predictions to frame
            pred[pref + 'y_pred'] = clf_res['y_pred']
            pred[pref + 'y_true'] = clf_res['y_test']
            if 'y_proba' in clf_res:
                y_proba = clf_res['y_proba']
                for i in range(y_proba.shape[1]):
                    pred['{}{}_{}'.format(pref, 'y_proba', i)] = y_proba[:, i]

            # Add fold meta data to frame
            pred[pref + 'model_name'] = clf_res['model'][CLF_NAME]
            pred[pref + 'fold_id'] = clf_res['fold']

            pred_res.append(pred)
    return functools.reduce(pd.DataFrame.append, pred_res)


def summarize_importances(res):
    imp_res = []
    for fold_res in res:
        for clf_res in fold_res:
            feat_imp = pd.DataFrame(clf_res['feat_imp']).T
            feat_imp['model_name'] = clf_res['model'][CLF_NAME]
            feat_imp['fold_id'] = clf_res['fold']
            imp_res.append(feat_imp)
    res = functools.reduce(pd.DataFrame.append, imp_res)
    res.index.name = 'feature'
    return res