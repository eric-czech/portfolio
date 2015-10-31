__author__ = 'eczech'

import pandas as pd
import numpy as np

from sklearn.grid_search import Parallel, delayed, BaseSearchCV
from sklearn.metrics import roc_curve
from sklearn import base
import functools
import collections
from .log import log, DEFAULT_LOG_FORMAT, DEFAULT_LOG_FILE, set_logger
from .common import *
from .importances import get_classifier_fi, get_regressor_fi

_HR = '-' * 80


def get_feature_importance(clf, columns, mode):
    if mode == MODE_CLASSIFIER:
        return get_classifier_fi(clf, columns)
    if mode == MODE_REGRESSOR:
        return get_regressor_fi(clf, columns)
    return None


def _get_clf_name(clf):
    if isinstance(clf, BaseSearchCV):
        return clf.estimator.__class__.__name__
    return clf.__class__.__name__


def run_fold(args):
    i, train, test, clfs, mode, X, y, kwargs = args
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    res = []

    for clf in clfs:
        model = base.clone(clf[CLF_IMPL])
        log('Running model {} ({}) on fold {}'.format(clf[CLF_NAME], _get_clf_name(model), i+1))
        model.fit(X_train, y_train)

        if kwargs.get('show_best_params') and isinstance(model, BaseSearchCV):
            log('\tBest grid parameters (model = {}, fold = {}): {}'
                .format(_get_clf_name(model), i+1, model.best_params_))

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


def run_regressors(X, y, clfs, cv, **kwargs):
    return run_models(X, y, clfs, cv, MODE_REGRESSOR, **kwargs)


def run_models(X, y, clfs, cv, mode, **kwargs):
    """ Fit multiple classification or regression model concurrently

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
            show_best_params: Prints BaseSearchCV "best_params_" in cross validation loops
    :return: A list of lists where the outer list contains as many entries as there are folds and the inner lists
        contain per-fold results for each estimator given
    """
    validate_mode(mode)

    clfs = [{CLF_NAME: k, CLF_IMPL: v} for k, v in clfs.items()]

    par_kwargs, kwargs = parse_kwargs(kwargs, 'par_')
    log_kwargs, kwargs = parse_kwargs(kwargs, 'log_')

    args_list = [(i, train, test, clfs, mode, X, y, kwargs) for i, (train, test) in enumerate(cv)]

    log_file = log_kwargs.get('file', DEFAULT_LOG_FILE)
    set_logger(log_file, log_kwargs.get('format', DEFAULT_LOG_FORMAT))

    print('Beginning cross validation (see {} for progress updates)'.format(log_file))
    res = Parallel(**par_kwargs)(delayed(run_fold)(args) for args in args_list)
    log('CV Complete ({} model(s) run)'.format(len(clfs)))
    log(_HR)
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
            if clf_res['feat_imp'] is None:
                continue
            feat_imp = pd.DataFrame(clf_res['feat_imp']).T
            feat_imp['model_name'] = clf_res['model'][CLF_NAME]
            feat_imp['fold_id'] = clf_res['fold']
            imp_res.append(feat_imp)
    if len(imp_res) == 0:
        return None
    res = functools.reduce(pd.DataFrame.append, imp_res)
    res.index.name = 'feature'
    return res


def summarize_grid_parameters(res):
    grid_res = collections.defaultdict(list)
    for fold_res in res:
        for clf_res in fold_res:
            model = clf_res['model']['value']

            # Attempt to find base search CV in pipeline, if model
            # was in fact a pipeline
            pipe_model = resolve_estimator_from_pipeline(model)
            if pipe_model is not None:
                model = pipe_model

            # Add to results if model is a parameter search wrapper
            if isinstance(model, BaseSearchCV):
                if hasattr(model, 'param_grid'):
                    keys = model.param_grid.keys()
                elif hasattr(model, 'param_distributions'):
                    keys = model.param_distributions.keys()
                params = model.best_estimator_.get_params()
                params = {k: v for k, v in params.items() if k in keys}
                params['fold_id'] = clf_res['fold']
                params['model_name'] = clf_res['model']['name']
                grid_res[clf_res['model']['name']].append(params)
    grid_res = [pd.melt(pd.DataFrame(grid_res[k]), id_vars=['fold_id', 'model_name']) for k in grid_res]
    if not grid_res:
        return None
    return functools.reduce(pd.DataFrame.append, grid_res)

