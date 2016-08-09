__author__ = 'eczech'

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

    if kwargs.get('data_resampler'):
        X_train, X_test, y_train, y_test = kwargs.get('data_resampler')(X_train, X_test, y_train, y_test)

    res = []

    for clf in clfs:
        model = base.clone(clf[CLF_IMPL])
        log('Running model {} ({}) on fold {} ==> dim(train) = {}, dim(test) = {}'
            .format(clf[CLF_NAME], _get_clf_name(model), i+1, X_train.shape, X_test.shape))

        # If explicitly provided, use a custom model fitting function
        if kwargs.get('model_fit_function'):
            kwargs.get('model_fit_function')(clf[CLF_NAME], model, X_train, y_train, X_test, y_test, i == -1)
        # Otherwise, run base estimator fit
        else:
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
        if kwargs.get('keep_training_data', False):
            fold_res['X_test'] = X_test
            fold_res['X_train'] = X_train
            fold_res['y_train'] = y_train
        if mode == MODE_CLASSIFIER and kwargs.get('predict_proba', True):
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
        Logging kwargs:
            log_*: log_file and log_format can be passed to determine logging output location and style
        Joblib kwargs:
            par_*: Arguments, prefixed by 'par_', to be passed to parallel executor (e.g. par_n_jobs, par_backend)
        Other:
            refit: Boolean indicating whether or not the given models should also be trained on the full dataset;
                defaults to false
            keep_training_data: Boolean indicating whether or not training/test datasets should be preserved in results
            predict_proba: Boolean indicating whether or not class probability predictions should be made
                for classifiers; defaults to false
            show_best_params: Prints BaseSearchCV "best_params_" in cross validation loops
            data_resampler: Optional function to be passed X_train, X_test, y_train and y_test for each fold.  This
                function is expected to return a new value for each of those given as a four-value tuple.  This is
                commonly useful for implementing downsampling/upsampling prior to training, or any other more
                complicated sort of cross-validation.

                Example: lambda X_train, X_test, y_train, y_test: \
                    X_train.head(10), X_test.head(10), y_train.head(10), y_test.head(10)
            model_fit_function: Optional custom fitting function taking a model name, model object,
                training/testing sets, and a flag indicating whether or not this is a "refit" on whole training set.

                Example: lambda clf_name, clf, X_train, y_train, X_test, y_test, is_refit: clf.fit(X_train, y_train)
    :return: A list of lists where the outer list contains as many entries as there are folds and the inner lists
        contain per-fold results for each estimator given.  If "refit=True", then two such lists will be returned as
        a two value tuple (the first will be the same result as if refit was false, and the second will be results with
        with nearly the same structure, minus the outer list for each fold)
    """
    validate_mode(mode)

    clfs = [{CLF_NAME: k, CLF_IMPL: v} for k, v in clfs.items()]

    par_kwargs, kwargs = parse_kwargs(kwargs, 'par_')
    log_kwargs, kwargs = parse_kwargs(kwargs, 'log_')

    args_list = [(i, train, test, clfs, mode, X, y, kwargs) for i, (train, test) in enumerate(cv)]

    log_file = log_kwargs.get('file', DEFAULT_LOG_FILE)
    set_logger(log_file, log_kwargs.get('format', DEFAULT_LOG_FORMAT))

    print('Beginning cross validation (see {} for progress updates)'.format(log_file))
    cv_res = Parallel(**par_kwargs)(delayed(run_fold)(args) for args in args_list)
    log('CV Complete ({} model(s) run)'.format(len(clfs)))

    # If refitting was not requested, return the cross validation results
    if not kwargs.get('refit'):
        log(_HR)
        return cv_res

    # Otherwise, train models on full dataset and then return both results
    # separately (for cross validation and full dataset training)
    idx = np.arange(0, len(y))
    args = (-1, idx, idx, clfs, mode, X, y, kwargs)

    log('Beginning model refitting')
    refit_res = run_fold(args)
    log('Model refitting complete ({} model(s) run)'.format(len(clfs)))
    log(_HR)
    return cv_res, refit_res


def extract_model_results(clf_name, cv_res):
    clf_res = []
    for fold_res in cv_res:
        for model_res in fold_res:
            if model_res['model']['name'] == clf_name:
                clf_res.append(model_res)
    return clf_res


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

            # Compute the score for this fold which may be returned either as a scalar value,
            # or a dictionary containing several name / scalar value pairs
            score = score_func(clf, y_test, y_pred)

            # If a scalar was returned, wrap it in a dictionary
            if not isinstance(score, dict):
                score = {'score': score}

            # Add the row to the running result
            row = {'fold_id': fold_id, 'model_name': clf_name}
            row.update(score)

            score_res.append(row)
    return pd.DataFrame(score_res)


def summarize_curve(res, curve_func=roc_curve):
    preds = summarize_predictions(res)

    def get_curve(x):
        model = x['model_name'].iloc[0]
        if 'y_proba_1' not in x.columns:
            print('Warning: No probability predictions found for model "{}"'.format(model))
            return None
        if 'y_proba_2' in x.columns:
            raise ValueError('Curve summarizations are not available for multinomial classifiers')
        x, y, thresh = curve_func(x['y_true'], x['y_proba_1'])
        return pd.DataFrame({'x': x, 'y': y, 'thresh': thresh})
    return preds.groupby(['model_name', 'fold_id']).apply(get_curve)\
        .reset_index()[['model_name', 'fold_id', 'x', 'y', 'thresh']]


def summarize_predictions(res, y_names=None):
    """
    Returns per-fold predictions from a cross-validation result
    :param res: Cross validation result
    :param y_names: Optional list of names to assign to multivariate or univariate outcomes; note
        that these must be given in the same order as the response fields used in training
    :return: Data frame containing predictions
    """
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
            y_pred = clf_res['y_pred']
            if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                pred[pref + 'y_pred'] = clf_res['y_pred']
                pred[pref + 'y_true'] = clf_res['y_test']
            else:
                for i in range(y_pred.shape[1]):
                    outcome = str(i) if y_names is None else y_names[i]
                    pred[pref + '{}y_pred_{}'.format(pref, outcome)] = clf_res['y_pred'][:, i]
                    pred[pref + '{}y_true_{}'.format(pref, outcome)] = clf_res['y_test'][:, i]

            if 'y_proba' in clf_res:
                y_proba = clf_res['y_proba']
                for i in range(y_proba.shape[1]):
                    outcome = str(i) if y_names is None else y_names[i]
                    pred['{}y_proba_{}'.format(pref, outcome)] = y_proba[:, i]

            # Add fold meta data to frame
            pred[pref + 'model_name'] = clf_res['model'][CLF_NAME]
            pred[pref + 'fold_id'] = clf_res['fold']

            pred_res.append(pred)
    return functools.reduce(pd.DataFrame.append, pred_res)


def summarize_importances(res, feat_imp_calc=None):
    imp_res = []
    for fold_res in res:
        for clf_res in fold_res:
            model = clf_res['model'][CLF_NAME]
            feat_imp = None

            # If a feature importance calculation function has been specified
            # for this model, use it instead of any pre-computed importances
            if feat_imp_calc is not None and model in feat_imp_calc:
                feat_imp = feat_imp_calc[model](clf_res['model'][CLF_IMPL])

            # Fallback on pre-computed importance if present
            if feat_imp is None and clf_res['feat_imp'] is not None:
                feat_imp = clf_res['feat_imp']

            # If no feature importance could be found/created, skip this model
            if feat_imp is None:
                continue

            feat_imp = pd.DataFrame(feat_imp).T
            feat_imp['model_name'] = clf_res['model'][CLF_NAME]
            feat_imp['fold_id'] = clf_res['fold']
            imp_res.append(feat_imp)
    if len(imp_res) == 0:
        return None
    res = functools.reduce(pd.DataFrame.append, imp_res)
    return res.reset_index(drop=True)


def summarize_weighted_importances(res, score_func, feat_agg=np.median, score_agg=np.median):
    """
    Computes model-accuracy-weighted feature scores across folds and model types

    :param res: Result from model training
    :param score_func: Scoring function used to compute model scores with
    :param feat_agg: Feature importance aggregation function for a specific model type
    :param score_agg: Model accuracy aggregation function across folds
    :return: Series containing model-accuracy-weighted feature importances
    """
    feat_imp = summarize_importances(res)

    # Compute feature scores across folds for each model
    feat_imp = feat_imp.groupby('model_name')\
        .apply(lambda x: x.drop(['fold_id', 'model_name'], axis=1).apply(feat_agg))

    def scale_to_unit(x):
        return (x - x.min())/(x.max() - x.min())

    # Normalize each feature score per model to [0, 1]
    feat_imp = feat_imp.apply(scale_to_unit, axis=1)

    # Compute per-model scores and aggregate across folds to give a single
    # value per model, and then use that value to compute per-model weight
    scores = summarize_scores(res, score_func, use_proba=False)
    scores = scores.groupby('model_name')['score'].apply(score_agg)
    scores = scores / scores.sum()
    #print(scores)

    # Multiply each feature + model score times model weight and sum across
    # models to give overall feature score
    scores = feat_imp.apply(lambda x: x * scores.loc[x.name], axis=1).sum().order()
    scores.index.name = 'feature'
    scores.name = 'score'

    # Return series with feature names in index and model-accuracy-weighted scores in value
    return scores


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

