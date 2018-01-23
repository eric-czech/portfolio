import numpy as np
import pandas as pd
from sklearn import base
from sklearn.externals.joblib import Parallel, delayed

from ml.api.constants import *
from ml.api.utils.array_utils import to_data_frame
from ml.model.log import log, log_hr, DEFAULT_LOG_FORMAT, DEFAULT_LOG_FILE, set_logger
from py_utils import arg_utils, io_utils
from py_utils.assertion_utils import assert_true


def _get_clf_name(clf):
    if hasattr(clf, 'estimator'):
        return clf.estimator.__class__.__name__
    # if hasattr(clf, 'base_estimator'):
        # return clf.base_estimator.__class__.__name__
    return clf.__class__.__name__


class TrainingModelResult(object):

    def __init__(self, fold, clf_name, clf, mode,
                 Y_pred, Y_test, X_names, Y_names,
                 X_train=None, X_test=None, Y_train=None, Y_proba=None):
        self.fold = fold
        self.clf_name = clf_name
        self.clf = clf
        self.mode = mode
        self.Y_pred = Y_pred
        self.Y_test = Y_test
        self.X_names = X_names
        self.Y_names = Y_names
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_proba = Y_proba
        self.validate()

    def validate(self):
        n_test = len(self.Y_pred)
        assert n_test == \
            (n_test if self.Y_test is None else len(self.Y_test)) == \
            (n_test if self.Y_proba is None else len(self.Y_proba))

        if self.X_train is not None and self.Y_train is not None:
            assert len(self.X_train) == len(self.Y_train)

    def has_test_data(self):
        return self.X_test is not None

    def has_training_data(self):
        return self.X_train is not None


class TrainingResampleResult(object):

    def __init__(self, model_results):
        self.model_results = model_results


class TrainingResult(object):

    def __init__(self, resample_results, refit_result, trainer_config, mode):
        self.resample_results = resample_results
        self.refit_result = refit_result
        self.trainer_config = trainer_config
        self.mode = mode

    def save(self, file_path, obj_name="CV Training Result"):
        io_utils.to_pickle(self, file_path, obj_name=obj_name)

    @staticmethod
    def load(file_path, obj_name="CV Training Result"):
        return io_utils.from_pickle(file_path, obj_name=obj_name)


def _convert_to_data_frames(Y_pred, Y_test, Y_train, Y_names=None, Y_pred_index=None):

    # Extract the name(s) associated with response values
    if Y_names is None:
        Y_names = Y_train.columns.tolist() if isinstance(Y_train, pd.DataFrame) else [Y_train.name]

    # Convert predicted values to a data frame (should be from either 1D or 2D numpy array)
    if Y_pred is not None:
        idx = Y_pred_index
        if idx is None:
            if Y_test is None:
                raise ValueError(
                        'Index for prediction frame (ie Y_pred) must be given '
                        'explicitly if Y_test is not provided'
                )
            else:
                idx = Y_test.index
        Y_pred = to_data_frame(Y_pred, index=idx, columns=Y_names)
        assert_true(Y_pred.columns.tolist() == Y_names,
                    lambda: 'Y_pred columns "{}" do not matched expected columns "{}"'
                    .format(Y_pred.columns.tolist(), Y_names))

    # Convert test response values to a data frame (should be from either a Series or DataFrame)
    if Y_test is not None:
        Y_test = to_data_frame(Y_test)
        assert_true(Y_test.columns.tolist() == Y_names,
                    lambda: 'Y_test columns "{}" do not matched expected columns "{}"'
                    .format(Y_test.columns.tolist(), Y_names))

    # Convert training response values to a data frame (should be from either a Series or DataFrame)
    if Y_train is not None:
        Y_train = to_data_frame(Y_train)
        assert_true(Y_train.columns.tolist() == Y_names,
                    lambda: 'Y_train columns "{}" do not matched expected columns "{}"'
                    .format(Y_train.columns.tolist(), Y_names))

    return Y_pred, Y_test, Y_train


def _data_prep(prep_fn, X_train, X_test, Y_train, Y_test, allow_none=False):
    X_train, X_test, Y_train, Y_test = prep_fn(X_train, X_test, Y_train, Y_test)
    for x in [X_train, X_test]:
        if allow_none and x is None:
            continue
        if not isinstance(x, pd.DataFrame):
            raise ValueError('Data prep function must return DataFrame instances for feature data')
    for y in [Y_train, Y_test]:
        if allow_none and y is None:
            continue
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series):
            raise ValueError('Data prep function must return DataFrame or Series instances for response data')
    return X_train, X_test, Y_train, Y_test


def run_predict(train_res, X, Y=None):
    from ml.api.results import properties

    if train_res.refit_result is None:
        raise ValueError('Refit models in training result cannot be missing (try retraining with config.refit=True)')

    X_names = properties.extract_property_by_consensus(train_res, lambda model_result: model_result.X_names, 'X_names')
    Y_names = properties.extract_property_by_consensus(train_res, lambda model_result: model_result.Y_names, 'Y_names')

    assert_true(X_names == X.columns.tolist(),
                lambda: 'Given feature set with columns "{}" does not match expected columns "{}"'
                .format(X.columns.tolist(), X_names))

    # Run the data preparation function if one was configured, and ensure that the results from this
    # function are still DataFrames or Series, not numpy arrays
    if train_res.trainer_config.data_prep_fn is not None:
        X, _, Y, _ = _data_prep(train_res.trainer_config.data_prep_fn, X, None, Y, None, allow_none=True)

    res = []

    for model_res in train_res.refit_result.model_results:

        log(
            'Running predictions for model {} ==> dim(X) = {}'
            .format(model_res.clf_name, X.shape)
        )

        # Make predictions on test data features
        Y_pred = model_res.clf.predict(X)

        # Convert predicted values to a data frame (should be from either 1D or 2D numpy array)
        # Y_pred = to_data_frame(Y_pred, index=X.index, columns=Y_names)
        # assert_true(Y_pred.columns.tolist() == Y_names,
        #             lambda: 'Y_pred columns "{}" do not matched expected columns "{}"'
        #             .format(Y_pred.columns.tolist(), Y_names))

        # Convert true response values as well as predictions to data frames
        Y_pred, Y_df, _ = _convert_to_data_frames(Y_pred, Y, None, Y_names=Y_names, Y_pred_index=X.index)

        # Initialize result for model using the minimal amount of information required
        # *Note: all X_* and Y_* values should be data frames at this point
        fold_res = TrainingModelResult(
            fold=0,
            clf_name=model_res.clf_name,
            clf=model_res.clf,
            mode=train_res.mode,
            Y_pred=Y_pred,
            Y_test=Y_df,
            X_names=X.columns.tolist(),
            Y_names=Y_names
        )

        if train_res.trainer_config.keep_test_data_enabled(model_res.clf_name):
            fold_res.X_test = X

        if train_res.mode == MODE_CLASSIFIER and train_res.trainer_config.predict_proba:
            # Compute probability predictions and put off conversion to data frame until later
            # due to the complexity of this in multi-class, multi-label situations (typically
            # all use in those cases will involve a user-specified function to interpret results
            # from models since they vary so widely within sklearn)
            fold_res.Y_proba = model_res.clf.predict_proba(X)

        fold_res.validate()

        res.append(fold_res)
    return TrainingResampleResult(model_results=res)


def run_fold(args):
    i, train, test, clfs, mode, X, Y, W, config = args
    X_train, X_test, Y_train, Y_test = X.iloc[train], X.iloc[test], Y.iloc[train], Y.iloc[test]

    # Initialize sample weighting
    if W is None:
        W_train, W_test = None, None
    else:
        W_train, W_test = W.iloc[train], W.iloc[test]

    # Run the data preparation function if one was configured, and ensure that the results from this
    # function are still DataFrames or Series, not numpy arrays
    if config.data_prep_fn is not None:
        X_train, X_test, Y_train, Y_test = _data_prep(
            config.data_prep_fn,
            X_train, X_test,
            Y_train, Y_test,
            W_train=W_train, W_test=W_test
        )

    res = []

    for clf in clfs:

        # If the estimator "implementation" is callable, assume it is a factory method for vanilla instances
        if callable(clf[CLF_IMPL]):
            est = clf[CLF_IMPL](i)
        # Otherwise, clone the presumably cloneable estimator instance
        else:
            est = base.clone(clf[CLF_IMPL])

        log(
            'Running model {} ({}) on fold {} ==> dim(X_train) = {}, dim(X_test) = {}, '
            'dim(Y_train) = {}, dim(Y_test) = {} [dim(W_train) = {}, dim(W_test) = {}'
            .format(
                clf[CLF_NAME], _get_clf_name(est), i+1, X_train.shape, X_test.shape,
                Y_train.shape, Y_test.shape,
                (None if W_train is None else W_train.shape),
                (None if W_test is None else W_test.shape)
            )
        )

        # If explicitly provided, use a custom model fitting function
        if config.model_fit_fn is not None:
            config.model_fit_fn(
                clf[CLF_NAME], est, X_train, Y_train, X_test, Y_test, i == -1,
                W_train=W_train, W_test=W_test
            )
        # Otherwise, run base estimator fit
        else:
            if W_train is None:
                est.fit(X_train, Y_train)
            else:
                est.fit(X_train, Y_train, sample_weight=W_train)

        # Make predictions on test data features
        Y_pred = est.predict(X_test)

        # Convert true response values as well as predictions to data frames
        Y_pred, Y_test_df, Y_train_df = _convert_to_data_frames(Y_pred, Y_test, Y_train)

        # Initialize result for model + fold using the minimal amount of information required
        # *Note: all X_* and Y_* values should be data frames at this point
        fold_res = TrainingModelResult(
            fold=i + 1,
            clf_name=clf[CLF_NAME],
            clf=est,
            mode=mode,
            Y_pred=Y_pred,
            Y_test=Y_test_df,
            X_names=X_train.columns.tolist(),
            Y_names=Y_train_df.columns.tolist()
        )

        # If configured to do so, add extra data associated with the results, which are often
        # expensive in terms of storage space so these things must be explicitly enabled
        if config.keep_training_data_enabled(clf[CLF_NAME]):
            fold_res.X_train = X_train
            fold_res.Y_train = Y_train_df
        if config.keep_test_data_enabled(clf[CLF_NAME]):
            fold_res.X_test = X_test
        if mode == MODE_CLASSIFIER and config.predict_proba:
            # Compute probability predictions and put off conversion to data frame until later
            # due to the complexity of this in multi-class, multi-label situations (typically
            # all use in those cases will involve a user-specified function to interpret results
            # from models since they vary so widely within sklearn)
            fold_res.Y_proba = est.predict_proba(X_test)

        fold_res.validate()

        res.append(fold_res)
    return TrainingResampleResult(model_results=res)


class TrainerConfig(object):

    def __init__(self, refit=True, runcv=True, keep_training_data=False, keep_test_data=False,
                 predict_proba=True, data_prep_fn=None, model_fit_fn=None):
        """
        :param refit: Boolean indicating whether or not the given models should also be trained on the full dataset;
            defaults to false
        :param runcv: Boolean indicating whether or not the given models should be run in cross validation or just
            trained on the training data a single time
        :param keep_training_data: Boolean indicating whether or not test datasets should be preserved in
            results, or a list of clf names for which this condition will be true
        :param keep_test_data: Boolean indicating whether or not test datasets should be preserved in
            results, or a list of clf names for which this condition will be true
        :param predict_proba: Boolean indicating whether or not class probability predictions should be made
            for classifiers; defaults to false
        :param data_prep_fn: Optional function to be passed X_train, X_test, Y_train and Y_test for each fold.  This
            function is expected to return a new value for each of those given as a four-value tuple.  This is
            commonly useful for implementing downsampling/upsampling prior to training, or any other more
            complicated sort of cross-validation.

            Example: lambda X_train, X_test, Y_train, Y_test: \
                X_train.head(10), X_test.head(10), Y_train.head(10), Y_test.head(10)
        :param model_fit_fn: Optional custom fitting function taking a model name, model object,
            training/testing sets, and a flag indicating whether or not this is a "refit" on whole training set.

            Example: lambda clf_name, clf, X_train, Y_train, X_test, Y_test, is_refit: clf.fit(X_train, Y_train)
        """
        self.refit = refit
        self.runcv = runcv
        self.keep_training_data = keep_training_data
        self.keep_test_data = keep_test_data
        self.predict_proba = predict_proba
        self.data_prep_fn = data_prep_fn
        self.model_fit_fn = model_fit_fn

    def keep_training_data_enabled(self, clf):
        if isinstance(self.keep_training_data, bool):
            return self.keep_training_data
        return clf in self.keep_training_data

    def keep_test_data_enabled(self, clf):
        if isinstance(self.keep_test_data, bool):
            return self.keep_test_data
        return clf in self.keep_test_data


class Trainer(object):

    def __init__(self, default_config=None):
        if default_config is None:
            self.default_config = TrainerConfig()
        else:
            if not isinstance(default_config, TrainerConfig):
                raise ValueError('Default configuration must be an instance of {}'.format(TrainerConfig))
            self.default_config = default_config

    def __init(self):
        pass

    @staticmethod
    def _validate_mode(mode):
        if mode not in MODES:
            raise ValueError('Given mode "{}" is not valid.  Must be one of: {}'.format(mode, MODES))

    def train_classifiers(self, X, Y, clfs, cv, config=None, W=None, **kwargs):
        return self._train(X, Y, clfs, cv, MODE_CLASSIFIER, config=config, W=W, **kwargs)

    def train_regressors(self, X, Y, clfs, cv, config=None, W=None, **kwargs):
        return self._train(X, Y, clfs, cv, MODE_REGRESSOR, config=config, W=W, **kwargs)

    def _train(self, X, Y, clfs, cv, mode, config=None, W=None, **kwargs):
        """ Fit multiple classification or regression model concurrently

        :param X: Feature data set (must be a pandas DataFrame)
        :param Y: Response to predict (must be a pandas Series or DataFrame)
        :param clfs: Estimators to fit (must be a dictionary of 'name': 'value' pairs).  For Example:
            {'logreg': LogisticRegression(),
            'gbr': GridSearchCV(GradientBoostingClassifier(), {'n_estimators': [10, 25, 50, 100]})}

            Note that this dictionary will automatically be sorted by key for the sake of repeatability in modeling runs
        :param cv: Cross validation definition
        :param mode: Either 'classifier' (#MODE_CLASSIFIER) or 'regressor' (#MODE_REGRESSOR)
        :param config: Trainer configuration; controls options like retention of training data, whether or not to
            refit a model to the entire training set, custom functions for model fitting, etc. See \code{TrainerConfig}
        :param W: Optional sample weight (must be pandas Series)
        :param kwargs:
            Logging kwargs:
                log_*: log_file and log_format can be passed to determine logging output location and style
            Joblib kwargs:
                par_*: Arguments, prefixed by 'par_', to be passed to parallel executor (e.g. par_n_jobs, par_backend)
        :return: TrainerResult
        """
        Trainer._validate_mode(mode)
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Feature data (X) must be a DataFrame')
        if not isinstance(Y, pd.DataFrame) and not isinstance(Y, pd.Series):
            raise ValueError('Response data (Y) must be a DataFrame or Series')
        if W is not None and not isinstance(W, pd.Series):
            raise ValueError('Sample weight (W) must be a Series')

        if config is None:
            config = self.default_config

        # Sort models by key/name and place into dictionary with name and value separated
        clfs = [{CLF_NAME: k, CLF_IMPL: clfs[k]} for k in sorted(list(clfs.keys()))]

        par_kwargs, kwargs = arg_utils.parse_kwargs(kwargs, 'par_')
        log_kwargs, kwargs = arg_utils.parse_kwargs(kwargs, 'log_')
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments are not valid: {}'.format(kwargs.keys()))

        args_list = [(i, train, test, clfs, mode, X, Y, W, config) for i, (train, test) in enumerate(cv)]

        log_file = log_kwargs.get('file', DEFAULT_LOG_FILE)
        set_logger(log_file, log_kwargs.get('format', DEFAULT_LOG_FORMAT))

        cv_res = None
        if config.runcv:
            print('Beginning cross validation (see {} for progress updates)'.format(log_file))
            cv_res = Parallel(**par_kwargs)(delayed(run_fold)(args) for args in args_list)
            log('CV Complete ({} model(s) run)'.format(len(clfs)))

        refit_res = None
        if config.refit:
            idx = np.arange(0, len(Y))
            args = (-1, idx, idx, clfs, mode, X, Y, W, config)

            print('Beginning model refitting')
            refit_res = run_fold(args)
            log('Model refitting complete ({} model(s) run)'.format(len(clfs)))

        print('Training complete')
        log_hr()

        return TrainingResult(
            resample_results=cv_res, refit_result=refit_res,
            trainer_config=config, mode=mode
        )

    def predict(self, train_res, X, Y=None):
        pred_res = run_predict(train_res, X, Y=Y)
        return TrainingResult(
            resample_results=[pred_res], refit_result=None,
            trainer_config=train_res.trainer_config, mode=train_res.mode
        )


