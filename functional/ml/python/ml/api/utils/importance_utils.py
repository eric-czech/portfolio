
import pandas as pd
from ml.api.utils import array_utils, estimator_utils
import logging
logger = logging.getLogger(__name__)


##### Learning Model Categorizations #####
# Note that string class names are used here for two reasons:
# 1. Avoiding import statements for each one of these classes
# 2. Ambiguous situtations where a single model class may exist in two places within sklearn (eg ExtraTreesClassifier)

SINGLE_TASK_LINEAR_REGRESSORS = [
    'Lars',
    'LarsCV',
    'LassoCV',
    'LassoLars',
    'LassoLarsCV',
    'LassoLarsIC',
    'Ridge',
    'RidgeCV',
    'BayesianRidge'
]

SINGLE_TASK_LINEAR_CLASSIFIERS = [
    'RidgeClassifier',
    'RidgeClassifierCV',
    'LogisticRegression',
    'LogisticRegressionCV'
]

TREE_REGRESSORS = [
    'ExtraTreesRegressor',
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'DecisionTreeRegressor'
]

TREE_CLASSIFIERS = [
    'ExtraTreesClassifier',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'DecisionTreeClassifier'
]


def _get_coef_single_task_linear_regressor(clf, X_names, Y_names):
    """
    Get coefficients from a regressor for a single task

    Coefficient array for regressor is assumed to have shape (n_features,)

    :param clf: Regressor instance
    :param X_names: Names of features
    :param Y_names: Names of tasks (should only be one in this case)
    :return: DataFrame with Feature, Task, and "Value" associated with importance (i.e. coefficient)
    """
    assert hasattr(clf, 'coef_'), \
        'Regressor does not have "coef_" attribute'
    coef = clf.coef_
    assert len(Y_names) == 1, \
        'Only one task should be available for importance from single task regressor (given tasks: {})'\
        .format(Y_names)
    assert len(coef.shape) == 1, \
        'Coefficient array was expected to be 1 dimensional (coef shape found = {})'.format(coef.shape)
    assert len(coef) == len(X_names), \
        'Number of feature coefficients ({}) does not equal number of given feature names ({})'\
        .format(len(coef), len(X_names))

    return pd.DataFrame({
        'Feature': X_names,
        'Task': Y_names[0],
        'Value': coef
    })


def _get_imp_tree_regressor(clf, X_names, Y_names):
    """
    Get feature importances from a tree regressor for any number of tasks

    Feature importance array for regressor is assumed to have shape (n_features,)

    Note that this is unique to tree regressors to have one single feature importance array
    for all outcome tasks.

    :param clf: Regressor instance
    :param X_names: Names of features
    :param Y_names: Names of tasks (should only be one in this case)
    :return: DataFrame with Feature, Task, and "Value" associated with importance (i.e. feature importance)
    """
    assert hasattr(clf, 'feature_importances_'), \
        'Regressor does not have "feature_importances_" attribute'
    feat_imp = clf.feature_importances_
    assert len(feat_imp.shape) == 1, \
        'Feature importance array was expected to be 1 dimensional (shape found = {})'.format(feat_imp.shape)
    assert len(feat_imp) == len(X_names), \
        'Number of feature importances ({}) does not equal number of given feature names ({})'\
        .format(len(feat_imp), len(X_names))

    d = []
    for y_name in Y_names:
        d.append(pd.DataFrame({
            'Feature': X_names,
            'Task': y_name,
            'Value': feat_imp
        }))
    return pd.concat(d)


def _get_imp_tree_classifier(clf, X_names, Y_names):
    """
    Get feature importances from a tree classifer for any number of tasks and classes

    Assumptions:
        - feature_importances_ array for classifier is assumed to have shape (n_features,)
        - classes_ array is assumed to be list of n_tasks arrays with each of length n_classes for corresponding task

    Note that this is unique to tree classifiers to have one single feature importance array
    for all outcome tasks AND classes.

    :param clf: Classifier instance
    :param X_names: Names of features
    :param Y_names: Names of tasks (should only be one in this case)
    :return: DataFrame with Feature, Task, Class, and "Value" associated with importance (i.e. feature importance)
    """
    assert hasattr(clf, 'feature_importances_'), \
        'Regressor does not have "feature_importances_" attribute'
    assert hasattr(clf, 'classes_'), \
        'Classifier does not have "classes_" attribute'

    feat_imp = clf.feature_importances_
    assert len(feat_imp.shape) == 1, \
        'Feature importance array was expected to be 1 dimensional (shape found = {})'.format(feat_imp.shape)
    assert len(feat_imp) == len(X_names), \
        'Number of feature importances ({}) does not equal number of given feature names ({})'\
        .format(len(feat_imp), len(X_names))

    d = []
    for i, y_name in enumerate(Y_names):
        for class_name in clf.classes_[i]:
            d.append(pd.DataFrame({
                'Feature': X_names,
                'Task': y_name,
                'Class': class_name,
                'Value': feat_imp
            }))
    return pd.concat(d)


def _get_coef_single_task_linear_classifier(clf, X_names, Y_names):
    """
    Get coefficients from a classifier that may be multiclass

    Coefficient array for classifier is assumed to have shape (n_classes, n_features)
    :param clf: Classifier instance
    :param X_names: Names of features
    :param Y_names: Names of tasks (should only be one in this case)
    :return: DataFrame with Feature, Task, Class, and "Value" associated with importance (i.e. coefficient)
    """
    assert hasattr(clf, 'coef_'), \
        'Classifier does not have "coef_" attribute'
    assert hasattr(clf, 'classes_'), \
        'Classifier does not have "classes_" attribute'

    coef = clf.coef_
    assert len(Y_names) == 1, \
        'Only one task should be available for importance from single task classifier (given tasks: {})'\
        .format(Y_names)
    assert coef.shape[0] == len(clf.classes_), \
        'Number of coefficient arrays does not equal number of classes'
    assert coef.shape[1] == len(X_names), \
        'Number of feature coefficients ({}) does not equal number of given feature names ({})'\
        .format(coef.shape[1], len(X_names))

    d = []
    class_names = clf.classes_
    for i, class_name in enumerate(class_names):
        d.append(pd.DataFrame({
            'Feature': X_names,
            'Task': Y_names[0],
            'Class': class_name,
            'Value': coef[i]
        }))

    return pd.concat(d)


def _get_importance(clf, clf_class, X_names, Y_names):
    # Determine which type of estimator we're dealing with here and attempt to
    # extract feature importance data frame from it

    if clf_class in SINGLE_TASK_LINEAR_REGRESSORS:
        return _get_coef_single_task_linear_regressor(clf, X_names, Y_names)

    if clf_class in SINGLE_TASK_LINEAR_CLASSIFIERS:
        return _get_coef_single_task_linear_classifier(clf, X_names, Y_names)

    if clf_class in TREE_REGRESSORS:
        return _get_imp_tree_regressor(clf, X_names, Y_names)

    if clf_class in TREE_CLASSIFIERS:
        return _get_imp_tree_classifier(clf, X_names, Y_names)

    # If this estimator is not yet supported, return None
    return None


def get_estimator_feature_importance(clf, X_names, Y_names, clf_name=None):
    """
    Get feature importance for an estimator

    :param clf: Sklearn estimator instance
    :param X_names: Feature names
    :param Y_names: Outcome task names
    :param clf_name: Optional name associated with estimator (useful for logging errors/warnings)
    :return: DataFrame with Feature, Task, Class (if classifier), and "Value" associated with importance
    """

    # "Resolve" estimator to make sure all further processing uses an actual learning estimator
    # and not a pipeline or search estimator
    clf = estimator_utils.resolve_estimator(clf)

    clf_class = clf.__class__.__name__

    # Default estimator name to class name if not given
    if clf_name is None:
        clf_name = clf_class

    try:
        d = _get_importance(clf, clf_class, X_names, Y_names)
        if d is None:
            logger.warn(
                'Feature importance calculation for estimator type {} (name = {}) is not yet supported'
                .format(clf_class, clf_name)
            )
        return d
    # Make sure to catch-all errors here so that the estimator name can be determined from the error message
    # (otherwise, these errors are very annoying if it's not immediately clear what model is the problem)
    except Exception as e:
        raise ValueError('Failed to determine feature importance for estimator "{}"'.format(clf_name)) from e

