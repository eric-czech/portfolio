
import pandas as pd
import numpy as np
from ml.api.constants import *
from ml.api.results import properties

import logging
logger = logging.getLogger(__name__)

PREDICTION_PANEL = 'Predictions'
FEATURE_PANEL = 'Features'
META_PANEL = 'Metadata'
FOLD_PROPERTY = 'Fold'
MODEL_PROPERTY = 'Model'
CATEGORY_PROPERTY = 'Category'
TASK_PROPERTY = 'Task'
CLASS_PRED_PREFIX = 'Class:Predicted'
CLASS_TRUE_PREFIX = 'Class:Actual'
CLASS_PROB_PREFIX = 'Class:Probability'
VALUE_PRED_PREFIX = 'Value:Predicted'
VALUE_TRUE_PREFIX = 'Value:Actual'


# def get_default_proba_fn(task_name, classes):
#     """
#     Appropriate default probability prediction function for use in `extract` with two-class, single-task result
#     """
#     def proba_fn(clf_name, clf, Y_proba):
#         assert Y_proba.shape[1] == len(classes)
#         return pd.DataFrame(Y_proba, columns=['{}:{}'.format(task_name, c) for c in classes])
#     return proba_fn


def _get_probability_predictions(model_res, proba_fn, Y_proba, Y_names):
    d = {}
    if proba_fn is not None:
        df_proba = proba_fn(model_res.clf_name, model_res.clf, Y_proba)
        for c in df_proba:
            d['{}:{}'.format(CLASS_PROB_PREFIX, c)] = df_proba[c].values
    else:
        is_single_task = len(Y_names) == 1
        proba_2d = isinstance(Y_proba, np.ndarray) and len(Y_proba.shape) == 2
        has_classes = hasattr(model_res.clf, 'classes_')
        if is_single_task and proba_2d and has_classes:
            class_names = model_res.clf.classes_
            for i, class_name in enumerate(class_names):
                d['{}:{}:{}'.format(CLASS_PROB_PREFIX, Y_names[0], class_name)] = Y_proba[:, i]
        else:
            logger.warning(
                'Model "{}" produced probability predictions but these are predictions are for '
                'multiple output classification problems and will not be automatically '
                'interpreted (since the representation of such predictions differs by model)'
                .format(model_res.clf_name)
            )
    return d


def extract(train_res, proba_fn=None):
    """
    Returns per-fold predictions from a cross-validation result
    :param train_res: Training result from \code{Trainer.train}
    :param proba_fn: Function to take a model name, fit instance, and probability predictions to
        produce a data frame containing the task and class names as columns where each column contains
        probability predictions for a single task and class label.  For example, given two tasks Task1 and Task2,
        this function should return a data frame with columns like ['Task1:Class1', 'Task1:Class2', ..., 'Task2:ClassN']
    :return: Data frame containing predictions
    """
    d_pred = []

    for resample_res in train_res.resample_results:
        for model_res in resample_res.model_results:

            # Initialize resulting predictions data frame as the test feature data, if present,
            # or an empty data frame.  Note that the original feature data and the prediction data
            # will be present in the same from but with a nested column index and two labels at
            # the top level separating them (`feat_label` and `pred_label` respectively)
            if model_res.has_test_data():
                # model_res.clf_name
                d = model_res.X_test.copy()
                d.columns = pd.MultiIndex.from_tuples([(FEATURE_PANEL, c) for c in d])
                d[(META_PANEL, FOLD_PROPERTY)] = model_res.fold
            else:
                # d = pd.DataFrame(
                #     {(META_PANEL, FOLD_PROPERTY): np.repeat(model_res.fold, len(model_res.Y_test))},
                #     index=model_res.Y_test.index
                # )
                d = pd.DataFrame(
                    {(META_PANEL, FOLD_PROPERTY): np.repeat(model_res.fold, len(model_res.Y_pred))},
                    index=model_res.Y_pred.index
                )

            # Add predictions and actuals to resulting frame within the `pred_label` panel
            def append(col, value, panel=PREDICTION_PANEL):
                d[(panel, col)] = value

            Y_pred = model_res.Y_pred
            Y_true = model_res.Y_test
            Y_proba = model_res.Y_proba
            Y_names = model_res.Y_names

            if train_res.mode == MODE_CLASSIFIER:
                # Add class probability predictions, which can be tricky in the case of multioutput classification
                # models since new fields have to be added for each output and all associated classes
                if Y_proba is not None:
                    d_pred_proba = _get_probability_predictions(model_res, proba_fn, Y_proba, Y_names)
                    for c in d_pred_proba:
                        append(c, d_pred_proba[c])

                # Add predicted and actual class labels
                for i, task_name in enumerate(model_res.Y_names):
                    append('{}:{}'.format(CLASS_PRED_PREFIX, task_name), Y_pred.iloc[:, i])
                    if Y_true is not None:
                        append('{}:{}'.format(CLASS_TRUE_PREFIX, task_name), Y_true.iloc[:, i])

            elif train_res.mode == MODE_REGRESSOR:
                # Add predicted and actual values
                for i, task_name in enumerate(model_res.Y_names):
                    append('{}:{}'.format(VALUE_PRED_PREFIX, task_name), Y_pred.iloc[:, i])
                    if Y_true is not None:
                        append('{}:{}'.format(VALUE_TRUE_PREFIX, task_name), Y_true.iloc[:, i])
            else:
                raise NotImplementedError(
                    'Prediction extraction for training mode "{}" not yet implemented'
                    .format(train_res.mode)
                )
            append(MODEL_PROPERTY, model_res.clf_name, panel=META_PANEL)

            d_pred.append(d)

    # Concatenate all data frames for each model + fold and sort columns
    d_pred = pd.concat(d_pred).sort_index(axis=1)

    # Name margins
    d_pred.columns.names = [CATEGORY_PROPERTY, TASK_PROPERTY]
    return d_pred


def melt(train_res, d_pred):
    """
    Melt predictions from columns into rows with Task, Model, Fold, Actual and Predicted values

    * Currently only supports regression

    :param train_res: Training results
    :param d_pred: DataFrame from `predictions.extract`
    :return: Flattened data from with individual data points in rows (one per model + fold + task)
    """
    mode = train_res.mode
    tasks = properties.get_prediction_tasks(train_res)
    d_pred = d_pred[[META_PANEL, PREDICTION_PANEL]]

    if mode == MODE_REGRESSOR:
        d = []
        for task in tasks:
            y_pred = d_pred[(PREDICTION_PANEL, '{}:{}'.format(VALUE_PRED_PREFIX, task))]
            y_true = d_pred[(PREDICTION_PANEL, '{}:{}'.format(VALUE_TRUE_PREFIX, task))]
            d.append(pd.DataFrame({
                'Predicted': y_pred,
                'Actual': y_true,
                'Task': task,
                'Model': d_pred[(META_PANEL, MODEL_PROPERTY)],
                'Fold': d_pred[(META_PANEL, FOLD_PROPERTY)]
            }))
        d = pd.concat(d)
        return d
    else:
        raise NotImplementedError('Prediction melting not implemented for "{}" training mode'.format(mode))


def visualize_seaborn(d_pred, mode, share_axes=True, figaspect=1, figsize=5):
    """
    Generates seaborn scatterplot of predictions by task and model

    :param d_pred: DataFrame from `predictions.melt`
    :param mode: Prediction source type; one of ['classifier', 'regressor']
    :param figsize: Seaborn FacetGrid size
    :param figaspect: Seaborn FacetGrid aspect
    :param share_axes: Seaborn FacetGrid share axes flag
    :return: Seaborn FacetGrid
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    if mode == MODE_REGRESSOR:
        g = sns.FacetGrid(
            d_pred, row='Model', col='Task', hue='Fold',
            margin_titles=True, size=figsize, aspect=figaspect,
            sharex=share_axes, sharey=share_axes
        )
        g.map(plt.scatter, 'Predicted', 'Actual')
        return g
    else:
        raise NotImplementedError('Prediction visualization not implemented for "{}" training mode'.format(mode))
        # TODO: Implement confusion matrix viz for classification


def visualize_plotly(d_pred, mode, bin_cols=None, bin_label_fn=None, count_transform_fn=None,
                     heatmap_kwargs={}, layout_kwargs={}, auto_plot=True):
    """
    Generates Plotly prediction heatmaps

    :param d_pred: DataFrame from `predictions.melt`
    :param mode: Prediction source type; one of ['classifier', 'regressor']
    :param bin_cols: Optional 2-item tuple with name of columns containing binned actual values and binned
        predictions (in that order); if not given then for regression for example, the predicted and actuals will
        be binned automatically
    :param bin_label_fn: Optional function that will take a bin name and return a string label for axes labels
    :param count_transform_fn: Function to apply to counts associated with bins in heatmap (np.log10 for example)
    :param auto_plot: Whether or not to plot figures immediately or just return them
    :param heatmap_kwargs: Optional extra parameters for Plotly Heatmap
    :param layout_kwargs: Optional extra parameters for Plotly Layout objects
    :return: Plotly Figure instance dict keyed by task and model name
    """
    import plotly.graph_objs as go

    figs = {}
    if mode == MODE_REGRESSOR:

        # Create binned prediction + actual values if not already present
        if bin_cols is None:
            d_pred['PredictedRange'] = pd.cut(d_pred['Predicted'], bins=12)
            d_pred['ActualRange'] = pd.cut(d_pred['Actual'], bins=12)
            bin_cols = ['ActualRange', 'PredictedRange']

        # Default bin label function to string conversion
        if bin_label_fn is None:
            bin_label_fn = str

        # Loop through each model + task combination and generate a heatmap of predicted vs actual frequencies
        for k, g in d_pred.groupby(['Model', 'Task']):
            model, task = k
            key = '{} ({})'.format(model, task)

            # Group by predicted and actual bin names and determine frequency of each
            d = g.groupby([bin_cols[0], bin_cols[1]]).size().unstack()

            # Apply overall count transformation if given
            if count_transform_fn is not None:
                d = d.applymap(count_transform_fn)

            # Relabel the bin names (or at least convert them to string)
            d.columns = [bin_label_fn(c) for c in d]      # Predictions
            d.index = [bin_label_fn(c) for c in d.index]  # Actuals
            assert np.all([type(v) == str for v in d]), 'Bin labels (for columns) must be strings'
            assert np.all([type(v) == str for v in d.index]), 'Bin labels (for index) must be strings'

            # Generate heatmap and layout arguments
            heatmap_args = dict(heatmap_kwargs)
            if 'colorscale' not in heatmap_args:
                heatmap_args['colorscale'] = 'Viridis'

            layout_args = dict(layout_kwargs)
            if 'xaxis' not in layout_args:
                layout_args['xaxis'] = dict(title='Predicted Range')
            if 'yaxis' not in layout_args:
                layout_args['yaxis'] = dict(title='Actual Range')
            if 'title' not in layout_args:
                layout_args['title'] = key

            # Create heatmap and note that reversal of x and y to index and columns here is intentional
            trace = go.Heatmap(z=d.values, y=d.index.values, x=d.columns.values, **heatmap_args)
            fig = go.Figure(data=[trace], layout=go.Layout(**layout_args))
            figs[key] = fig
    else:
        raise NotImplementedError('Prediction visualization not implemented for "{}" training mode'.format(mode))

    # Plot figures immediately, if configured to do so
    if auto_plot:
        from py_utils import plotly_utils
        return [plotly_utils.iplot(fig) for fig in figs.values()]
    else:
        return figs


def _visualize(d_pred, mode, backend='seaborn', **kwargs):
    """
    Generates visualization of predictions from fit models

    See individual backend plotting functions below for more options/details:
    1. `visualize_seaborn`
    2. `visualize_plotly`

    :param d_pred: DataFrame from `predictions.melt`
    :param mode: Prediction source type; one of ['classifier', 'regressor']
    :param backend: Backend to use for plotting; one of ['seaborn', 'plotly']
    :return: plot objects [varies based on backend]
    """
    backends = ['seaborn', 'plotly']
    assert backend in backends, 'Backend can currently only be one of {}'.format(backends)

    if backend == 'seaborn':
        return visualize_seaborn(d_pred, mode, **kwargs)
    if backend == 'plotly':
        return visualize_plotly(d_pred, mode, **kwargs)

    return None


def visualize(train_res, d_pred, backend='seaborn', **kwargs):
    """
    Generates visualization of predictions from fit models

    See individual backend plotting functions below for more options/details:
    1. `visualize_seaborn`

    :param train_res: Training results
    :param d_pred: DataFrame from `predictions.melt`
    :param backend: Backend to use for plotting; one of ['seaborn']
    :return: plot objects [varies based on backend]
    """
    return _visualize(d_pred, train_res.mode, backend, **kwargs)
