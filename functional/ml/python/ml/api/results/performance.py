
import pandas as pd
from ml.api.results.predictions import PREDICTION_PANEL, META_PANEL
from ml.api.results.predictions import MODEL_PROPERTY, FOLD_PROPERTY
from ml.api.results.predictions import VALUE_PRED_PREFIX, CLASS_PRED_PREFIX, CLASS_PROB_PREFIX, \
    VALUE_TRUE_PREFIX, CLASS_TRUE_PREFIX, TASK_PROPERTY
from ml.api.results import properties
from ml.api.constants import MODE_CLASSIFIER, MODE_REGRESSOR

METRIC_PROPERTY = 'Metric'


def extract(train_res, d_pred, score_fn):
    """
    Compute performance metrics based on training results and prediction data

    :param train_res: Training results
    :param d_pred: DataFrame from `predictions.extract`
    :param score_fn: Function with signature fn(y_true, y_class[, y_proba]) to return dict with keys equal
        to performance metric name and values equal to the numeric value of that metric (note that this
        function must always return a value for each metric)
    :return: DataFrame containing performance metrics for each task and model
    """

    d = d_pred[[PREDICTION_PANEL, META_PANEL]]

    tasks = properties.get_prediction_tasks(train_res)

    def get_scores(g):
        g = g[PREDICTION_PANEL]
        res = {}
        for task in tasks:
            if train_res.mode == MODE_CLASSIFIER:
                y_class = g[CLASS_PRED_PREFIX + ':' + task]
                y_true = g[CLASS_TRUE_PREFIX + ':' + task]
                k = CLASS_PROB_PREFIX + ':' + task
                y_proba = g.filter(regex=k)
                y_proba = y_proba if len(y_proba.columns) > 0 else None
                scores = score_fn(y_true, y_class, y_proba)

            elif train_res.mode == MODE_REGRESSOR:
                y_pred = g[VALUE_PRED_PREFIX + ':' + task]
                y_true = g[VALUE_TRUE_PREFIX + ':' + task]
                scores = score_fn(y_true, y_pred)
            else:
                raise NotImplementedError('Training mode "{}" not implemented'.format(train_res.mode))
            res.update({(k, task): scores[k] for k in scores})
        res = pd.Series(res)
        res.index = pd.MultiIndex.from_tuples(res.index.values)
        res.index.names = [METRIC_PROPERTY, TASK_PROPERTY]
        return res

    # Group by model name and fold id, and compute scores for each task
    d = d.groupby([(META_PANEL, MODEL_PROPERTY), (META_PANEL, FOLD_PROPERTY)]).apply(get_scores)

    # Cleanup margin names
    d.index.names = [n[1] for n in d.index.names]
    d.columns.names = [METRIC_PROPERTY, TASK_PROPERTY]

    return d


def sample_tasks(d_score, limit=10, random_state=None):
    tasks = pd.Series(d_score.columns.get_level_values(TASK_PROPERTY)).drop_duplicates()
    return tasks.sample(n=min(limit, len(tasks)), random_state=random_state)


def melt(d_score):
    """
    Melts performance data with results in columns into rows, one for each model, fold, task, and metric

    :param d_score: DataFrame from `performance.extract`
    :return: DataFrame with Model, Fold, Task, Metric (eg RMSE or Accuracy), and Value
    """
    # Stack model scores so that instead of one column per task + metric, we have a single row
    # for each model, fold, task, and metric
    d_score = d_score.stack(level=[TASK_PROPERTY, METRIC_PROPERTY])
    d_score.name = 'Value'
    d_score = d_score.reset_index().reset_index(drop=True)
    return d_score


def visualize_plotly(d_score, kind='box', aggfunc=None, auto_plot=True, layout_kwargs={}):
    """
    Generate Plot.ly boxplot or scatterplot of model performance scores

    :param d_score: DataFrame from `performance.melt`
    :param kind: One of 'scatter' or 'box'
    :param aggfunc: Function to use to aggregate (e.g. 'mean', 'median') feature importance values
        across folds; if left as None, all values will be plotted rather than being aggregated first
    :param auto_plot: Whether or not to plot figures immediately or just return them
    :param layout_kwargs: Optional extra parameters for Plotly Layout objects
    :return: Plotly Figure instance dict keyed by task name
    """
    import plotly.graph_objs as go

    figs = {}

    # Apply aggregation across folds, if configured to do so
    if aggfunc is not None:
        d_score = (
            d_score.groupby(['Task', 'Model', 'Metric'])
            .agg({'Value': aggfunc})
            .reset_index()
            .assign(Fold=0)
        )

    # Loop through tasks, and create a new figure for each one
    for task, g_task in d_score.groupby('Task'):
        traces = []

        # Determine the sort order for the plot based on the mean importance value across
        # models for each feature
        o = g_task.groupby('Model')['Value'].mean().sort_values().index.values

        # Loop through metrics and generate a trace for each
        for k, g in g_task.groupby('Metric'):

            # Apply sort order
            g = g.set_index('Model').loc[o].reset_index()

            # Generate specified type of trace
            if kind == 'box':
                traces.append(go.Box(
                    x=g['Model'], y=g['Value'],
                    name=k
                ))
            elif kind == 'scatter':
                traces.append(go.Scatter(
                    x=g['Model'], y=g['Value'], name=k, mode='markers',
                    text=g['Fold'].apply(lambda v: 'Fold {}'.format(v))
                ))
            else:
                raise ValueError('Plot type (i.e. "kind" argument) "{}" is not valid'.format(kind))

        # Add figure to rolling result
        layout_args = dict(layout_kwargs)
        if 'title' not in layout_args:
            layout_args['title'] = task
        fig = go.Figure(data=traces, layout=go.Layout(**layout_args))
        figs[task] = fig

    # Plot figures immediately, if configured to do so
    if auto_plot:
        from py_utils import plotly_utils
        return [plotly_utils.iplot(fig) for fig in figs.values()]
    else:
        return figs


def visualize_seaborn(d_score, figsize=2, figaspect=5, legend_margin=(1.13, 1)):
    """
    Generates seaborn boxplot of performance metrics

    :param d_score: DataFrame from `performance.melt`
    :param figsize: Seaborn FacetGrid size
    :param figaspect: Seaborn FacetGrid aspect
    :param legend_margin: Annoying property that frequently needs to be adjusted to make legends not
        interfere with main plotting area (default works in some cases)
    :return: Seaborn FacetGrid
    """
    import seaborn as sns

    # Create a boxplot per task (in rows) with model names on the x axis and colors denoting metrics
    g = sns.FacetGrid(d_score, row=TASK_PROPERTY, size=figsize, aspect=figaspect, legend_out=False)

    # Map grid to boxplot function
    def plot_data(*args, **kwargs):
        sns.boxplot(data=kwargs['data'], x=MODEL_PROPERTY, y='Value', hue=METRIC_PROPERTY)
    g.map_dataframe(plot_data)
    [ax.legend(bbox_to_anchor=legend_margin) for ax in g.axes.ravel()]

    # Return the plot grid and the underlying data
    return g


def visualize(d_score, backend='plotly', metrics=None, tasks=None, **kwargs):
    """
    Generates visualization of performance metrics

    See individual backend plotting functions below for more options/details:
    1. `visualize_seaborn`
    2. `visualize_plotly`

    :param d_score: DataFrame from `performance.melt`
    :param backend: Backend to use for plotting; one of ['seaborn', 'plotly']
    :param metrics: Optional performance metrics to include (if None all will be used)
    :param tasks: Optional outcome tasks to include (if None all will be used)
    :return: plot objects [varies based on backend]
    """

    # Filter to only the given metrics, if any were provided
    if metrics is not None:
        d_score = d_score[d_score[METRIC_PROPERTY].isin(metrics)]

    # Filter to only the given tasks, if any were provided
    if tasks is not None:
        d_score = d_score[d_score[TASK_PROPERTY].isin(tasks)]

    backends = ['seaborn', 'plotly']
    assert backend in backends, 'Backend can currently only be one of {}'.format(backends)

    if backend == 'seaborn':
        return visualize_seaborn(d_score, **kwargs)
    if backend == 'plotly':
        return visualize_plotly(d_score, **kwargs)

    return None
