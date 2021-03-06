
import numpy as np
import pandas as pd
from ml.api.results.predictions import PREDICTION_PANEL, META_PANEL
from ml.api.results.predictions import MODEL_PROPERTY, FOLD_PROPERTY
from ml.api.results.predictions import VALUE_PRED_PREFIX, CLASS_PRED_PREFIX, CLASS_PROB_PREFIX, \
    VALUE_TRUE_PREFIX, CLASS_TRUE_PREFIX, TASK_PROPERTY
from ml.api.results import properties
from ml.api.constants import MODE_CLASSIFIER, MODE_REGRESSOR
import logging
logger = logging.getLogger(__name__)

METRIC_PROPERTY = 'Metric'


def extract(train_res, d_pred, score_fn, by_fold=True):
    """
    Compute performance metrics based on training results and prediction data

    :param train_res: Training results
    :param d_pred: DataFrame from `predictions.extract`
    :param score_fn: Function with signature fn(y_true, y_class[, y_proba]) to return dict with keys equal
        to performance metric name and values equal to the numeric value of that metric (note that this
        function must always return a value for each metric)
    :param by_fold: Flag indicating whether or not scores should be computed for each fold or across
        all out-of-sample predictions a single time
    :return: DataFrame containing performance metrics for each task and model
    """

    d = d_pred[[PREDICTION_PANEL, META_PANEL]].copy()

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

    # If not scoring by fold, pretend all the folds were the same for the sake of scoring
    if not by_fold:
        d[(META_PANEL, FOLD_PROPERTY)] = 'All'
    d = d.groupby([(META_PANEL, MODEL_PROPERTY), (META_PANEL, FOLD_PROPERTY)]).apply(get_scores)

    # Cleanup margin names
    d.index.names = [n[1] for n in d.index.names]
    d.columns.names = [METRIC_PROPERTY, TASK_PROPERTY]

    return d


def extract_lift(train_res, d_pred, task=None, class_value=1, n_group=10):
    """
    Extract lift\gains metrics

    :param train_res: Training results
    :param d_pred: DataFrame from `predictions.extract`
    :param task: Name of classification task to extract metrics for (will default to first task if not specified)
    :param class_value: Value of class for task to assume as the "positive" case; Note that if more than 2 classes
        are present, this function will still proceed in a one vs rest comparison
    :param n_group: Number of groups to create for lift/gain calculations
    :return:
    """

    assert train_res.mode == MODE_CLASSIFIER, \
        'Lift/Gain extraction only supported for {} tasks (mode given = "{}")'.format(MODE_CLASSIFIER, train_res.mode)
    assert not pd.isnull(class_value), 'Class value cannot be null'

    tasks = properties.get_prediction_tasks(train_res)
    if len(tasks) > 1 and task is None:
        task = tasks[0]
        logger.warn(
            'Multiple classification tasks are present ["{}"] and the first will be used for lift/gain extraction; '
            'Override this behavior by setting the "task" argument -- Task selected = "{}"'\
            .format('", "'.join(tasks), task)
        )
    else:
        task = tasks[0]

    # Determine predicted probability and actual label columns
    # e.g. 'Class:Probability:Task1:1' and 'Class:Actual:Task1'
    c_pred = '{}:{}:{}'.format(CLASS_PROB_PREFIX, task, class_value)
    c_true = '{}:{}'.format(CLASS_TRUE_PREFIX, task)

    def get_lift(g):
        n = len(g)

        # Sort predictions from highest to lowest probability
        assert np.all(g[c_pred].between(0, 1))
        assert np.all(g[c_pred].notnull())
        g = g.sort_values(c_pred, ascending=False)

        # Determine baseline rate of positive outcomes
        v_rate = g[c_true].mean()

        # Determine the number of records in each group
        group_size = np.floor(n / float(n_group))

        # Create list of indexes representing which group each record falls within, and cap
        # this index at the maximum possible (this will push the remainder of the records
        # into the last group if n not evenly divisible by group_size)
        q = np.clip(np.floor(np.arange(n) / group_size), 0, n_group - 1).astype(np.int64) + 1

        # Group by the group index and compute statistics about the true outcome
        g = g.groupby(q)[c_true].agg(['mean', 'sum', 'count'])
        g.columns = ['Sensitivity', 'ModelCount', 'N']
        g.index.name = 'PredictionGroup'

        # Finally, add lift/gain metrics
        g['NCumulative'] = g['N'].cumsum()
        g['BaselineRate'] = v_rate
        g['BaselineCount'] = g['N'] * v_rate
        g['LiftModel'] = g['Sensitivity'] / v_rate
        g['LiftBaseline'] = 1.
        g['GainModel'] = g['ModelCount'].cumsum()
        g['GainBaseline'] = g['BaselineCount'].cumsum()
        return g

    # Group predictions by model and calculate lift/gain for each;
    # Note that this ignores fold settings
    return d_pred[PREDICTION_PANEL].groupby(d_pred[META_PANEL][MODEL_PROPERTY])[[c_pred, c_true]].apply(get_lift)


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


def visualize_plotly(d_score, kind='box', aggfunc=None, auto_plot=True, layout_kwargs={}, separate_by='Task'):
    """
    Generate Plot.ly boxplot or scatterplot of model performance scores

    :param d_score: DataFrame from `performance.melt`
    :param kind: One of 'scatter' or 'box'
    :param aggfunc: Function to use to aggregate (e.g. 'mean', 'median') feature importance values
        across folds; if left as None, all values will be plotted rather than being aggregated first
    :param auto_plot: Whether or not to plot figures immediately or just return them
    :param layout_kwargs: Optional extra parameters for Plotly Layout objects
    :param separate_by: Field for which a separate plot will be created (one of None, 'Task' or 'Metric')
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

    if separate_by == 'Task':
        c_outer = 'Task'
        c_inner = 'Metric'
    elif separate_by == 'Metric':
        c_outer = 'Metric'
        c_inner = 'Task'
    elif separate_by is None:
        c_outer, c_inner = None, None
    else:
        raise ValueError('Plot separation field "{}" is not valid'.format(separate_by))

    if separate_by is None:
        traces = []

        # Determine the sort order for the plot based on the mean importance value across
        # models for each feature
        o = d_score.groupby('Model')['Value'].mean().sort_values().index.values

        for k, g in d_score.groupby('Metric'):

            # Apply sort order
            g = g.set_index('Model').loc[o].reset_index()

            trace = go.Box(x=g['Model'], y=g['Value'], boxmean='sd', boxpoints='all', name=k)
            traces.append(trace)

        layout_args = dict(layout_kwargs)
        layout_args['boxmode'] = 'group'
        layout_args['hovermode'] = 'closest'
        if 'title' not in layout_args:
            layout_args['title'] = 'Performance Scores by Model'

        fig = go.Figure(data=traces, layout=go.Layout(**layout_args))
        figs['All'] = fig
    else:
        # Loop through outer separating field, and create a new figure for each one
        for k_outer, g_outer in d_score.groupby(c_outer):
            traces = []

            # Determine the sort order for the plot based on the mean importance value across
            # models for each feature
            o = g_outer.groupby('Model')['Value'].mean().sort_values().index.values

            # Loop through inner field and generate a trace for each
            for k_inner, g_inner in g_outer.groupby(c_inner):

                # Apply sort order
                g = g_inner.set_index('Model').loc[o].reset_index()

                # Generate specified type of trace
                if kind == 'box':
                    traces.append(go.Box(
                        x=g['Model'], y=g['Value'],
                        name=k_inner
                    ))
                elif kind == 'scatter':
                    traces.append(go.Scatter(
                        x=g['Model'], y=g['Value'], name=k_inner, mode='markers',
                        text=g['Fold'].apply(lambda v: 'Fold {}'.format(v))
                    ))
                else:
                    raise ValueError('Plot type (i.e. "kind" argument) "{}" is not valid'.format(kind))

            # Add figure to rolling result
            layout_args = dict(layout_kwargs)
            if 'title' not in layout_args:
                layout_args['title'] = k_outer
            fig = go.Figure(data=traces, layout=go.Layout(**layout_args))
            figs[k_outer] = fig

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

    if TASK_PROPERTY not in d_score:
        raise ValueError('Required field "{}" not found .. did you call performance.melt first?'.format(TASK_PROPERTY))

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


def _get_single_value(g, c):
    v = g[c].unique()
    assert len(v) == 1, 'Field "{}" contains multiple unique values (values = {})'.format(c, v)
    return v[0]


def visualize_lift(d_lift, title='Lift Chart', use_counts=True, auto_plot=True):
    """
    Generate lift chart

    :param d_lift: DataFrame from `performance.extract_lift`
    :param title: Title for graph
    :param use_counts: Flag indicating whether or not x axis should be display as raw counts or percentages
    :param auto_plot: Whether or not to plot figure immediately or just return it
    """
    import plotly.graph_objs as go

    d = d_lift.reset_index().copy()
    d_base = d.groupby('PredictionGroup').apply(_get_single_value, c='LiftBaseline')
    x = d.groupby('PredictionGroup').apply(_get_single_value, c='NCumulative')
    d = d.pivot(index='PredictionGroup', columns='Model', values='LiftModel')

    traces = []
    for c in d:
        traces.append(go.Scatter(
            x=d.index,
            y=d[c],
            name=c
        ))
    traces.append(go.Scatter(
        x=d_base.index,
        y=d_base,
        name='Baseline'
    ))
    if use_counts:
        xaxis = go.XAxis(
            ticktext=[str(v) for v in x.values],
            tickvals=x.index.values,
            title='Number of Predictions'
        )
    else:
        xaxis = go.XAxis(
            ticktext=['{0:.1f}%'.format(100 * v) for v in x.values / x.max()],
            tickvals=x.index.values,
            title='Percent Predictions'
        )

    layout = go.Layout(
        xaxis=xaxis,
        yaxis=go.YAxis(tickformat='%', title='Lift'),
        title=title
    )
    fig = go.Figure(data=traces, layout=layout)
    if auto_plot:
        from py_utils import plotly_utils
        return plotly_utils.iplot(fig)
    else:
        return fig


def visualize_gain(d_lift, title='Gain Chart', use_counts=True, auto_plot=True):
    """
    Generate gain chart

    :param d_lift: DataFrame from `performance.extract_lift`
    :param title: Title for graph
    :param use_counts: Flag indicating whether or not x and y axes should be display as raw counts or percentages
    :param auto_plot: Whether or not to plot figure immediately or just return it
    """
    import plotly.graph_objs as go

    d = d_lift.reset_index().copy()

    d_ct = d.groupby('Model')['ModelCount'].sum()
    assert d_ct.nunique() == 1
    n_pos = 1 if use_counts else d_ct.iloc[0]

    d_base = d.groupby('PredictionGroup').apply(_get_single_value, c='GainBaseline')
    x = d.groupby('PredictionGroup').apply(_get_single_value, c='NCumulative')
    d = d.pivot(index='PredictionGroup', columns='Model', values='GainModel')

    traces = []
    for c in d:
        traces.append(go.Scatter(
            x=d.index,
            y=d[c] / n_pos,
            name=c
        ))
    traces.append(go.Scatter(
        x=d_base.index,
        y=d_base / n_pos,
        name='Baseline'
    ))

    if use_counts:
        xaxis = go.XAxis(
            ticktext=[str(v) for v in x.values],
            tickvals=x.index.values,
            title='Number of Predictions'
        )
        yaxis = go.YAxis(
            tickformat=',',
            title='Gain'
        )
    else:
        xaxis = go.XAxis(
            ticktext=['{0:.1f}%'.format(100 * v) for v in x.values / x.max()],
            tickvals=x.index.values,
            title='Percent Predictions'
        )
        yaxis = go.YAxis(
            tickformat='%',
            title='Gain'
        )

    layout = go.Layout(xaxis=xaxis, yaxis=yaxis, title=title)
    fig = go.Figure(data=traces, layout=layout)
    if auto_plot:
        from py_utils import plotly_utils
        return plotly_utils.iplot(fig)
    else:
        return fig
