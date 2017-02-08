from ml.api.utils import importance_utils
import pandas as pd


def extract(train_res, feat_imp_fn=None):
    """
    Extract feature importances from training results
    :param train_res: Training results
    :param feat_imp_fn: Optional dict mapping estimator names to feature importance calculation functions;
        these functions should have the signature fn(clf, X_names, Y_names, clf_name) and should return
        a data frame with the same structure as that returned by this function (except for the model/estimator name)
    :return: DataFrame with Model (estimator name), Feature, Task, Class (only for classifiers),
        and "Value" associated with importance
    """
    d = []
    for resample_res in train_res.resample_results:
        for model_res in resample_res.model_results:
            clf = model_res.clf
            clf_name = model_res.clf_name
            X_names = model_res.X_names
            Y_names = model_res.Y_names
            feat_imp = importance_utils.get_estimator_feature_importance(clf, X_names, Y_names, clf_name=clf_name)
            if feat_imp is not None:
                feat_imp['Model'] = clf_name
                feat_imp['Fold'] = model_res.fold
                d.append(feat_imp)
    if len(d) > 0:
        return pd.concat(d)
    return None


def normalize(d_imp, how='minmax'):
    """
    Normalize feature importances across features and folds (i.e. normalize within models and tasks)

    :param d_imp: DataFrame from `importances.extract`
    :param how: One of 'minmax' or 'zscore' where:
        minmax: Scale importances from 0 to 100
        zscore: Standardize importances using mean and std
    :return:
    """
    def normalize_group(g):
        v = g['Value'].abs()
        if how == 'minmax':
            v = 100 * (v - v.min()) / (v.max() - v.min())
        elif how == 'scale':
            v = (v - v.mean()) / v.std()
        else:
            raise ValueError('Normalization type (ie "how" argument) "{}" is not valid'.format(how))
        g['Value'] = v
        return g
    return d_imp.groupby(['Task', 'Model'], group_keys=False).apply(normalize_group)


def visualize_plotly(d_imp, kind='box', aggfunc=None, auto_plot=True, layout_kwargs={}):
    """
    Generate Plot.ly boxplot or scatterplot of feature importances

    :param d_imp: DataFrame from `importances.extract`
    :param kind: One of 'scatter' or 'box'
    :param aggfunc: Function to use to aggregate (e.g. 'mean', 'median') feature importance values
        across folds; if left as None, all values will be plotted rather than being aggregated first
    :param auto_plot: Whether or not to plot figures immediately or just return them
    :param layout_kwargs: Optional extra parameters for Plotly Layout objects
    :return: Plotly Figure instace dict keyed by task name
    """
    import plotly.graph_objs as go

    figs = {}

    # Apply aggregation across folds, if configured to do so
    if aggfunc is not None:
        d_imp = (
            d_imp.groupby(['Task', 'Model', 'Feature'])
            .agg({'Value': aggfunc})
            .reset_index()
            .assign(Fold=0)
        )

    # Loop through tasks, and create a new figure for each one
    for task, g_task in d_imp.groupby('Task'):
        traces = []

        # Determine the sort order for the plot based on the mean importance value across
        # models for each feature
        o = g_task.groupby('Feature')['Value'].mean().sort_values().index.values

        # Loop through models and generating a trace for each
        for k, g in g_task.groupby('Model'):

            # Apply sort order
            g = g.set_index('Feature').loc[o].reset_index()

            # Generate specified type of trace
            if kind == 'box':
                traces.append(go.Box(
                    x=g['Feature'], y=g['Value'],
                    name=k
                ))
            elif kind == 'scatter':
                traces.append(go.Scatter(
                    x=g['Feature'], y=g['Value'], name=k, mode='markers',
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


def visualize(d_imp, backend='plotly', **kwargs):
    """
    Generates visualization of feature importances

    See individual backend plotting functions below for more options/details:
    1. `visualize_plotly`

    :param d_imp: DataFrame from `importances.extract`
    :param backend: Backend to use for plotting; one of ['plotly']
    :return: plot objects [varies based on backend]
    """
    assert backend in ['plotly'], 'Backend can currently only be "plotly"'

    if backend == 'plotly':
        return visualize_plotly(d_imp, **kwargs)

    return None
