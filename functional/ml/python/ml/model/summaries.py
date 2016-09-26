__author__ = 'eczech'

from . import models
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.ensemble import partial_dependence as ptl_dep
from sklearn.metrics import roc_curve
import plotly.graph_objs as go
from plotly import offline
import seaborn as sns


DEFAULT_REGRESSOR_SCORE = lambda clf, y_true, y_pred: mean_squared_error(y_true, y_pred)


def summarize_grid_parameters(res):
    grid_params = models.summarize_grid_parameters(res)
    if grid_params is None:
        return None
    return grid_params.groupby(['model_name', 'variable'])['value'].apply(lambda x: x.value_counts())


def plot_model_scores(res, score_func=DEFAULT_REGRESSOR_SCORE, figsize=(12, 6), use_proba=None, **kwargs):
    scores = models.summarize_scores(res, score_func, use_proba=use_proba)
    ax = scores.boxplot('score', 'model_name', figsize=figsize, **kwargs)
    return ax

INTERPOLATE_FILL = lambda x: x.ffill().bfill()
INTERPOLATE_LINEAR = lambda x: x.interpolate()


def plot_curve(res, curve_func=roc_curve, interpolation=INTERPOLATE_LINEAR, plot_size=(8, 6)):
    d_roc = models.summarize_curve(res, curve_func=curve_func)

    n_models = len(d_roc['model_name'].unique())
    nrow, ncol = int((1 + n_models)/2), 2

    fig, ax = plt.subplots(nrow, ncol)
    fig.set_size_inches(ncol * plot_size[0], nrow * plot_size[1])

    for i, (model, grp) in enumerate(d_roc.groupby('model_name')):
        ax = plt.subplot(nrow, ncol, i+1)
        d = grp.pivot_table(index='x', columns='fold_id', values='y')
        d['random'] = d.index.values
        ax = interpolation(d).plot(ax=ax)
        ax.set_title(model)


def plot_feature_importance(res, limit=25, xrot=35, rmargin=100, bmargin=160,
                            width=1000, height=400, filename=None, normalize=True,
                            feat_imp_calc=None, asFigure=False):

    if isinstance(res, pd.DataFrame):
        feat_imp = res
    else:
        feat_imp = models.summarize_importances(res, feat_imp_calc=feat_imp_calc)

    if feat_imp is None:
        return None

    # If requested, make sure all feature importance scores
    # per model and fold are normalized to [0, 1]
    if normalize:
        feat_imp = feat_imp\
            .set_index(['model_name', 'fold_id'])\
            .apply(lambda r: r / r.sum(), axis=1)\
            .reset_index()

    feat_means = feat_imp.drop('fold_id', axis=1)
    feat_means = feat_means.groupby('model_name')
    feat_error = feat_means.std().T
    top_feats = feat_means.mean().mean().T.abs().sort_values(ascending=False).head(limit).index.values
    feat_means = feat_means.mean().T.abs()

    traces = []
    for model in feat_means:
        d = feat_means[model][top_feats]
        trace = go.Scatter(
            x=d.index.values,
            y=d.values,
            mode='markers',
            name=model,
            error_y=dict(
                type='data',
                array=feat_error[model][top_feats].values
            )
        )
        traces.append(trace)
    layout = go.Layout(
        width=width,
        height=height,
        xaxis=dict(tickangle=xrot),
        margin=dict(b=bmargin, r=rmargin),
        title='Feature Importances'
    )
    fig = go.Figure(data=traces, layout=layout)
    if asFigure:
        return fig
    if filename is None:
        return offline.iplot(fig, show_link=False)
    else:
        return offline.plot(fig, show_link=False, filename=filename)


def plot_predictions(res, figsize=(12, 4)):
    preds = models.summarize_predictions(res)
    figs = []
    for model, grp in preds.groupby('model_name'):
        figs.append(plt.figure())
        ax = grp.plot(kind='scatter', x='y_pred', y='y_true', figsize=figsize)
        ax.set_title(model)
    return figs


def plot_weighted_feature_importances(res, score_func, feat_agg=np.median, score_agg=np.median,
                                      figsize=(18, 4), limit=25):
    return models.summarize_weighted_importances(res, score_func, feat_agg=feat_agg, score_agg=score_agg)\
        .sort_values(ascending=False).head(limit).plot(kind='bar', figsize=figsize)


def plot_confusion_matrix(d_pred, ignore_fold_id=False, use_pct=False):
    """
    Plot confusion matrix for summarized predictions

    :param d_pred: DataFrame from ml.models.summarize_predictions
    :param ignore_fold_id: If true, CM counts will be aggregated across folds
    :param use_pct: If true, counts will be shown as percentages instead
    :return: Seaborn grid containing confusion matrices
    """
    d_cm = d_pred.copy()

    # Rename prediction value columns if prefixed by fold indicator string
    if 'fold:model_name' in d_cm:
        pred_cols = d_cm.filter(regex='^fold:').columns.tolist()
        d_cm = d_cm[pred_cols]
        d_cm = d_cm.rename(columns=lambda c: c.replace('fold:', ''))

    # If flag to ignore fold id is true (i.e. CM should aggregate across folds)
    # set the fold id to the same value for all predictions
    if ignore_fold_id:
        d_cm['fold_id'] = 0

    # Get frequency of prediction + actual pairs, per model and fold
    d_cm = d_cm.groupby(['model_name', 'fold_id', 'y_pred', 'y_true']).size()
    d_cm.name = 'count'
    d_cm = d_cm.reset_index()

    # Plot CM heatmaps with model name in rows and fold ids in columns
    def facet(data, color):
        data = data.pivot_table(index='y_pred', columns='y_true', values='count')
        if use_pct:
            data = data / data.sum().sum()
            fmt = '.2%'
        else:
            fmt = 'd'
        sns.heatmap(data, cbar=False, annot=True, fmt=fmt)

    g = sns.FacetGrid(d_cm, row='model_name', col='fold_id', margin_titles=True)
    g = g.map_dataframe(facet)
    g = g.set_axis_labels(x_var='Actual', y_var='Predicted')
    return g


def plot_gbrt_partial_dependence(est, X, features, **kwargs):
    """
    Returns partial dependence plot for the given *trained* estimator

    :param est: Estimator that has already been fit
    :param X: Features used in training
    :param features: List of tuples or single feature names (tuples indicate 2d partial dep)
    :param kwargs: Other arguments for sklearn plot_partial_dependence
    :return: fig, axs
    """
    fig, axs = ptl_dep.plot_partial_dependence(est, X, features, feature_names=X.columns.tolist(), **kwargs)
    return fig, axs

