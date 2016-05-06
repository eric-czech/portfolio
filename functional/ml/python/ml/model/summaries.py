__author__ = 'eczech'

from . import models
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.ensemble import partial_dependence as ptl_dep
from sklearn.metrics import roc_curve

DEFAULT_REGRESSOR_SCORE = lambda clf, y_true, y_pred: mean_squared_error(y_true, y_pred)


def summarize_grid_parameters(res):
    grid_params = models.summarize_grid_parameters(res)
    if grid_params is None:
        return None
    return grid_params.groupby(['model_name', 'variable'])['value'].apply(lambda x: x.value_counts())


def plot_model_scores(res, score_func=DEFAULT_REGRESSOR_SCORE, figsize=(12, 6), use_proba=None):
    scores = models.summarize_scores(res, score_func, use_proba=use_proba)
    ax = scores .boxplot('score', 'model_name', figsize=figsize)
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


def plot_feature_importance(res, figsize=(12, 4), limit=25):
    feat_imp = models.summarize_importances(res)
    if feat_imp is None:
        return None

    feat_means = feat_imp.drop('fold_id', axis=1)
    feat_error = feat_means.groupby('model_name').aggregate(np.std).T
    feat_means = feat_means.groupby('model_name').aggregate(np.mean).T
    feat_means = feat_means.abs()
    figs = []
    for model in feat_means:
        figs.append(plt.figure())
        yerr = feat_error[model]
        ax = feat_means[model].sort_values(ascending=False)\
            .head(limit).plot(kind='bar', figsize=figsize, yerr=yerr)
        ax.set_title(model)
    return figs


def plot_predictions(res, figsize=(12, 4)):
    preds = models.summarize_predictions(res)
    figs = []
    for model, grp in preds.groupby('model_name'):
        figs.append(plt.figure())
        ax = grp.plot(kind='scatter', x='y_pred', y='y_true', figsize=figsize)
        ax.set_title(model)
    return figs


def plot_partial_dependence(est, X, features, **kwargs):
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


def plot_weighted_feature_importances(res, score_func, feat_agg=np.median, score_agg=np.median,
                                      figsize=(18, 4), limit=25):
    return models.summarize_weighted_importances(res, score_func, feat_agg=feat_agg, score_agg=score_agg)\
        .sort_values(ascending=False).head(limit).plot(kind='bar', figsize=figsize)