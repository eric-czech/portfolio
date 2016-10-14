
import numpy as np
import pandas as pd
from plotly import tools
import plotly.graph_objs as go
from plotly import offline


def _repeat_mean(X, r):
    """ Returns mean of given features repeated some specified number of times
    :param X: Feature DataFrame
    :param r: Number of times to repeat mean vector as data frame
    :return: DataFrame
    """
    m = len(X.columns)
    x = X.mean().repeat(r).reshape((m, r)).T
    return pd.DataFrame(x, columns=X.columns)


def get_univariate_dependence_1d(clf, X, features, pred_fun, feature_grids=None,
                              print_progress=True, grid_window=[0, 1], grid_size=50,
                             discrete_thresh_ct=10, fill_mode='mean'):

    assert fill_mode in ['zeros', 'mean'], '"fill_mode" must be either "zeros" or "mean"'

    res = {}
    for i, feature in enumerate(features):
        if print_progress:
            print('Processing feature {} of {} ...'.format(i+1, len(features)))

        if feature_grids is not None and feature in feature_grids:
            grid = feature_grids[feature]
        else:
            x = X[feature]
            if len(x.unique()) <= discrete_thresh_ct:
                grid = np.sort(x.unique())
            else:
                if isinstance(grid_window, dict):
                    assert feature in grid_window
                    gw = grid_window[feature]
                else:
                    gw = grid_window
                xmin = np.percentile(x, gw[0] * 100)
                xmax = np.percentile(x, gw[1] * 100)
                grid = np.linspace(xmin, xmax, grid_size)

        if fill_mode == 'mean':
            Xz = _repeat_mean(X, len(grid))
        elif fill_mode == 'zeros':
            Xz = pd.DataFrame(np.zeros((len(grid), len(X.columns))), columns=X.columns)
        Xz[feature] = grid
        res[feature] = pd.Series(pred_fun(clf, Xz), index=grid)
    return res


def plot_univariate_dependence(udp, n_cols=3, title='Univariate Dependence', smooth_window=None,
                            filename=None, fig_dimension=None, transform=None,
                            sharex=False, sharey=False):
    n_feats = len(udp)
    if n_feats < n_cols:
        n_cols = n_feats
    n_rows = int(np.ceil(n_feats/n_cols))
    if sharey and n_feats % n_cols != 0:
        raise NotImplementedError(
            'When using shared Y-Axis values, make sure that number of columns ({}) evenly '\
            'divides number of variables ({}) (i.e. make sure there are no empty subplots) '\
            'or Plot.ly will not show some plots correctly.'.format(n_cols, n_feats)
        )

    fig = tools.make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=list(udp.keys()), print_grid=False,
        shared_xaxes=sharex, shared_yaxes=sharey
    )
    for i, feature in enumerate(udp):
        s_udp = udp[feature]
        x = list(s_udp.index.values)
        y = s_udp.values
        if smooth_window is not None:
            y = pd.rolling_mean(y, min_periods=1, window=smooth_window)
        y = list(y)
        line = dict(color='black')
        trace = go.Scatter(x=x, y=y, name=feature, line=line)
        fig.append_trace(trace, int(np.floor(i / n_cols) + 1), (i % n_cols) + 1)

        if transform is not None:
            y_trans = transform['function'](x, y)
            trace = go.Scatter(x=x, y=y_trans, name=transform['title'], line=dict(color=transform['color']))
            fig.append_trace(trace, int(np.floor(i / n_cols) + 1), (i % n_cols) + 1)

    fig['layout'].update(title=title, showlegend=False)
    if fig_dimension is not None:
        fig['layout'].update(width=fig_dimension[0], height=fig_dimension[1])
    if filename is None:
        return offline.iplot(fig, show_link=False)
    else:
        return offline.plot(fig, show_link=False, auto_open=True, filename=filename)


def get_partial_dependence_1d(clf, X, features, pred_fun, grid_size=100, grid_window=[0, 1],
                              sample_rate=1, random_state=None, discrete_thresh_ct=10, print_progress=True):
    if isinstance(features, str):
        features = [features]

    if sample_rate < 1:
        if random_state:
            np.random.seed(random_state)
        idx = np.random.choice(np.arange(0, len(X)), size=int(sample_rate*len(X)), replace=False)
        X = X.iloc[idx, :]

    res = {}
    for i, feature in enumerate(features):
        if print_progress:
            print('Processing feature {} of {} ...'.format(i+1, len(features)))
        x = X[feature]
        if len(x.unique()) <= discrete_thresh_ct:
            grid = np.sort(x.unique())
        else:
            if isinstance(grid_window, dict):
                assert feature in grid_window
                gw = grid_window[feature]
            else:
                gw = grid_window
            xmin = np.percentile(x, gw[0] * 100)
            xmax = np.percentile(x, gw[1] * 100)
            grid = np.linspace(xmin, xmax, grid_size)

        X_pd = X.copy()
        pdp = {}
        for p in grid:
            X_pd[feature] = p
            pdp[p] = pred_fun(clf, X_pd)
        res[feature] = pd.DataFrame(pdp)
    return res


def plot_partial_dependence(pdp, n_cols=3, title='Partial Dependence', smooth_window=None,
                            filename=None, fig_dimension=None, transform=None,
                            sharex=False, sharey=False):
    n_feats = len(pdp)
    if n_feats < n_cols:
        n_cols = n_feats
    n_rows = int(np.ceil(n_feats/n_cols))
    if sharey and n_feats % n_cols != 0:
        raise NotImplementedError(
            'When using shared Y-Axis values, make sure that number of columns ({}) evenly '\
            'divides number of variables ({}) (i.e. make sure there are no empty subplots) '\
            'or Plot.ly will not show some plots correctly.'.format(n_cols, n_feats)
        )

    fig = tools.make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=list(pdp.keys()), print_grid=False,
        shared_xaxes=sharex, shared_yaxes=sharey
    )
    for i, feature in enumerate(pdp):
        pdp_mean = pdp[feature].mean()
        x = list(pdp_mean.index.values)
        y = pdp_mean.values
        if smooth_window is not None:
            y = pd.rolling_mean(y, min_periods=1, window=smooth_window)
        y = list(y)
        line = dict(color='black')
        trace = go.Scatter(x=x, y=y, name=feature, line=line)
        fig.append_trace(trace, int(np.floor(i / n_cols) + 1), (i % n_cols) + 1)

        if transform is not None:
            y_trans = transform['function'](x, y)
            trace = go.Scatter(x=x, y=y_trans, name=transform['title'], line=dict(color=transform['color']))
            fig.append_trace(trace, int(np.floor(i / n_cols) + 1), (i % n_cols) + 1)

    fig['layout'].update(title=title, showlegend=False)
    if fig_dimension is not None:
        fig['layout'].update(width=fig_dimension[0], height=fig_dimension[1])
    if filename is None:
        return offline.iplot(fig, show_link=False)
    else:
        return offline.plot(fig, show_link=False, auto_open=True, filename=filename)


