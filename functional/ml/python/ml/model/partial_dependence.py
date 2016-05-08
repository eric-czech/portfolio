
import numpy as np
import pandas as pd
from plotly import tools
import plotly.graph_objs as go
from plotly import offline


def get_partial_dependence_1d(clf, X, features, pred_fun, force_discrete=False, grid_size=100, grid_window=[0, 1],
                              sample_rate=1, random_state=None):
    if isinstance(features, str):
        features = [features]

    if sample_rate < 1:
        if random_state:
            np.random.seed(random_state)
        idx = np.random.choice(np.arange(0, len(X)), size=int(sample_rate*len(X)), replace=False)
        X = X.iloc[idx, :]

    res = {}
    for feature in features:
        x = X[feature]
        if len(x.unique()) <= 10 or force_discrete:
            grid = x.unique()
        else:
            xmin = np.percentile(x, grid_window[0] * 100)
            xmax = np.percentile(x, grid_window[1] * 100)
            grid = np.linspace(xmin, xmax, grid_size)

        X_pd = X.copy()
        pdp = {}
        for p in grid:
            X_pd[feature] = p
            pdp[p] = pred_fun(clf, X_pd)
        res[feature] = pd.DataFrame(pdp)
    return res


def plot_partial_dependence(pdp, n_cols=3, title='Partial Dependence', smooth_window=None):
    n_feats = len(pdp)
    if n_feats < n_cols:
        n_cols = n_feats
    n_rows = int(np.ceil(n_feats/n_cols))
    fig = tools.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(pdp.keys()), print_grid=False)
    for i, feature in enumerate(pdp):
        pdp_mean = pdp[feature].mean()
        x = list(pdp_mean.index.values)
        y = pdp_mean.values
        if smooth_window is not None:
            y = pd.rolling_mean(y, min_periods=1, window=smooth_window)
        y = list(y)
        line = dict(color='black')
        trace = go.Scatter(x=x, y=y, name=feature, fillcolor='blue', line=line)
        fig.append_trace(trace, int(np.floor(i / n_cols) + 1), (i % n_cols) + 1)

    fig['layout'].update(title=title, showlegend=False)
    offline.plot(fig, show_link=False, auto_open=True)
