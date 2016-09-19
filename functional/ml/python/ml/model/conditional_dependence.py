from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def plot_ice(
    X, pdp, random_state=None, n_sample=1000,
    n_cluster=6, cluster_alg=KMeans,
    n_interaction=1, interaction_alg=RandomForestClassifier,
    primary_cmap=plt.cm.spectral, interaction_cmap=plt.cm.RdYlGn,
    figsize=None, alphas=[.3, .75, .5]):

    pdp_vars = list(pdp.keys())
    assert len(pdp_vars) > 0, 'No PDP given to plot'
    assert np.all(np.sort(pdp[pdp_vars[0]].index.values) == np.arange(len(X)))

    # Restrict number of samples to be no greater than number of records
    n_sample = len(X) if len(X) < n_sample else n_sample

    # Initialize figure
    figsize = figsize if figsize is not None else (20, 7 * len(pdp_vars))
    fig = plt.figure(figsize=figsize)

    gr = 4
    gc = 6 + n_interaction * 6 + 2
    gs = gridspec.GridSpec(gr * len(pdp_vars), gc)

    for i, pdp_var in enumerate(pdp_vars):
        # Index = Sample Index, Column = Variable Value
        d_ice = pdp[pdp_var].sample(n=n_sample, random_state=random_state)

        # Cluster samples
        d_clust = cluster_alg(n_clusters=n_cluster).fit_predict(d_ice)

        v_clust, v_cts = np.unique(d_clust, return_counts=True)
        n_clust = len(v_clust)

        cmap = plt.cm.spectral
        cmap = [cmap(i) for i in np.linspace(0.1, 0.8, n_clust)]
        m_col = dict(zip(list(v_clust), cmap))

        d_col = dict(zip(d_ice.index.values, d_clust))
        d_ice_mean = d_ice.mean()
        d_plt = d_ice.T

        axi = gr*i
        ax = plt.subplot(gs[axi:(axi+3), 0:6])
        for c in d_plt:
            v_clust = d_col[c]
            ax.plot(d_plt.index.values, d_plt[c].values, color=m_col[v_clust], alpha=alphas[0])
        ax.plot(d_ice_mean.index.values, d_ice_mean.values, color='black', alpha=alphas[1], linewidth=7)
        ax.set_title('{} (# Clusters = {})'.format(pdp_var, n_clust))

        X_clust = X.iloc[d_ice.index.values]
        m_clust = interaction_alg().fit(X_clust, d_clust)
        m_clust_imp = pd.Series(m_clust.feature_importances_, index=X.columns).sort_values()

        axi = gr*i + 3
        ax = plt.subplot(gs[axi, 0:6])
        ax.hist(X_clust[pdp_var], bins=30)

        for j in range(n_interaction):
            top_var = m_clust_imp.index.values[-(j+1)]
            v_var = pd.Series(X_clust[top_var].values, index=d_ice.index)
            v_var = (v_var - v_var.quantile(.01)) / (v_var.quantile(.99) - v_var.quantile(.01))
            v_var = v_var.clip(0., 1.)

            axi = gr*i
            axj = 6*(j+1)
            ax = plt.subplot(gs[axi:(axi+3), axj:(axj+6)])
            cmap = plt.cm.RdYlGn
            for c in d_plt:
                color = cmap(v_var.loc[c])
                ax.plot(d_plt.index.values, d_plt[c].values, color=color, alpha=alphas[2])
                #ax.plot(np.log10(d_plt.index.values), d_plt[c].values, color=color, alpha=.5)
            ax.set_title('{} (vs {})'.format(pdp_var, top_var))

            axi = gr*i + 3
            ax = plt.subplot(gs[axi, axj:(axj+6)])
            ax.hist(X_clust[top_var], bins=30)

        axi = gr*i
        ax = plt.subplot(gs[axi:(axi+gr), (gc-2):gc])
        m_clust_imp.index = [c[-10:] for c in m_clust_imp.index]
        m_clust_imp.tail(10).plot(kind='barh', ax=ax)

    plt.tight_layout()
