__author__ = 'eczech'

import pandas as pd
import numpy as np
import math

from pbn import operations as ops
from pbn import conversions as convert
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.grid_search import ParameterGrid, Parallel, delayed
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _parse_kwargs(kwargs, prefix):
    keys = [k for k in kwargs if k.startswith(prefix)]
    new_kwargs = {k.replace(prefix, '', 1): kwargs[k] for k in keys}
    old_kwargs = {k: kwargs[k] for k in kwargs if k not in keys}
    return new_kwargs, old_kwargs


def get_space_color_clusters(img_df, alpha, n_components):
    img_df = img_df.copy()
    img_df[['l', 'a', 'b']] = img_df[['l', 'a', 'b']] * alpha
    mm = GMM(n_components=n_components)

    img_pred = mm.fit(img_df).predict(img_df)
    if not mm.converged_:
        LOGGER.warning(
            'Space-color mixture model did not converge for parameters alpha = {}, n_components = {}'\
            .format(alpha, n_components))

    img_all = pd.concat([img_df, pd.Series(img_pred)], axis=1)
    img_all = img_all.rename(columns={0: 'c'})
    img_all.index.name = 'order'

    color_clusters = img_all.groupby('c')[['l', 'a', 'b']].mean()

    n_before = len(img_all)
    img_merged = img_all.reset_index()\
        .merge(color_clusters.reset_index(), on='c', suffixes=['', '_m'])\
        .sort('order').set_index('order')
    assert n_before == len(img_merged),\
        'Some rows were somehow lost during join: size before = {}, size after = {}'.format(n_before, len(img_merged))
    assert np.all(img_merged.apply(np.isfinite).apply(np.all)), 'Merged data frame has NA values somehow'

    return {
        'img_df': img_merged,
        'alpha': alpha,
        'n_components': n_components,
        'model': mm
    }


def _run_cluster_grid(args):
    i, img_df, grid = args
    LOGGER.info('Running pbn clustering for parameters {}'.format(grid))
    return get_space_color_clusters(img_df, grid['alpha'], grid['n_components'])


def run_cluster_grid(img_df, param_grid, **kwargs):
    par_kwargs, kwargs = _parse_kwargs(kwargs, 'par_')

    LOGGER.info('Running pbn clustering for param grid {}'.format(param_grid))
    param_grid = ParameterGrid(param_grid)
    args_list = [(i, img_df, g) for i, g in enumerate(param_grid)]

    res = Parallel(**par_kwargs)(delayed(_run_cluster_grid)(args) for args in args_list)

    LOGGER.info('Grid clustering run complete')
    return res


def get_color_clusters(img_df, color_cols, alpha, n_colors):
    X = img_df[color_cols] / alpha
    mm = GMM(n_components=n_colors)
    y = pd.Series(mm.fit(X).predict(X))

    if not mm.converged_:
        LOGGER.warning(
            'Color mixture model did not converge for parameters n_colors = {}'
            .format(n_colors))
    mean_map = dict([(i, v * alpha) for i, v in enumerate(mm.means_)])
    img_df[['{}_c'.format(c) for c in color_cols]] = y.map(mean_map).apply(pd.Series)
    img_df['cc'] = y
    return {'img_df': img_df, 'n_colors': n_colors, 'model': mm}


def get_cluster_highlights(img_df, img_shape, cluster_col):
    """ Returns random image colors highlighting cluster borders
        Args:
            img_df: Data frame with at least 'x' and 'y' positions of pixels
            img_shape: 2D resulting array shape (should be the same as original image)
            coluster_col: Column in img_df to use as clusters (usually one of 'c' or 'cc')
    """
    colors = {}
    res = np.empty(img_shape)
    print(res.shape)
    for cluster in img_df[cluster_col].unique():
        colors[cluster] = [np.random.rand(), np.random.rand(), np.random.rand()]
    for i, r in img_df[['x', 'y', cluster_col]].iterrows():
        res[r['x'], r['y']] = colors[r[cluster_col]]
    return res


def get_cluster_properties(img_df, img_shape, **kwargs):
    """
    threshold=.8, shrinkage=1, n_iterations=1
    """

    # Verify that each color cluster really only has one color associated
    n_counts = img_df.groupby('cc')\
        .apply(lambda x: len(x[['l_m_c', 'a_m_c', 'b_m_c']].drop_duplicates()))\
        .value_counts()
    assert len(n_counts) == 1, \
        'Found cluster labels that correspond to more than one cluster (counts = {})'.format(n_counts)

    collapsed_color_clusters = img_df.groupby('cc')\
        .apply(lambda x: x[['l_m_c', 'a_m_c', 'b_m_c']].iloc[0])

    color_matrix = ops.reravel(img_df[['cc']], *img_shape[:2])
    color_matrix = np.int64(np.reshape(color_matrix, img_shape[0:2]))

    spatial_clusters = ops.cluster_by_proximity(color_matrix, **kwargs)

    cluster_props = ops.get_cluster_props(spatial_clusters, collapsed_color_clusters, color_matrix)

    return {'clusters': spatial_clusters, 'properties': cluster_props}


def invert_img_df(img_df, alpha, shape):
    img_color = ops.reravel(img_df, *shape)
    img_color = img_color / alpha
    return convert.lab_to_rgb(img_color)


def show_cluster_grid_result(res, img_shape, fig_size=(12,12)):
    img_centers = res['img_df'][[c for c in res['img_df'].columns if c.endswith('_m')]]
    img_inverted = invert_img_df(img_centers, res['alpha'], img_shape[0:2])
    plt.figure(figsize=fig_size)
    plt.title('alpha = {},\nn_components = {}'.format(res['alpha'], res['n_components']))
    plt.imshow(img_inverted)


def show_cluster_grid_results(res, img_shape, n_cols=3, fig_size=(12, 12)):
    n_rows = int(len(res) / n_cols)
    plt.figure(figsize=fig_size)
    for i, c_res in enumerate(res):
        img_centers = c_res['img_df'][[c for c in c_res['img_df'].columns if c.endswith('_m')]]
        img_inverted = invert_img_df(img_centers, c_res['alpha'], img_shape[0:2])
        plt.subplot(n_rows, n_cols, i+1)
        plt.title('i={}, alpha = {},\nn_components = {}'.format(i, c_res['alpha'], c_res['n_components']))
        plt.imshow(img_inverted)


def render(cluster_properties, img_rgb, alpha, bkg, edg, lbl, scale_factor=1):
    pbn_unsolved, color_index = ops.render_pbn(cluster_properties, img_rgb, alpha,
        bkg=bkg, edg=edg, lbl=lbl, solution=False, scale_factor=scale_factor)
    pbn_solved, _ = ops.render_pbn(cluster_properties, img_rgb, alpha,
        bkg=bkg, edg=edg, lbl=lbl, solution=True, scale_factor=scale_factor)
    return {
        'pbn_unsolved': convert.lab_to_rgb(pbn_unsolved / alpha),
        'pbn_solved': convert.lab_to_rgb(pbn_solved / alpha),
        'color_index': color_index
    }


def _sort_rgb_index(rgb_index):
    rgb_sorted = rgb_index.copy()
    pca = PCA(n_components=3)
    rgb_sorted['index'] = pca.fit_transform(rgb_sorted[['r', 'g', 'b']])[:,0]
    return rgb_sorted.sort('index')


def export_color_swatch(file_path, color_index, alpha,
                        colors_per_col=10, swatch_height=200, swatch_width=200,
                        font_size=50, fig_size=(8, 11)):
    rgb_df = []
    for c, l in color_index.items():
        c = convert._lab_to_rgb(np.array(c) / alpha)
        rgb_df.append((l, c[0], c[1], c[2]))
    rgb_df = pd.DataFrame(rgb_df, columns=['letter', 'r', 'g', 'b'])

    rgb_df = _sort_rgb_index(rgb_df).set_index('letter').sort('index')

    rgb_sorted = []
    for k, r in rgb_df.iterrows():
        v = r.to_dict()
        v['key'] = k
        rgb_sorted.append(v)

    n_colors = len(rgb_sorted)

    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = fig.add_subplot(111)

    x_margin = 10
    y_margin = 10

    for i, color in enumerate(rgb_sorted):
        c = (color['r'], color['g'], color['b'])

        x = int(i / colors_per_col) * swatch_width + x_margin
        y = (i % colors_per_col) * swatch_height + y_margin
        rect = patches.FancyBboxPatch((x, y), int(swatch_width/2),
                                                 swatch_height, fc=c,
                                                 boxstyle="round,pad=.5,rounding_size=1")
        ax.add_patch(rect)
        ax.text(x + swatch_width * .55, y + swatch_height * .7, color['key'], fontsize=font_size)

    plt.xlim([0, math.ceil(n_colors / colors_per_col) * swatch_width + 2 * x_margin])
    plt.ylim([0, colors_per_col * swatch_height + 2 * y_margin])
    plt.gca().invert_yaxis()
    plt.axis('off')

    fig.savefig(file_path)
    return file_path
