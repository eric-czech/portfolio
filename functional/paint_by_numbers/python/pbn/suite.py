__author__ = 'eczech'

from pbn import operations as ops
import pandas as pd
import numpy as np

def get_pbn(img, max_colors=25, alpha=1):
    colors = ops.unravel_old(img)
    colors.head()

    preds = ops.cluster_color_space(colors, max_clusters=max_colors, alpha=alpha)

    pred_colors = pd.concat([preds, colors], axis=1)
    pred_colors = pred_colors.rename(columns={0: 'CLUSTER'})
    color_clusters = pred_colors.groupby('CLUSTER').mean().applymap(lambda x: int(np.round(x)))

    pred_colors = pred_colors.merge(color_clusters, left_on='CLUSTER', right_index=True, suffixes=['', '_MEAN'])

    # Reconstructed after color clustering
    img_color_cluster = ops.reravel(pred_colors[[c for c in pred_colors.columns if '_MEAN' in c]], *img.shape[0:2])
    img_color_cluster /= float(255)

    img_cl = ops.reravel(pred_colors[['CLUSTER']], img.shape[0], img.shape[1])
    img_cl = np.reshape(img_cl, img.shape[0:2])

    img_cl = ops.blur(img_cl)
    img_cl = ops.blur(img_cl)
    spatial_clusters = ops.cluster_euclidean_space(np.int64(img_cl))

    # Recon after aptial clustering
    img_spatial_cluster = ops.image_from_clusters(spatial_clusters, color_clusters, img, use_random_color=True)

    clusters = ops.get_cluster_props(spatial_clusters, color_clusters)

    img_pbn = ops.render_pbn(clusters, img, solution=False)
    img_sol = ops.render_pbn(clusters, img, solution=True)

    return clusters, img_color_cluster, img_spatial_cluster, img_pbn, img_sol


