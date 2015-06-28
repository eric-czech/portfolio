__author__ = 'eczech'

import pandas as pd
import numpy as np
from scipy import ndimage
from pbn import operations as ops
from pbn import conversions as convert

def collapse_colors(flt_cluster, n_colors=10):
    res = flt_cluster.copy()
    from sklearn.mixture import GMM
    m_cols = [c for c in flt_cluster.columns if c.endswith('_m')]
    d = flt_cluster[m_cols]
    mm = GMM(n_components=n_colors)
    mm.fit(d)
    y = pd.Series(mm.predict(d))
    mean_map = dict([ (i, v) for i, v in enumerate(mm.means_) ])

    ccols = ['{}_c'.format(c) for c in m_cols]
    res[ccols] = y.map(mean_map).apply(pd.Series)
    return res


def denoise_raw_img(img_d, n_iter=1):
    for i in range(n_iter):
        img_d = ndimage.median_filter(img_d, 3)
    return img_d


def denoise_flat_img(img_flat, img_raw_shape, color_cols, n_iter=1, suffix='dn', alpha=1):
    """ Converts lab image represented as n * 3 data frame to denoised version of the same
    """
    img_raw = ops.reravel(img_flat[color_cols], *img_raw_shape[:2])

    # Denoise image as rbg (it doesn't work well with lab)
    img_raw = denoise_raw_img(convert.lab_to_rgb(img_raw / alpha), n_iter=n_iter)

    # Put image back in lab form with alpha adjustment
    img_raw = convert.rgb_to_lab(img_raw) * alpha

    denoise_cols = ['{}_{}'.format(c, suffix) for c in color_cols]
    img_flat[denoise_cols] = ops.unravel(img_raw, denoise_cols)[denoise_cols]
    return img_flat