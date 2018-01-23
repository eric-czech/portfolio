
import pandas as pd
import numpy as np


def parallel_group_by(df_grp, func, n_jobs=-1):
    """
    Applies per-group function to each group in the given grouped data frame

    :param df_grp: Grouped DataFrame
    :param func: Function to apply to each group
    :return: DataFrame containing results from each group
    """
    from sklearn.externals.joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in df_grp)
    return pd.concat(parts)


def one_value(x):
    if len(x) > 1:
        raise ValueError('Found cell with more than one value; First 10 values = {}'.format(x.iloc[:10]))
    return x.iloc[0]


def first_non_null_value(df):
    """
    Return first non-null value in each row (or the first value if all are null)

    :param df: DataFrame to collapse
    :return: Series with first non-null value in each row
    """
    a = df.values
    n_rows, n_cols = a.shape
    col_index = pd.isnull(a).argmin(axis=1)
    flat_index = n_cols * np.arange(n_rows) + col_index
    return pd.Series(a.ravel()[flat_index], index=df.index)


###################
# Style Functions #
###################


# **** Example showing how to map continuous values to colors in general with matplotlib:
# """Color background in a range according to the data."""
# with _mpl(Styler.background_gradient) as (plt, colors):
#     rng = s.max() - s.min()
#     # extend lower / upper bounds, compresses color range
#     norm = colors.Normalize(s.min() - (rng * low),
#                             s.max() + (rng * high))
#     # matplotlib modifies inplace?
#     # https://github.com/matplotlib/matplotlib/issues/5427
#     normed = norm(s.values)
#     c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
#     return ['background-color: %s' % color for color in c]


def _cfn_percentile(v):
    r = v.rank(method='dense')
    return r / r.max()


def colorfn_percentile(npct=5, nanpct=0.):
    """
    Create color function for generating discrete percentiles.

    This is useful for generating color gradients in tables that are easy to identify associated percentiles
    for.  For example, with npct=5, colors will correspond to the [0, 20]%, (20-40], (40-60], (60-80]%, (80-100]%.

    :param npct: Number of percentiles to allow to be mapped to colors
    :param nanpct: Percentile bucket to map nan values to (defaults to 0 meaning that nan
        values will be colored just as the lowest percentile bucket)

    """
    nanpct = max(min(nanpct, 1), 0)
    def fn(v):
        # Convert values to percentiles in [0, 1]
        v = _cfn_percentile(v)

        # Convert nan values to the given percentile
        v = v.where(v.notnull(), nanpct)

        # Reduce all percentile values by a tiny amount to ensure that no values are == 1 and
        # that all values on exact boundaries are brought below those boundaries (i.e. 20% --> 19.9999%)
        v *= (1. - 1e-16)

        # Determine divisor for bucketing
        p = 1 / npct

        return np.floor(v / p) * p
    return fn


def custom_gradient(v, cfn=None, cmap='PuBu'):
    """
    Function to apply gradient background colors to table cells.

    :param v: Series to be stylized
    :param cfn: "Color Function" with signature `fn(v)` that will convert any given Series to numbers in [0, 1] (or nan)
    :param cmap: Name of matplotlib colormap to use
    """
    v = cfn(v)
    if not np.all(((v >= 0) & (v <= 1)) | np.isnan(v)):
        raise ValueError('Color conversion function returned value outside of [0, 1] (or nan)')
    return to_colors(v, cmap)


def to_colors(v, cmap):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(v)]
    return ['background-color: %s' % color for color in c]
