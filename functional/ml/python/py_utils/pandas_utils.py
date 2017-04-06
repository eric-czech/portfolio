
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
