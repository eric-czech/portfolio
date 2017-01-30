
import pandas as pd


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
