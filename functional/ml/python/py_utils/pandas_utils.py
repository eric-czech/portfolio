
import pandas as pd
from joblib import Parallel, delayed


def parallel_group_by(df_grp, func, n_jobs=-1):
    """
    Applies per-group function to each group in the given grouped data frame

    :param df_grp: Grouped DataFrame
    :param func: Function to apply to each group
    :return: DataFrame containing results from each group
    """
    parts = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in df_grp)
    return pd.concat(parts)

