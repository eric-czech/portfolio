__author__ = 'eczech'

import pandas as pd
import numpy as np


def _to_list(x):
    return [x] if isinstance(x, str) else x


def _get_time_lags(d, lags, date_cols, value_col):
    select_cols = _to_list(date_cols) + [value_col]
    d = d[select_cols].set_index(date_cols)[value_col]
    d = pd.concat([d.shift(i) for i in lags], axis=1)
    d.columns = ['{}.{}'.format(value_col, i) for i in lags]
    return d


def get_time_lags(d, lags, date_cols, value_cols):
    return pd.concat([_get_time_lags(d, lags, date_cols, value_col) for value_col in _to_list(value_cols)], axis=1)


def get_group_time_lags(d, lags, group_cols, date_col, value_cols):
    return d.groupby(group_cols).apply(get_time_lags, lags, date_col, value_cols)


BACKFILL_MISSING = lambda x: x.bfill()
DROP_MISSING = lambda x: x.dropna()


def _get_time_aggregates(d, interval, date_cols, value_cols, agg_func=np.sum, fill_func=None):
    parts = []
    for value_col in _to_list(value_cols):
        p = _get_time_lags(d, np.arange(interval[0], interval[1]+1), date_cols, value_col)

        # If a "fill" function was given
        if fill_func is not None:
            p = fill_func(p)

        p = p.apply(agg_func, axis=1)
        agg_name = '{}.{}.{}'.format(value_col, interval[0], interval[1])
        parts.append(pd.DataFrame({agg_name: p}))
    return pd.concat(parts, axis=1)


def get_time_aggregates(d, intervals, date_cols, value_cols, agg_func=np.sum, fill_func=None):
    return pd.concat(
        [_get_time_aggregates(d, i, date_cols, value_cols, agg_func, fill_func) for i in intervals], axis=1)


def get_group_time_aggregates(d, intervals, group_cols, date_cols, value_cols, agg_func=np.sum, fill_func=None):
    return d.groupby(group_cols).apply(get_time_aggregates, intervals, date_cols, value_cols, agg_func, fill_func)


def get_aggregate_deltas(d, agg_interval_1, agg_interval_2):
    """ Subtracts col for agg_interval_1 from col for agg_interval_2
    :param d:
    :param agg_interval_1:
    :param agg_interval_2:
    :param value_cols:
    :return:
    """
    value_cols = np.unique([c.split('.')[0] for c in d])
    parts = []
    for col in value_cols:
        i1, i2 = agg_interval_1, agg_interval_2
        c_r = '{}.{}:{}.{}:{}'.format(col, i1[0], i1[1], i2[0], i2[1])
        c_1 = '{}.{}.{}'.format(col, i1[0], i1[1])
        c_2 = '{}.{}.{}'.format(col, i2[0], i2[1])
        parts.append(pd.DataFrame({c_r: d[c_1] - d[c_2]}))
    return pd.concat(parts, axis=1)
