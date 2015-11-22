__author__ = 'eczech'

import pandas as pd
import numpy as np
from ml import query


def get_raw_data():
    d_exp = '/Users/eczech/data/pbto2/export/data_clean.pkl'
    d_exp = pd.read_pickle(d_exp)
    return d_exp.drop(['icp', 'hco3'], axis=1)


def _get_feature_values(x):
    # Initialize result row to time-invariant values
    r = x.iloc[0].loc[['age', 'sex', 'gcs', 'marshall', 'gos-3', 'gos-6', 'gos-12', 'gos-24']].to_dict()

    # Add summaries about time since injury
    r['tsi_min'] = x['tsi_min'].min()
    r['tsi_max'] = x['tsi_min'].max()
    r['tsi_nhours'] = len(x['tsi_min'])

    # Add summaries over time-variant values
    get_min = lambda v, p: pd.Series([v[p+'_l'].min(), v[p+'_r'].min()]).dropna().min()
    get_max = lambda v, p: pd.Series([v[p+'_l'].max(), v[p+'_r'].max()]).dropna().max()

    r['min_pupil_response'] = get_min(x, 'pupil_response')
    r['max_pupil_response'] = get_max(x, 'pupil_response')

    r['min_pupil_size'] = get_min(x, 'pupil_size')
    r['max_pupil_size'] = get_max(x, 'pupil_size')

    r['max_log_pupil_size_ratio'] = pd.Series(np.abs(np.log(x['pupil_size_l']/x['pupil_size_r']))).dropna().max()
    r['max_diff_pupil_size'] = pd.Series(np.abs(x['pupil_size_l'] - x['pupil_size_r'])).dropna().max()

    # Compute distribution statistics for more dense measurements
    feats = ['pbto2', 'spo2', 'hco3a', 'icp1', 'map', 'paco2', 'pao2', 'pha', 'bo2']
    for feat in feats:
        d = x[feat].dropna()
        if len(d) == 0:
            continue
        pcts = [.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5]
        quants = np.percentile(d, pcts)
        for i, p in enumerate(pcts):
            r['{}_p{}'.format(feat, int(p*10))] = quants[i]
        r[feat+'_min'] = d.min()
        r[feat+'_max'] = d.max()
        r[feat+'_mean'] = d.mean()
        r[feat+'_std'] = d.std()

    # Add special summaries for pbto2
    d = x['pbto2'].dropna()
    d = d[d > 0]  # 0 values are measurement errors
    for t in [5, 10, 15, 20, 25]:
        if len(d) > 0:
            r['pbto2_n_under_{}'.format(t)] = len(d[d < t])
            #r['pbto2_pct_under_{}'.format(t)] = len(d[d < t])/len(d)
            r['pbto2_mean_under_{}'.format(t)] = d[d < t].mean() if not pd.isnull(d[d < t].mean()) else t
        else:
            r['pbto2_n_under_{}'.format(t)] = np.nan
            #r['pbto2_pct_under_{}'.format(t)] = np.nan
            r['pbto2_mean_under_{}'.format(t)] = np.nan
    for t in [30, 35]:
        if len(d) > 0:
            r['pbto2_n_over_{}'.format(t)] = len(d[d > t])
            r['pbto2_mean_over_{}'.format(t)] = d[d > t].mean() if not pd.isnull(d[d > t].mean()) else t
        else:
            r['pbto2_n_over_{}'.format(t)] = np.nan
            r['pbto2_mean_over_{}'.format(t)] = np.nan

    # Add more specific pbto2 features capturing time spent in optimal ranges
    def get_range_pct(v, tsi_max, pbto2_min, pbto2_max):
        v = v.dropna()
        v = v[(v['pbto2'] > 0) & (v['tsi_min'] <= tsi_max)]
        if len(v) == 0:
            return 0
        return len(v[(v['pbto2'] >= pbto2_min) & (v['pbto2'] < pbto2_max)]) / len(v)

    max_tsi = 4320
    r['pbto2_pct_in_range_lo'] = get_range_pct(x[['tsi_min', 'pbto2']], max_tsi, 0, 20)
    r['pbto2_pct_in_range_mi'] = get_range_pct(x[['tsi_min', 'pbto2']], max_tsi, 20, 60)
    r['pbto2_pct_in_range_hi'] = get_range_pct(x[['tsi_min', 'pbto2']], max_tsi, 60, np.inf)

    def get_pbto2_bins(v, tsi_max, bins=[1, 5, 10, 15, 20, 25, 30, np.inf], interpolate=True):
        """
        Returns counts of pbto2 values within given bins.

        Bins are inclusive on the left and will be returned named by left boundary, i.e. if the bin name
        is defined as 5 - 10, the count for bin "5" will be everything in [5, 9]
        """
        v = v.dropna()
        if interpolate:
            v = interpolate_ts(v, 'tsi_min', 'pbto2', 60)
        v = v[(v['pbto2'] > 0) & (v['tsi_min'] <= tsi_max)]
        l = ['pbto2_ct_bin_{}'.format(str(b).zfill(2)) for b in bins[:-1]]
        v = pd.cut(v['pbto2'], bins, include_lowest=True, labels=l).value_counts()
        return v.sort_index()

    r.update(get_pbto2_bins(x[['tsi_min', 'pbto2']], max_tsi).to_dict())

    r = pd.Series(r)

    r.index.name = 'feature'
    return r


def interpolate_ts(v, time, value, freq):
    """ Interpolates values with name #value in a data frame with time value #time """
    v = v[[time, value]]
    v = v.set_index(time)
    idx = np.arange(v.index.min(), v.index.max() + freq, freq)
    v = v.reindex(pd.Int64Index(idx, name=time)).interpolate()
    return v.reset_index()


def get_raw_feature_data(d):
    d = d.groupby('uid').apply(_get_feature_values).unstack().reset_index()
    assert d.groupby('uid').size().max() == 1, 'Found multiple records for the same uid'

    validate_response_features(d)
    d = bin_outcome(d, outcome_favorable_alive, 'favalive')
    d = bin_outcome(d, outcome_favorable_all, 'favorable')
    d = bin_outcome(d, outcome_alive, 'alive')
    return d


def validate_response_features(d):
    for feat in get_gos_feats(d):
        mask = (d[feat].between(1, 5)) | (d[feat].isnull())
        assert np.all(mask), 'GOS feature {} has an invalid value'.format(feat)


def outcome_favorable_alive(x):
    if pd.isnull(x):
        return np.nan
    # GOS of 2 or 3 indicates 'Unfavorable' outcome
    if x in [2, 3]:
        return 0
    # GOS of 4 or 5 indicates 'Favorable' outcome
    if x in [4, 5]:
        return 1
    return np.nan


def outcome_favorable_all(x):
    if pd.isnull(x):
        return np.nan
    # GOS of 2 or 3 indicates 'Unfavorable' outcome, 1 means Dead (also unfavorable here)
    if x in [1, 2, 3]:
        return 0
    # GOS of 4 or 5 indicates 'Favorable' outcome
    if x in [4, 5]:
        return 1
    return np.nan


def outcome_alive(x):
    if pd.isnull(x):
        return np.nan
    return int(x > 1)


def bin_outcome(d, outcome_func, suffix):
    d_res = d.copy()
    gos = get_gos_feats(d_res)
    for feat in gos:
        d_res[feat+'-'+suffix] = d_res[feat].apply(outcome_func)
    return d_res


def get_gos_feats(d):
    r = [c for c in d if c.startswith('gos') and len(c.split('-')) == 2]
    return list(pd.Series(r, index=[int(v.split('-')[1]) for v in r]).sort_index().values)


# def interpolate_gos(d):
#     gos = get_gos_feats(d)
#     d_res = d.copy()
#     for i, r in d_res.iterrows():
#         x = r[gos]
#         x_fill = x.ffill().bfill()
#         d_res.loc[i, gos] = x_fill
#     return d_res

# # Test GOS interpolation
# dt = d_feat.copy()
# dt = dt[dt['uid'] == 628]
# print(dt[gos_feats])
# dt = interpolate_gos(dt)
# dt[gos_feats]


import math
from sklearn.preprocessing import StandardScaler


def _is_outcome(c):
    return c.startswith('gos')


def _impute(d, add_indicators=True):
    d_res = d.copy()
    for c in d_res:
        if not _is_outcome(c) and d_res[c].isnull().sum() > 0:
            fill_value = d_res[c].dropna().mean()
            if add_indicators:
                d_res[c+':ind'] = np.where(d_res[c].isnull(), 1, 0)
            d_res[c] = d_res[c].fillna(fill_value)
    return d_res


def _log(d):
    log_cols = ['pao2']
    d_res = d.copy()
    for c in d:
        if c.split('_')[0] in log_cols:
            d_res[c] = d_res[c].apply(lambda x: math.asinh(x) if not pd.isnull(x) else np.nan)
    return d_res


def _scale(d):
    d_res = d.copy()
    # Scale all values except outcome
    for c in d_res:
        if not _is_outcome(c):
            m = d_res[c].dropna().mean()
            s = d_res[c].dropna().std()
            d_res[c] = d_res[c].apply(lambda x: (x - m)/s)
    return d_res


DEFAULT_FEAT_STATS = [
    (['uid', 'age', 'sex', 'gcs', 'marshall', 'gos-3']),
    (['pbto2'], ['p10', 'std', 'mean']),
    (['pao2', 'icp'], ['mean', 'p100', 'p900', 'std'])
]


def get_prepared_features(d, feats=DEFAULT_FEAT_STATS, add_na_indicators=False, outcome=None,
                          scale=True, impute=True):
    regex = ['uid']
    for feat in feats:
        for v in feat[0]:
            if len(feat) == 1:
                regex.append('({})'.format(v))
                continue
            for s in feat[1]:
                regex.append('({}_{}$)'.format(v, s))

    # print('|'.join(regex))
    d = query.feature_regex_query(d, '|'.join(regex)).set_index('uid')

    na_frac = d.apply(lambda x: x.isnull().sum()/len(x)).order()
    na_frac = na_frac[na_frac > 0]
    na_frac['__n_records__'] = len(d)

    if impute:
        d = _impute(d, add_indicators=add_na_indicators)
    if scale:
        d = _scale(d)
    if outcome is not None:
        d = select_outcome(d, outcome)

    return d, na_frac


def select_outcome(d, outcome):
    d = d.rename(columns={outcome: 'outcome'})
    d = d[~d['outcome'].isnull()]
    return d

