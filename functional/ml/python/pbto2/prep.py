__author__ = 'eczech'

import pandas as pd
import numpy as np
import datetime
import re
import functools

PR_TYPES = 'sluggish|fixed|nonreactive|uta|brisk|brsik|birsk'
PR_TYPE_REGEX = re.compile('({})'.format(PR_TYPES))
PR_LEFT_REGEX = re.compile('left - ({})'.format(PR_TYPES))
PR_RIGHT_REGEX = re.compile('right - ({})'.format(PR_TYPES))
PS_SIZE_REGEX = re.compile('(\d+)mm')
PS_LEFT_REGEX = re.compile('left - (\d+)mm')
PS_RIGHT_REGEX = re.compile('right - (\d+)mm')
NUM_REGEX = re.compile("[-+]?\d*\.\d+|\d+")
DATE_FREQ_MIN = 60

COL_TRANS_MAP = {
    'right_pupil_response': 'pupil_response_r',
    'left_pupil_response' : 'pupil_response_l',
    'right_pupil_size'    : 'pupil_size_r',
    'left_pupil_size'     : 'pupil_size_l',
    'icp_1'               : 'icp1'
}


def extract_first_float(x):
    if isinstance(x, float) or isinstance(x, np.float64):
        return x
    try:
        return float(NUM_REGEX.findall(str(x))[0])
    except:
        return None


def normalize_col(c):
    c = c.strip().lower().replace(' ', '_').replace(',', '')
    return COL_TRANS_MAP.get(c, c)


def find_sheets_by_col(d, d_sheet, col, first_only=False):
    """ Utility function for searching for sheets based on a column name"""
    res = []
    for k in d_sheet:
        df = d[k].copy()
        df.columns = [normalize_col(c) for c in df]
        df['sheet_name'] = int(k)
        if col is None or normalize_col(col) in df:
            if first_only:
                return df
            res.append(df)
    return functools.reduce(pd.DataFrame.append, res)


def split_pupil_response(d):
    def find_value(x, regex):
        """ Extract numeric pupil response category from value (using given regex), if it exists"""
        try:
            return regex.findall(x.lower())[0]
        except:
            return None
    d['pupil_response_l2'] = d['pupil_response'].apply(find_value, args=(PR_LEFT_REGEX,))
    d['pupil_response_r2'] = d['pupil_response'].apply(find_value, args=(PR_RIGHT_REGEX,))
    return d.drop('pupil_response', axis=1)


def convert_pupil_response(x):
    if pd.isnull(x) or isinstance(x, datetime.datetime):
        return None
    try:
        res = PR_TYPE_REGEX.findall(x.lower())[0]
        return {'bikrs': 'brisk'}.get(''.join(sorted(res)), res)
    except:
        return None


def prep_pupil_response(d):
    d = split_pupil_response(d)
    d['pupil_response_l'] = d['pupil_response_l'].apply(convert_pupil_response)
    d['pupil_response_r'] = d['pupil_response_r'].apply(convert_pupil_response)
    d['pupil_response_l2'] = d['pupil_response_l2'].apply(convert_pupil_response)
    d['pupil_response_r2'] = d['pupil_response_r2'].apply(convert_pupil_response)
    d['pupil_response_l'] = np.where(d['pupil_response_l'].isnull(), d['pupil_response_l2'], d['pupil_response_l'])
    d['pupil_response_r'] = np.where(d['pupil_response_r'].isnull(), d['pupil_response_r2'], d['pupil_response_r'])
    d = d.drop(['pupil_response_l2', 'pupil_response_r2'], axis=1)

    pupil_res = list(d['pupil_response_r'].unique())
    pupil_res += list(d['pupil_response_l'].unique())
    pupil_res = [v for v in pupil_res if not pd.isnull(v)]
    pupil_res = list(np.unique(pupil_res))
    assert pupil_res == ['brisk', 'fixed', 'nonreactive', 'sluggish', 'uta'], \
        'Found unexpected pupil response values'
    res_map = {'uta': np.nan, 'nonreactive': 1, 'fixed': 2, 'sluggish': 3, 'brisk': 4}
    d['pupil_response_r'] = d['pupil_response_r'].apply(lambda x: res_map.get(x, np.nan))
    d['pupil_response_l'] = d['pupil_response_l'].apply(lambda x: res_map.get(x, np.nan))
    return d


def split_pupil_size(d):
    def find_value(x, regex):
        """ Extract numeric pupil size in mm from value (using given regex), if it exists"""
        try:
            return regex.findall(x.lower())[0]
        except:
            return None
    d['pupil_size_l2'] = d['pupil_size'].apply(find_value, args=(PS_LEFT_REGEX,))
    d['pupil_size_r2'] = d['pupil_size'].apply(find_value, args=(PS_RIGHT_REGEX,))
    return d.drop('pupil_size', axis=1)


def convert_pupil_size(x):
    if pd.isnull(x) or isinstance(x, datetime.datetime):
        return None
    try:
        return float(x)
    except:
        try:
            float(PS_SIZE_REGEX.findall(x.lower())[0])
        except:
            return None


def prep_pupil_size(d):
    d = split_pupil_size(d)
    d['pupil_size_l'] = d['pupil_size_l'].apply(convert_pupil_size)
    d['pupil_size_r'] = d['pupil_size_r'].apply(convert_pupil_size)
    d['pupil_size_l2'] = d['pupil_size_l2'].apply(convert_pupil_size)
    d['pupil_size_r2'] = d['pupil_size_r2'].apply(convert_pupil_size)
    d['pupil_size_l'] = np.where(d['pupil_size_l'].isnull(), d['pupil_size_l2'], d['pupil_size_l'])
    d['pupil_size_r'] = np.where(d['pupil_size_r'].isnull(), d['pupil_size_r2'], d['pupil_size_r'])
    d['pupil_size_l'] = d['pupil_size_l'].astype(np.float64)
    d['pupil_size_r'] = d['pupil_size_l'].astype(np.float64)
    d = d.drop(['pupil_size_l2', 'pupil_size_r2'], axis=1)
    return d


# Deprecated
# def interp_timeseries(x):
#     """ Interpolate data points for each patient, filling values with those closest in time"""
#     x = x.set_index('datetime')
#     idx = pd.date_range(x.index.min(), x.index.max(), freq=str(DATE_FREQ_MIN)+'Min', name='datetime')
#     return x.reindex(idx, method='nearest')


def interp_timeseries_linear(x, limit=4):
    x = x.set_index('datetime').drop('uid', axis=1)
    idx = pd.date_range(x.index.min(), x.index.max(), freq=str(DATE_FREQ_MIN)+'Min', name='datetime')
    return x.reindex(idx, method=None).interpolate(limit=limit)


def convert_date(x):
    r = pd.to_datetime(x)
    if not isinstance(r, pd.Timestamp):
        return None
    # Return date converted to nearest frequency interval
    return pd.to_datetime(DATE_FREQ_MIN * round(r.value / (DATE_FREQ_MIN * 60 * 1.E9)), unit='m')


def clean_bad_pbto2_values(d):
    """ Removes bad PbtO2 measurements based on a visual inspection

    See 2_data_models_long for details
    :param d: Long format data frame with pbto2, datetime, and uid
    :return: Data frame in same form with bad measurements removed
    """
    def remove_after_first_zero(df, var, uids):
        d1 = df[~df['uid'].isin(uids)]
        d2 = df[df['uid'].isin(uids)]
        d2 = d2.groupby('uid', group_keys=False)\
            .apply(lambda x: x[x['datetime'] < x[x[var] <= 0]['datetime'].min()])
        return d1.append(d2)

    rm_uids = [827]
    d_clean = d[~d['uid'].isin(rm_uids)].copy()

    rm_zero_uids = [751, 772]
    d_clean = d_clean[(~d_clean['uid'].isin(rm_zero_uids)) | (d_clean['pbto2'] > 0)]

    trim_uids = [566, 588, 593, 1001, 1007]
    d_clean = remove_after_first_zero(d_clean, 'pbto2', trim_uids)

    assert len(d['uid'].unique()) - len(rm_uids) == len(d_clean['uid'].unique())
    return d_clean
