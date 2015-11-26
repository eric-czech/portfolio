__author__ = 'eczech'


import pandas as pd
import numpy as np
import os
import re
import gzip


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# These are the fields in the raw data that will NOT be used as predictors
non_features = ['tumorID', 'res_AUC', 'res_IC50', 'response', 'type']
response = 'response'

REPO = 'https://raw.githubusercontent.com/guester/Charleston-Data-Analytics-Cancer-Genomics-Data-Challenge/master/'


def fetch(url):
    """ Fetch and cache a csv file from the Genomics repo"""
    fpath = '/Users/eczech/data/meetups/genomics/cache/genomics_analytics_{}.pkl'.format(url.split('/')[-1])
    d = None
    if os.path.exists(fpath):
        d = pd.read_pickle(fpath)
    else:
        d = pd.read_csv(url)
        d.to_pickle(fpath)

    # Drop any columns that are exclusively NA
    d = d.dropna(axis=1, how='all')

    # Rename non feature columns
    return d.rename(columns={
        'Z_score': 'response',
        'response_metric_IC50': 'res_IC50',
        'response_metric_AUC': 'res_AUC',
        'IC50': 'res_IC50'
    })


def fetch_all(file_type):
    """ Return all 3 data tables for a given file type (eg COSMIC, HCC, or nothing)"""
    d_o = fetch(REPO + '{}.csv'.format(file_type))
    d_o['type'] = 'o'

    d_h = fetch(REPO + 'HCC_{}.csv'.format(file_type))
    d_h['type'] = 'h'

    d_c = fetch(REPO + 'COSMIC_{}.csv'.format(file_type))
    d_c['type'] = 'c'
    return d_o, d_h, d_c


def merge(d_o, d_h, d_c, na_thresh=10):
    """ Merges original, HCC, and COSMIC datasets together

    Note that currently, COSMIC data will be preferred in the event of a tumor contained in multiple datasets

    :param d_o: Original data
    :param d_h: HCC data
    :param d_c: COSMIC data
    :param na_thresh: If a column has more than this many NA values after the merge, it will be removed
    :return: A single DataFrame containing the merged sets
    """
    d_m = d_o.append(d_h)

    if na_thresh is not None:
        n_ct = d_m.apply(lambda x: x.isnull().sum() if x.name not in non_features else 0)
        # print('Frequency of null counts after merge for original data: ')
        # print(n_ct.value_counts())

    # Remove any records for shared tumors
    #d_c = d_c[~d_c['tumorID'].isin(d_m['tumorID'].unique())] # prefer old data
    d_m = d_m[~d_m['tumorID'].isin(d_c['tumorID'].unique())]  # prefer new, COSMIC data

    d_m = d_m.append(d_c)
    if na_thresh is not None:
        n_ct = d_m.apply(lambda x: x.isnull().sum() if x.name not in non_features else 0)
        # print('Frequency of null counts after merge with COSMIC data: ')
        # print(n_ct.value_counts())

        cols = n_ct[n_ct <= na_thresh].index.values
        print('Removing {} cols with more than {} pct NA values'.format(len(n_ct[n_ct > na_thresh]), na_thresh))
        return d_m[cols]

    return d_m


def resolve_missing_response(datasets):
    answers = fetch(REPO + 'answer_key_all_samples.csv')
    for k in datasets:
        for c in ['res_AUC', 'res_IC50', 'response']:
            resolve_missing_response_in_col(datasets[k]['data'], answers, c)


def resolve_missing_response_in_col(d, answers, col):

    m = {r['tumorID']: r[col] for i, r in answers.iterrows()}
    x = []
    for i, r in d[['tumorID', col]].iterrows():
        if not pd.isnull(r[col]) and r[col] != 'hidden':
            x.append(r[col])
        else:
            x.append(m.get(r['tumorID'], r[col]))
    d[col] = np.array(x, dtype=np.float64)


def validate_tumor_uniqueness(datasets):
    for k in datasets:
        ct = datasets[k]['data'].groupby('tumorID').size().value_counts()
        assert len(ct) == 1 and ct.index.values[0] == 1, 'Found a duplicated tumor in dataset "{}"'.format(k)


def validate_non_feature_equality(datasets):
    def validate_equality(col, datasets):
        d = pd.DataFrame({k:datasets[k]['data'].sort('tumorID')[col].values for k in datasets})

        idx_cols = ['tumorID', 'type']
        idx = datasets[list(datasets.keys())[0]]['data'].sort('tumorID')[idx_cols]
        d[idx_cols] = idx.reset_index(drop=True)
        d = d.set_index(idx_cols)

        d = d.dropna(how='all')
        assert len(d) > 0
        equals = d.apply(lambda x: len(np.unique(x)) == 1, axis=1)
        if not np.all(equals):
            print(d[~equals])
            raise ValueError('Found differing values in raw datasets for column {}'.format(col))

    # Ignoring 'response' for now becuase it is different across all 3 datasets
    # for new COSMIC tables
    for col in ['tumorID', 'res_AUC', 'res_IC50']:
        validate_equality(col, datasets)


def get_raw_datasets():
    d_o_cn, d_h_cn, d_c_cn = fetch_all('copy_number_log2ratios')
    d_o_ge, d_h_ge, d_c_ge = fetch_all('gene_expression')
    d_o_mu, d_h_mu, d_c_mu = fetch_all('mutations')

    # Remove unnecessary tumors in COSMIC ge data
    bad_ids = set(list(d_c_mu.tumorID) + list(d_c_cn.tumorID))
    bad_ids = set(d_c_ge.tumorID) - bad_ids
    d_c_ge = d_c_ge[~d_c_ge.tumorID.isin(bad_ids)]
    assert len(bad_ids) == 34, \
        'Expected removal of 34 extra tumors in COSMIC gene expression, found {} instead'.format(len(bad_ids))

    # For some reason, the IC50 values for gene expression in the COSMIC
    # datasets is different than CN and MU.  Correct that here:
    res_map = dict(zip(d_c_mu['tumorID'], d_c_mu['res_IC50']))
    d_c_ge['res_IC50'] = d_c_ge['tumorID'].apply(lambda x: res_map[x])  # Throws an error if a tumor ID is not present

    # Create a dictionary with all datsets merged together
    datasets = {
        # Copy Number data
        'cn': {'type': 'numeric', 'data': merge(d_o_cn, d_h_cn, d_c_cn)},

        # Gene Expression data
        'ge': {'type': 'numeric', 'data': merge(d_o_ge, d_h_ge, d_c_ge)},

        # Gene Mutation data
        'mu': {'type': 'character', 'data': merge(d_o_mu, d_h_mu, d_c_mu, na_thresh=None)}
    }

    # Fill in response values for any records originally containing "hidden" as a value
    resolve_missing_response(datasets)

    # Verify that there is now no more than 1 record for each tumor ID
    validate_tumor_uniqueness(datasets)

    # Make sure that IC50 and AUC values are the same across all 3 data types (ge, mu, and cn)
    validate_non_feature_equality(datasets)

    return datasets


def all_datasets(datasets):
    return [d['data'] for d in datasets.values()]


def numeric_datasets(datasets):
    return [d['data'] for d in datasets.values() if d['type'] == 'numeric']


def get_sparse_tumor_ids(datasets, na_thresh=.9):
    na_cases = []
    for d in numeric_datasets(datasets):
        na_ct = d.set_index(non_features).apply(lambda x: np.sum(pd.isnull(x))/len(x), axis=1)
        print('Frequency of row-wise NA pct values over threshold for removal:')
        print(na_ct[na_ct > 0].value_counts())
        na_cases.extend(na_ct[na_ct > na_thresh].reset_index()['tumorID'])
    na_cases = list(np.unique(na_cases))
    return na_cases


def join_all_datasets(datasets, na_thresh=.9):
    # Merge all data on non-feature index

    na_cases = get_sparse_tumor_ids(datasets, na_thresh=na_thresh)
    print('The following tumors will be removed due to high sparsity in their feature values: {}'.format(na_cases))

    join_features = ['tumorID', 'res_AUC', 'res_IC50', 'type']
    data = pd.concat([
        datasets[k]['data']\
            .drop('response', axis=1)\
            .set_index(join_features)\
            .rename(columns=lambda c: ':'.join([k, c])) \
        for k in datasets.keys()],
        axis=1
    ).reset_index()

    # Remove rows with mostly NA's in one of the numeric datasets (computed above)
    data = data[~data['tumorID'].isin(na_cases)]

    return data


def get_response_value(data):
    """ Computes response as normalized AUC or IC50, depending on which is present (preferring IC50 if both present)"""
    # Calculate mean and std for each sensitivity measurement type,
    # where present (and also grouped by data source type)
    mean_ic50 = data.groupby('type')['res_IC50'].mean().dropna().to_dict()
    std_ic50 = data.groupby('type')['res_IC50'].std().dropna().to_dict()
    mean_auc = data.groupby('type')['res_AUC'].mean().dropna().to_dict()
    std_auc = data.groupby('type')['res_AUC'].std().dropna().to_dict()

    def compute_response(x):
        t = x['type']
        # Return nan if this type has no response values
        if not t in mean_ic50 and not t in mean_auc:
            return np.nan

        # Make sure at least one response value is not null
        if pd.isnull(x['res_AUC']) and pd.isnull(x['res_IC50']):
            raise ValueError('AUC and IC50 are null')

        # Prefer IC50 if both response values are present
        if not pd.isnull(x['res_IC50']):
            return (x['res_IC50'] - mean_ic50[t]) / std_ic50[t]
        return (x['res_AUC'] - mean_auc[t]) / std_auc[t]

    # Compute the overall response by switching on where AUC or IC50 is present
    return data.apply(compute_response, axis=1).values

# -------------------
# Feature Preparation
# -------------------

REGEX = re.compile('\w|-')


def get_hold_out_mask(data, n_ho):
    np.random.seed(123)  # Make sure the random selection is the same each time
    idx = np.random.permutation(np.arange(0, len(data)))[:n_ho]
    return np.array([True if i in idx else False for i in range(len(data))])


def clean_string(x):
    return ''.join(REGEX.findall(x.replace('_', '-')))


def encode_strings(d, c):
    feats = []
    for v in d[c]:
        if pd.isnull(v):
            feats.append({})
            continue
        vals = v.split(',')
        feats.append({c+':'+clean_string(val):1 for val in vals})
    return pd.DataFrame(feats)


def prep_data(d, scalers=None):
    d = d.set_index(non_features)

    # This dictionary will contain a mapping from feature name to standardizing scaler and
    # may be passed in from a previous preparation (ie the scalers may have been fit on a training
    # set and should be used here on a hold out set)
    if scalers is None:
        scalers = {}

    res = {}
    for c in d.columns.tolist():
        # If the column is a string, create a dummy encoding for the column
        # that includes NA but then uses NA as the reference level (by dropping it)
        if d[c].dtype == np.object:
            feats = []
            for v in d[c]:
                if pd.isnull(v):
                    feats.append({})
                    continue
                vals = v.split(',')
                feats.append({c+':'+clean_string(val): 1 for val in vals})
            feats = pd.DataFrame(feats).fillna(0).apply(lambda x: x.astype(np.int64))
            for feat in feats:
                assert feat not in res, 'Key "{}" already exists in result'.format(feat)
                res[feat] = feats[feat].values
            #if feats.apply(lambda x : x.sum()).sum() > 0:
                #print(feats.apply(lambda x : x.sum()).sum())

        # If the column is a float, assure that it is always present and apply standardization
        elif d[c].dtype == np.float64:
            assert not np.any(d[c].isnull()), 'Found NA values for column {}'.format(c)
            assert c not in res, 'Key "{}" already exists in result'.format(c)
            if c in scalers:
                scaler = scalers[c]
            else:
                scaler = StandardScaler().fit(d[c])
                scalers[c] = scaler
            res[c] = scaler.transform(d[c])

        else:
            raise ValueError('Data type "{}" for column "{}" is not yet supported'.format(d[c].dtype, c))
    res = pd.DataFrame(res).fillna(0)
    res.index = d.index
    return res, scalers


def normalize_data_split(d_tr_prep, d_ho_prep, min_mu_ct=3):
    """ Removes infrequent mutations as well as any columns in hold out set not in training"""

    d_tr = d_tr_prep.copy()
    d_ho = d_ho_prep.copy()

    # Remove infrequent mutations
    drop_cols = [c for c in d_tr if c.startswith('mu:')]
    drop_cols = d_tr[drop_cols].apply(lambda x: x.fillna(0).sum() < min_mu_ct)
    drop_cols = drop_cols[drop_cols].index.values
    d_tr = d_tr.drop(drop_cols, axis=1)

    # Accept columns in hold out set only if they're in training set
    d_ho = d_ho[[c for c in d_ho if c in d_tr.columns]]

    # At this point the hold out set should have fewer columns than the training set, but
    # those missing columns should all be due to non-present mutations.  Verify that here
    # and then add in those missing mutation features with 0 values so that both data sets
    # have equivalent features
    c_mu_tr = [c for c in d_tr if c.startswith('mu:')]
    c_mu_ho = [c for c in d_ho if c.startswith('mu:')]
    diff_mu = list(set(c_mu_tr) - set(c_mu_ho))
    assert len(diff_mu) == len(d_tr.columns) - len(d_ho.columns), \
        'Failed to verify that missing features in hold out set were all non-present mutations'

    # Adding in missing mutations with 0 values
    for c in diff_mu:
        d_ho[c] = 0

    # And finally, make sure training and hold out have same number of columns
    assert len(d_tr.columns) == len(d_ho.columns), 'Training and hold out do not have same number of columns'

    return d_tr, d_ho


def select_features(d_tr, d_ho, k=8000):
    """ Selects features using univariate F test
    :param d_tr: Training data
    :param d_ho: Hold out data
    :param k: Number of features to keep
    :return: Training and holding data with subset of features
    """
    selector = SelectKBest(f_regression, k=k)
    selector.fit(d_tr, d_tr.reset_index()['response'])
    cols = d_tr.columns[selector.get_support()]
    return d_tr[cols], d_ho[cols]


# -------------------
# Data Import/Export
# -------------------

PATH = '/Users/eczech/repos/misc/Charleston-Data-Analytics-Cancer-Genomics-Data-Challenge/modeling/data'
EXPORT = PATH + '/data_n_ho_{}_{}.csv.gz'


def export_dataset(d_tr, d_ho, n_ho):
    with gzip.open(EXPORT.format(n_ho, 'training'), 'wt') as f:
        d_tr.reset_index().to_csv(f, index=False)
    with gzip.open(EXPORT.format(n_ho, 'holdout'), 'wt') as f:
        d_ho.reset_index().to_csv(f, index=False)


def load_dataset(n_ho):
    id_features = [c for c in non_features if c != 'response']

    d_tr = pd.read_csv(EXPORT.format(n_ho, 'training')).set_index(id_features)
    d_ho = pd.read_csv(EXPORT.format(n_ho, 'holdout')).set_index(id_features)
    return d_tr, d_ho