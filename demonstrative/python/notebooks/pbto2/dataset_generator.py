
# coding: utf-8

import sys
####### Variables
#export_file = '/Users/eczech/data/pbto2/export/data_clean_interp_4.pkl'
export_file = sys.argv[1]
enable_interp = bool(sys.argv[2])
interp_limit = int(sys.argv[3])
#######
print('Running data generator with args: export_file = {}, enable_interp = {}, interp_limit = {}'.format(export_file, enable_interp, interp_limit))



import pandas as pd
import numpy as np

import re
import functools
import datetime
from pbto2 import prep

d = pd.read_excel('/Users/eczech/data/pbto2/pbto2_20151023_clean.xlsx', None)
is_num = lambda x: re.compile('^\d+$').match(x) is not None
d_sheet = [k for k in d.keys() if is_num(k)]
o_sheet = [k for k in d.keys() if k not in d_sheet]


# Find the unique set of all columns that appear in data sheets
col_lists = [[prep.normalize_col(c) for c in d[k].columns.tolist()] for k in d_sheet]
#pd.Series(pd.DataFrame(col_lists).values.ravel()).order().unique()


dt = prep.find_sheets_by_col(d, d_sheet, None)

# #### Account for "unnamed" Columns

# This should be empty if all "Unnamed" column issues have been addressed.
# This involves at least the following updates to the input spreadsheet:
# - Add 'Vital Signs' header for patient 1046
# - Change ICP header for patient 1052 to ICP1 and ICP2
qc = dt[~dt['sheet_name'].isin([1051, 1055, 1046, 1052, 980])].set_index('sheet_name')[[c for c in dt if 'unnamed' in c]].dropna(how='all', axis=0).iloc[:50,:]
assert len(qc) == 0, 'Found unexpected "unnamed" columns'

dt = dt.drop([c for c in dt if 'unnamed' in c], axis=1)

# ## Field Examination

def type_counts(x):
    return pd.Series([v.__class__.__name__ for v in x]).value_counts()


# ### Date Cols
date_cols = ['blood_gases', 'date', 'date/time', 'vital_signs']

# ### Floating Point Cols

float_cols = [
    'base_deficit_arterial', 'base_excess_arterial', 'hco3', 'hco3a', 
    'icp1', 'icp2', 'map', 'paco2', 'pao2', 'pha', 'sao2', 'spo2'
]


# ### Prep

# In[73]:

d_meas = dt.copy()
d_meas = prep.prep_pupil_size(d_meas)
d_meas = prep.prep_pupil_response(d_meas)
for c in date_cols:
    d_meas[c] = d_meas[c].apply(prep.convert_date)
for c in float_cols:
    d_meas[c] = d_meas[c].apply(prep.extract_first_float)
d_meas = d_meas.rename(columns={'date/time': 'datetime'})

# Take only one record pertaining to each timestamp and patient, preferring those with the most data present
d_meas['num_na'] = d_meas.apply(pd.isnull).sum(axis=1)
d_meas = d_meas.groupby(['uid', 'datetime'], group_keys=False).apply(lambda x: x.sort('num_na').head(1)).drop(['num_na', 'date'], axis=1)

# For now, subset to only these measurements
d_meas = d_meas[['pbto2', 'pao2', 'pha', 'icp1', 'paco2', 'map', 'uid', 'datetime']]

# Interpolate some missing values, if flag enabled
if enable_interp:
    d_meas = d_meas.groupby('uid').apply(prep.interp_timeseries_linear, limit=interp_limit).reset_index()


# # BO2

# In[109]:

d_bo2 = d['BO2&Injy time'].copy()
d_bo2.columns = [prep.normalize_col(c) for c in d_bo2]
d_bo2 = d_bo2.drop(['date'], axis=1)
d_bo2['datetime'] = d_bo2['datetime'].apply(prep.convert_date)

# Take record with max bo2 for each instance of patient and date + time
d_bo2 = d_bo2.groupby(['uid', 'datetime'], group_keys=False).apply(lambda x: x.sort('bo2').tail(1))


# # Demographics

d_demo = d['Demographics'].copy()
d_demo.columns = [prep.normalize_col(c) for c in d_demo]
def prep_itime(x):
    if pd.isnull(x) or not isinstance(x, datetime.time):
        return None
    return x.hour * 3600 #+ x.minute * 60 + x.second 

d_demo['itime_sec'] = d_demo['itime'].apply(prep_itime)

def get_injury_date(x):
    v = (x['idate'].value / 1.E9)
    if not pd.isnull(x['itime_sec']):
        v += x['itime_sec']        
    return pd.to_datetime(v, unit='s')
d_demo['injury_date'] = d_demo.apply(get_injury_date, axis=1)
d_demo = d_demo.drop(['idate', 'itime', 'itime_sec'], axis=1)


# Make sure there is only one outcome record per patient
assert d_demo.groupby('uid').size().max() == 1, 'Found multiple outcomes for same uid'


# # Marshall Scores

d_marsh = d['Marsh & 12mGOS'].copy()
d_marsh = d_marsh.drop('gos12', axis=1)

def resolve_duplicates(x):
    if len(x) == 1:
        return x
    if len(x['marshall'].unique()) != 1:
        raise ValueError('UID {} has multiple marshall scores that are not the same: {}'
                         .format(x['uid'].iloc[0], x['marshall'].unique()))
    return x.head(1)
d_marsh = d_marsh.groupby('uid', group_keys=False).apply(resolve_duplicates).dropna()

# Make sure there is only one outcome record per patient
assert d_marsh.groupby('uid').size().max() == 1, 'Found multiple outcomes for same uid'


# # Outcomes

d_outcome = d['Outcome'].copy()
d_outcome.columns = [prep.normalize_col(c) for c in d_outcome]

d_outcome = d_outcome.set_index(['uid', 'post_injury_test_period']).unstack()
d_outcome.columns = ['{}-{}'.format(c[0], c[1].lower().replace('month', '').strip()) for c in d_outcome]
d_outcome = d_outcome.reset_index()

# Make sure there is only one outcome record per patient
assert d_outcome.groupby('uid').size().max() == 1, 'Found multiple outcomes for same uid'


# # Merge Pre-Checks

# In[97]:

# Determine the unique set of patient ids present across all datasets
d_all = {'d_meas': d_meas, 'd_bo2': d_bo2, 'd_demo': d_demo, 'd_marsh': d_marsh, 'd_outcome': d_outcome}
uid = []
for d in d_all.values():
    uid.extend(d['uid'].unique())
uid = np.unique(uid)
print('Number of patients total = {}'.format(len(uid)))

# Determine the number of uids NOT present in the overall set for each dataset
for d in d_all:
    print('Number of UIDs missing in dataset {} = {}'.format(d, len(np.setdiff1d(uid, d_all[d]['uid'].unique()))))
    


# From this, it seems safe to conclude that many of the "Demographic" patient records don't necessarily have other records so they'll be ignored for now.

# # Feature Generation

# Join the only two datasets that have records over time 
# (all others are specific to a patient but do not vary in time)
d_exp = pd.merge(d_meas, d_bo2, how='outer', on=['uid', 'datetime'])

# Join the above to demographic data, ignoring demographic records that do not match
d_exp = pd.merge(d_exp, d_demo, how='left', on=['uid'])

# Make sure there were no records in 'd_exp' without a record in 'd_demo'
assert d_exp['injury_date'].isnull().sum() == 0, 'Found vital signs data with no corresponding demographic record'
# Make sure non of the vital signs records have a null date
assert d_exp['datetime'].isnull().sum() == 0, 'Found vital signs record with null date'

d_exp['uid'] = d_exp['uid'].astype(np.int64)


def get_times(x):
    # Add time since injury value to data frame for patient
    x['tsi'] = x['datetime'] - x['injury_date']
    x['tsa'] = x['datetime'] - x['datetime'].min()
    return x
d_exp = d_exp.groupby('uid', group_keys=False).apply(get_times)
d_exp['tsi_min'] = [int(x.value / (1E9 * 60)) for x in d_exp['tsi']]
d_exp['tsa_min'] = [int(x.value / (1E9 * 60)) for x in d_exp['tsa']]


# Find patients with injury dates recorded as after first vital signs (shouldn't be possible)
bad_uids = [int(v) for v in d_exp[d_exp['tsi_min'] <= 0]['uid'].unique()]

# Set all tsi_min values to NA for these patients
d_exp['tsi_min'] = d_exp['tsi_min'].where(~d_exp['uid'].isin(bad_uids), np.nan)


# ### Merge Other Datasets

d_exp = pd.merge(d_exp, d_marsh, how='left', on=['uid'])
d_exp = pd.merge(d_exp, d_outcome, how='left', on=['uid'])


# # Export

# In[104]:
print('Saving data set to file {}'.format(export_file))
d_exp.to_pickle(export_file)
#d_exp.to_pickle('/Users/eczech/data/ptbo2/export/data_clean.pkl')


# In[ ]:




