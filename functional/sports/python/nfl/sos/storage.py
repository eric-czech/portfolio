__author__ = 'eczech'


from nfl import storage as nfl_db
from nfl.fdb.const import *


def get_sos_fetcher(years):
    years = [int(y) for y in years]
    sos_raw = nfl_db.get_data(DB, COLL_SOS_V1, {'Year': {'$in': years}})
    damping_factors = sos_raw['Damping'].unique()

    sos = sos_raw.set_index(['Type', 'Damping', 'Year', 'Date'])
    sos.columns.name = 'TeamId'
    sos = sos.stack()
    sos.name = 'Rank'

    sos_def = sos.reset_index().groupby(['Type', 'Damping', 'Year']).apply(lambda x: len(x['TeamId'].unique()))
    sos_def.name = 'DefaultRank'
    sos_def = 1 / sos_def

    def sos_fetcher(sos_type, year, date, team):
        res = {}
        for damping in damping_factors:
            key = (sos_type, damping, year, date, team)
            val = sos.loc[key] if key in sos.index else sos_def.loc[(sos_type, damping, year)]
            res[damping] = val
        return res
    return sos_fetcher, damping_factors


def get_sos_data(years, damping=.15):
    years = [int(y) for y in years]
    sos_raw = nfl_db.get_data(DB, COLL_SOS_V1, {'Year': {'$in': years}, 'Damping': damping})
    sos_raw = sos_raw.drop('Damping', axis=1)
    sos = sos_raw.set_index(['Type', 'Year', 'Date'])
    sos.columns.name = 'TeamId'
    sos = sos.stack()
    sos.name = 'Rank'

    sos_def = sos.reset_index().groupby(['Type', 'Year']).apply(lambda x: len(x['TeamId'].unique()))
    sos_def.name = 'DefaultRank'
    sos_def = 1 / sos_def

    return sos, sos_def
