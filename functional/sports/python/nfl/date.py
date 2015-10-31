__author__ = 'eczech'

import pandas as pd
from nfl import storage as nfl_db
from nfl.fdb import const as fdb_const
from pandas.tseries.offsets import Week


# Split weeks on Wednesdays
WEEK = Week(weekday=2, normalize=True)


def truncate_date(d):
    if pd.isnull(d):
        return d
    return WEEK.apply(d) - 1 * WEEK


def get_reg_season_weeks():
    games = nfl_db.get_data(fdb_const.DB, fdb_const.COLL_GAMES)
    games = games[games['GameType'] == 'Regular Season']
    games.head()

    dates = games[['Year', 'Date']].drop_duplicates()
    dates['Week'] = dates['Date'].apply(truncate_date)

    def adjust_date(x):
        min_week = x['Week'].min()
        res = x.set_index('Date')['Week'].apply(lambda w: int(1 + (w - min_week).delta / (604800 * 1E9)))
        res.name = 'WeekNumber'
        return res

    dates = dates.groupby('Year').apply(adjust_date).reset_index()

    # Apply known adjustment for 2001 season due to Sep 11th, which caused
    # a week to be skipped and all week numbers after the first to be 1 too high
    def sep_11_adj(r):
        return r['WeekNumber'] if r['Year'] != 2001 or r['WeekNumber'] == 1 else r['WeekNumber']-1
    dates['WeekNumber'] = dates.apply(sep_11_adj, axis=1).values

    return dates

