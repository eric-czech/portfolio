__author__ = 'eczech'


import numpy as np
from nfl.sos import storage as sos_db
from nfl.fo import storage as fo_db
from nfl.fdb import features as nfl_feat
from nfl import storage as nfl_db
from nfl.fdb import const as fdb_const
import pandas as pd


def _add_game_fractions(X):
    X['Attr:Player:GS:2+yr'] = X['Attr:Player:GS:2yr'] + X['Attr:Player:GS:3yr'] + X['Attr:Player:GS:4+yr']
    X = X.drop(['Attr:Player:GS:0yr', 'Attr:Player:GS:2yr', 'Attr:Player:GS:3yr', 'Attr:Player:GS:4+yr'], axis=1)
    #X = X.drop(['Attr:Player:GS:2yr', 'Attr:Player:GS:3yr', 'Attr:Player:GS:4+yr'], axis=1)
    X = X.drop(['Attr:Player:G:0yr', 'Attr:Player:G:2yr', 'Attr:Player:G:3yr', 'Attr:Player:G:4+yr'], axis=1)
    #X = X.drop(['Attr:Player:G:2yr', 'Attr:Player:G:3yr', 'Attr:Player:G:4+yr'], axis=1)

    X['Attr:Player:GF:1yr'] = X['Attr:Player:GS:1yr'] / X['Attr:Player:G:1yr']
    X['Attr:Player:GF:1yr'] = np.where(X['Attr:Player:GF:1yr'].isnull(), 0., X['Attr:Player:GF:1yr'])

    X['Stat:Player:ContribGF'] = X['Stat:Player:ContribG'] / (X['Attr:Game:GameNumber'] - 1)
    X['Stat:Player:ContribGF'] = np.where(X['Stat:Player:ContribGF'].isnull(), 0., X['Stat:Player:ContribGF'])
    return X


def prep(X):
    def get_opponent(x):
        if x['Attr:Player:TeamId'] == x['Attr:Game:TeamId']:
            return x['Attr:Game:OpponentTeamId']
        else:
            return x['Attr:Game:TeamId']
    X['Attr:Player:OpponentTeamId'] = X.apply(get_opponent, axis=1)
    X = X.drop(['Attr:Game:TeamId', 'Attr:Game:OpponentTeamId'], axis=1)

    id_cols = [
        'Attr:Player:PlayerLink', 'Attr:Player:TeamId', 'Attr:Player:OpponentTeamId',
        'Attr:Game:OutcomeLink', 'Attr:Game:Year', 'Attr:Game:Date'
    ]

    X = X.set_index(id_cols)
    X['Attr:Player:IsHome'] = X['Attr:Player:IsHome'].astype(np.int64)
    X = _add_game_fractions(X)

    c_keep = [c for c in X if c.startswith('Attr:Player:G')]
    c_keep += [c for c in X if 'DVOA' in c]
    c_keep += [c for c in X if c.startswith('Stat:Player:Contrib')]
    c_keep += [c for c in X if c.startswith('Attr:Weather')]
    c_keep += ['Attr:Game:GameNumber', 'Attr:Player:IsHome']
    c_attr = [c for c in X if c.startswith('Attr:') and not c in c_keep]
    X = X.drop(c_attr, axis=1)

    return X, c_keep


def_pos = ['LB', 'DL', 'DT', 'DE', 'DB']
off_line_pos = ['C', 'OG', 'OT', 'OL']
def_stat = ['Ast', 'Sack', 'Solo', 'Int']
off_passers = ['QB']
off_rushers = ['QB', 'RB']
off_receivers = ['RB', 'TE', 'WR']
off_pos = ['C', 'K', 'OG', 'OL', 'OT', 'P', 'QB', 'RB', 'TE', 'WR']
#stat_passers = ['Cmp', 'Att', 'Int', 'Rate', 'Yds', 'YPA']
stat_passers = ['Rate', 'Yds', 'YPA', 'Att', 'TD', 'Int']
stat_rushers = ['Att', 'Yds', 'Avg', 'TD']


def prep_QB(X, c_keep):
    def filter(c):
        res = False
        res |= c.startswith('Stat:Player:') and \
            (':passing:' in c or ':rushing:' in c) and \
            c.split(':')[3] in stat_passers

        # Allow stats about this teams qb and rbs as well
        # as what other qbs and rbs have done against them
        res |= c.startswith('Stat:Team') and \
            c.split(':')[2] in ['QB'] and \
            c.split(':')[3] in ['passing'] and \
            c.split(':')[4] in stat_passers and \
            c.split(':')[5] in ['1st']
        res |= c.startswith('Stat:Team') and \
            c.split(':')[2] in ['RB', 'QB'] and \
            c.split(':')[3] in ['rushing'] and \
            c.split(':')[4] in stat_rushers and \
            c.split(':')[5] in ['1st']

        # Allow defensive stats other teams have put up against
        # this one
        # res |= c.startswith('Stat:Team') and \
        #     c.split(':')[2] in def_pos and \
        #     c.split(':')[3] in ['defense', 'fumbles'] and \
        #     c.split(':')[4] in def_stat + ['Fum'] and \
        #     c.split(':')[5] == '2nd'

        # Allow opponents defensive stats
        # res |= c.startswith('Stat:Opp:') and \
        #     c.split(':')[2] in def_pos and \
        #     c.split(':')[3] == 'defense' and \
        #     c.split(':')[4] in def_stat and \
        #     c.split(':')[5] == '1st'
        # Allow opponent stats for what other QB's have put up against them
        res |= c.startswith('Stat:Opp:') and \
            c.split(':')[2] in ['QB'] and \
            c.split(':')[3] in ['passing'] and \
            c.split(':')[4] in stat_passers and \
            c.split(':')[5] == '2nd'
        # Also allow rushing numbers put up against opponent
        res |= c.startswith('Stat:Opp:') and \
            c.split(':')[2] in ['RB', 'QB'] and \
            c.split(':')[3] in ['rushing'] and \
            c.split(':')[4] in stat_rushers and \
            c.split(':')[5] == '2nd'
        return res
    X = X[[c for c in X if c in c_keep or filter(c)]]
    return X


def prep_RB(X, c_keep):
    def filter(c):
        res = False
        res |= c.startswith('Stat:Player:') and ':rushing:' in c and \
            c.split(':')[3] in stat_rushers

        # Allow numbers for what the player's team puts up
        # in rushing and passing regularly
        res |= c.startswith('Stat:Team') and \
            c.split(':')[2] in ['QB'] and \
            c.split(':')[3] in ['passing'] and \
            c.split(':')[4] in stat_passers and \
            c.split(':')[5] in ['1st']
        res |= c.startswith('Stat:Team') and \
            c.split(':')[2] in ['RB', 'QB'] and \
            c.split(':')[3] in ['rushing'] and \
            c.split(':')[4] in stat_rushers and \
            c.split(':')[5] in ['1st']

        # Allow the same as above, except accept numbers for
        # the same positions and statistics put up against the
        # opposing team
        res |= c.startswith('Stat:Opp:') and \
            c.split(':')[2] in ['QB'] and \
            c.split(':')[3] in ['passing'] and \
            c.split(':')[4] in stat_passers and \
            c.split(':')[5] == '2nd'
        res |= c.startswith('Stat:Opp:') and \
            c.split(':')[2] in ['RB', 'QB'] and \
            c.split(':')[3] in ['rushing'] and \
            c.split(':')[4] in stat_rushers and \
            c.split(':')[5] == '2nd'
        return res
    X = X[[c for c in X if c in c_keep or filter(c)]]
    return X


def prep_pos(X, position, keep):
    X, c_keep = prep(X)
    c_keep += keep
    if position == 'QB':
        X = prep_QB(X, c_keep)
    elif position == 'RB':
        X = prep_RB(X, c_keep)
    else:
        raise ValueError('Position {} not yet supported'.format(position))

    return X


def _get_training_data(data, response, position, keep=[]):
    data = data[data['Attr:Player:Pos'] == position]
    return prep_pos(data, position, keep), data[response]


def resolve_sos_adjusted_values(X, adjusted_cols):
    """
    Resolves values stored as dictionaries with SOS adjusted averages to expected
    values on the original scale

    :param X: Feature DataFrame
    :param adjusted_cols: Dict with keys of columns to return SOS-adjusted
        values for and values equal to another dictionary with SOS type for adjusted value
        as well as a damping factor filter list.  Examples:
        {'Stat:Player:passing:Yds': {'sos_damping_filter': ['90', '92'], 'sos_type': 'passing_defense'}}
        {'Stat:Player:passing:Yds': {'sos_damping_filter': None, 'sos_type': 'passing_defense'}}
    :return:
    """
    res = X.copy()
    X = X.reset_index()
    sos_fetcher, damping_factors = sos_db.get_sos_fetcher(X['Attr:Game:Year'].unique())
    for col in X:
        # Only columns containing dictionary values need to be resolved
        if not np.all(X[col].apply(lambda x: pd.isnull(x) or isinstance(x, dict))):
            continue

        # Extract original, unadjusted values from dictionaries (dict key == 'v')
        non_adj_value = X[col].apply(lambda x: np.nan if pd.isnull(x) else x['v']).values

        # If this is a dict-valued column but not in the map of
        # columns to return adjusted values for, just return the
        # actual, unadjusted value instead
        if col not in adjusted_cols:
            res[col] = non_adj_value
            continue

        res[col+':Adj:None'] = non_adj_value

        # Otherwise, create new features for each adjusted value
        # for each damping factor present
        # value form: {'a': {'20': 8.5, '30': 8.52, '40': 8.520, ...}, 'n': 3, 'v': 272.6666666666667}
        damping_factor_filter = adjusted_cols[col]['sos_damping_filter']
        for d in damping_factors:
            if damping_factor_filter is not None and d not in damping_factor_filter:
                continue
            adj_val = []
            for i, r in X.iterrows():
                if pd.isnull(r[col]):
                    adj_val.append(np.nan)
                    continue
                year = r['Attr:Game:Year']
                date = r['Attr:Game:Date']
                team = r['Attr:Player:OpponentTeamId']
                val = r[col]['a'][d] / sos_fetcher(adjusted_cols[col]['sos_type'], year, date, team)[d]
                adj_val.append(val)
            res[col+':Adj:'+d] = adj_val
        res = res.drop(col, axis=1)
    return res


def get_position_data(years, position, response, sos_adj_values,
                     min_game=8, min_response=None, top_player_pct=.25,
                     add_weather=True, non_na_cols=None):
    """
    Creates a training data set for the given parameters

    :param years: Tuple with two items, minimum year and maximum year to get training data for (inclusive)
    :param position: Player position to get training data for (e.g. 'QB', 'RB')
    :param response: Name of field to use as response
    :param sos_adj_values: Map of fields to get SOS adjusted values for (see #resolve_sos_adjusted_values)
    :param min_game: Minimum game number to get training data for
    :param min_response: Minimum values for response field (can be None, which means simply "non-NA")
    :param top_player_pct: Percentile of top-performing players to keep
    :param add_weather: Flag for whether or not weather statistics should be added
    :param non_na_cols: Columns used to limit dataset for based on whether or not these values are NA
    :return: X, y
    """

    d = nfl_db.get_data(fdb_const.DB, fdb_const.COLL_TRAIN_V1, query={
            'Attr:Game:Year': {'$gte': years[0], '$lte': years[1]},
            'Attr:Player:Pos': position,
            'Attr:Game:GameNumber': {'$gte': min_game}
    })

    # Subset to only records where response > #min_response, or just subset
    # to response != nan if no #min_response given
    if min_response is not None:
        d = d[d[response] > min_response]
    else:
        d = d[~d[response].isnull()]

    # Subset to records where all fields in #non_na_cols are not null, if any
    # such columns were given
    if non_na_cols is not None:
        for c in non_na_cols:
            d = d[~d[c].isnull()]

    # Subset data to only the top #top_player_pct of players based on response value
    if top_player_pct is not None:
        n_players = len(d['Attr:Player:PlayerLink'].unique())
        top_players = d.groupby('Attr:Player:PlayerLink')[response]\
            .mean().order(ascending=False).reset_index()\
            .head(int(n_players * top_player_pct))
        d = d[d['Attr:Player:PlayerLink'].isin(top_players['Attr:Player:PlayerLink'])]

    # Add weather to training data
    if add_weather:
        stadium_weather = nfl_feat.get_stadium_weather()
        c_map = {
            'Attr:Weather:LocationString': 'Attr:Game:Location',
            'Attr:Weather:Date': 'Attr:Game:Date'
        }
        d = pd.merge(
            d, stadium_weather.rename(columns=c_map),
            on=['Attr:Game:Location', 'Attr:Game:Date'],
            how='left'
        )

    # Create initial training set using field filtering logic based on player position
    weather_feats = [c for c in d if c.startswith('Attr:Weather')]
    X, y = _get_training_data(d, response, position, keep=weather_feats)

    # Resolve SOS-adjusted values
    X = resolve_sos_adjusted_values(X, sos_adj_values)

    return X, y


def _log_ratio(X, col1, col2):
    res = np.log(X[col1] / X[col2])
    min_ratio = res.apply(lambda x: x if np.isfinite(x) else np.nan).dropna().min()
    return res.apply(lambda x: x if np.isfinite(x) else min_ratio)


def _diff(X, col1, col2):
    return X[col1] - X[col2]


def add_binary_derived_fields(X, derived_fields):
    X = X.copy()
    for field in derived_fields:
        if field['op'] == 'diff':
            func = _diff
        elif field['op'] == 'log_ratio':
            func = _log_ratio
        else:
            raise ValueError('Operation "{}" is not supported or not valid'.format(field['op']))
        X[field['result']] = func(X, field['left_field'], field['right_field'])
    return X


def prep_dvoa_derived(X):
    derived_fields = [
        {'op': 'diff', 'result': 'Stat:Team:Total DVOA:Diff',
         'left_field': 'Stat:Team:All:Total DVOA:1st', 'right_field': 'Stat:Opp:All:Total DVOA:1st'},
        {'op': 'diff', 'result': 'Stat:Team:Offense DVOA:Diff',
         'left_field': 'Stat:Team:All:Offense DVOA:1st', 'right_field': 'Stat:Opp:All:Defense DVOA:1st'},
        {'op': 'diff', 'result': 'Stat:Team:Pass DVOA:Diff',
         'left_field': 'Stat:Team:All:Off Pass DVOA:1st', 'right_field': 'Stat:Opp:All:Def Pass DVOA:1st'},
        {'op': 'diff', 'result': 'Stat:Team:Rush DVOA:Diff',
         'left_field': 'Stat:Team:All:Off Rush DVOA:1st', 'right_field': 'Stat:Opp:All:Def Rush DVOA:1st'}
    ]
    X = add_binary_derived_fields(X, derived_fields)
    return X.drop([c for c in X if 'All:ST DVOA:1st' in c], axis=1)
