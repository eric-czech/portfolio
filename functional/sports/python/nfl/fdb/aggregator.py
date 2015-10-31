__author__ = 'eczech'

import pandas as pd
import numpy as np
from nfl.sos import storage as sos_db
from nfl.fo import storage as fo_db
from nfl.fdb.const import *


def get_game_aggregate_old(game_player_stats):
    def get_metric_agg(d):
        metric = d['Metric'].iloc[0]
        for m in [':Avg', ':Rate', ':Lg', ':YPA']:
            if metric.endswith(m):
                return d['Value'].mean()
        return d['Value'].sum()
    return game_player_stats.groupby([
                'TimeFrame', 'TeamId', 'OpponentTeamId', 'OutcomeLink',
                'Year', 'Date', 'Pos', 'Metric']).apply(get_metric_agg)


WEIGHTED_METRICS = {
    'passing:YPA': 'passing:Att',
    'passing:Rate': 'passing:Att',
    'rushing:Avg': 'rushing:Att',
    'receiving:Avg': 'receiving:Rec',
    'kickoff_returns:Avg': 'kickoff_returns:Num',
    'punt_returns:Avg': 'punt_returns:Num',
    'kickoffs:Avg': 'kickoffs:Num',
    'punting:Avg': 'punting:Punts'
}
MAX_METRICS = ['Lg']


def get_game_aggregate(game_player_stats, agg_type):
    if agg_type not in [AGG_TYPE_PLAYER_FIRST, AGG_TYPE_TEAM_SECOND, AGG_TYPE_TEAM_FIRST]:
        raise ValueError('Aggregation type "{}" not supported'.format(agg_type))

    group_cols = ['TimeFrame', 'TeamId', 'OpponentTeamId', 'OutcomeLink', 'Year', 'Date', 'Pos']

    # Short-circuit to sums only for player-only aggregations
    if agg_type == AGG_TYPE_PLAYER_FIRST:
        stats = game_player_stats.groupby(group_cols + ['Metric'])['Value'].sum()
        stats.name = 'Value'
        return stats

    # For team aggregations, things are more complicated:
    def get_metric_agg(d):
        metrics = d['Metric'].unique()
        d = d.set_index(['Metric', 'PlayerLink'])['Value'].sort_index()
        res = {}
        for metric in metrics:
            if metric.split(':')[-1] in MAX_METRICS:
                res[metric] = d.loc[metric].max()
                continue
            if metric in WEIGHTED_METRICS:
                w = WEIGHTED_METRICS[metric]
                if w in d.index:
                    res[metric] = np.dot(d.loc[w] / d.loc[w].sum(), d.loc[metric])
                    continue
            res[metric] = d.loc[metric].sum()

        res = pd.Series(res)
        res.name = 'Metric'
        return res

    stats = game_player_stats.groupby(group_cols).apply(get_metric_agg)
    if isinstance(stats, pd.Series):
        names = list(stats.index.names)
        names[-1] = 'Metric'
        stats.index = stats.index.set_names(names)
    else:
        stats = stats.stack()
    stats.name = 'Value'
    return stats


def get_sos_type(pos):
    if pos == 'QB':
        return 'passing:defense'
    if pos == 'RB':
        return 'rushing:defense'
    return None


def get_game_team_dvoa_stats(dvoa_data, game, team_selector):
    key = (game['Year'], game['Date'], team_selector[0], team_selector[1])
    if key not in dvoa_data.index:
        print('Failed to find DVOA data for game {}'.format(game))
        return pd.DataFrame()
    stats = dvoa_data.loc[key].to_dict()
    idx = pd.MultiIndex.from_tuples([('Past', team_selector[0])], names=['TimeFrame', 'TeamId'])
    stats = pd.DataFrame(stats, index=idx)
    stats.columns = [('All', c) for c in stats]
    return stats


def get_game_dvoa_stats(dvoa_data, game):
    d1 = get_game_team_dvoa_stats(dvoa_data, game, (game['TeamId'], game['OpponentTeamId']))
    d2 = get_game_team_dvoa_stats(dvoa_data, game, (game['OpponentTeamId'], game['TeamId']))
    return d1.append(d2)


def get_default_aggregator(year):
    dvoa_data = fo_db.get_derived_game_dvoa([year])
    sos_fetcher, damping_factors = sos_db.get_sos_fetcher([year])

    def sos_aggregator(game, game_player_stats, agg_type):
        # Sum by position and game
        res = get_game_aggregate(game_player_stats, agg_type)

        def sos_averages(x):
            n_games = len(x['OutcomeLink'].unique())
            sos_type = get_sos_type(x['Pos'].iloc[0])

            if sos_type is not None and agg_type == AGG_TYPE_PLAYER_FIRST:

                def get_sos_ranks(opp_team):
                    rx = sos_fetcher(sos_type, game['Year'], game['Date'], opp_team)
                    return rx
                x['Ranks'] = [get_sos_ranks(t) for t in x['OpponentTeamId']]

                def get_sos_values(g):
                    mv = {'n': n_games, 'v': g['Value'].sum() / n_games, 'a': {}}
                    for d in damping_factors:
                        dv = np.dot(g['Ranks'].apply(lambda rk: rk[d]), g['Value']) / n_games
                        mv['a'][d] = dv
                    return mv
                return x.groupby(['Pos', 'Metric']).apply(get_sos_values)
            else:
                return x.groupby(['Pos', 'Metric'])['Value'].sum() / n_games

        res = res.reset_index().groupby(['TimeFrame', 'TeamId']).apply(sos_averages)


        # Unstacking is only necessary when there was more than 1 group above
        if not isinstance(res, pd.DataFrame):
            res = res.unstack(level=[-2, -1])

        # Add DVOA statistics, if applicable
        if agg_type == AGG_TYPE_TEAM_FIRST:
            res = pd.concat([res, get_game_dvoa_stats(dvoa_data, game)], axis=1)

        #if agg_type == nfl_feat.AGG_TYPE_PLAYER_FIRST and 'Michael Vick' in game_player_stats['Player'].unique():
            #print(res)

        return res
    return sos_aggregator
