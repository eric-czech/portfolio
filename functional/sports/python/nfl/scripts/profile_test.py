from nfl.fdb.const import *
from nfl.fdb import features as nfl_feat
from nfl import storage as nfl_db
import pandas as pd
import numpy as np



def get_time_frame(x, game):
    if x < game['Date']:
        return 'Past'
    if x > game['Date']:
        return 'Future'
    return 'Current'


def convert_number(values):
    metric, x = values
    if metric == 'kicking:Miss':
        return len(str(x).split(','))
    if not isinstance(x, str):
        return x
    if x.endswith('t'):
        return float(x.replace('t', ''))
    if len(x.split('/')) == 2:
        return int(x.split('/')[0])
    if x == '--':
        return 0
    return float(x)


def get_averages(x):
    n_games = len(x['OutcomeLink'].unique())
    x = x.groupby(['OutcomeLink', 'Pos', 'Metric'])['Value'].sum()
    x.name = 'Value'
    return x.reset_index().groupby(['Pos', 'Metric'])['Value'].sum() / n_games


def get_game_context(game, player_stats, rosters, year):
    opponent_id = game['OpponentLink'].split('/')[3]
    home_team_id = opponent_id if '@' in game['Opponent'] else game['TeamId']
    away_team_id = game['TeamId'] if '@' in game['Opponent'] else opponent_id

    players = rosters[(rosters['TeamId'].isin([home_team_id, away_team_id])) & (rosters['Year'] == year)]

    game_player_stats = pd.merge(players, player_stats.drop('PlayerName', axis=1), on=['Year', 'TeamId', 'PlayerLink'])

    def time_frame(x):
        return get_time_frame(x, game)
    game_player_stats['TimeFrame'] = game_player_stats['Date'].apply(time_frame)

    game_player_stats['Value'] = [convert_number(t) for t in game_player_stats[['Metric', 'Value']].values]

    game_team_stats = game_player_stats.groupby(['TimeFrame', 'TeamId', 'OutcomeLink', 'Pos', 'Metric'])['Value'].sum()
    game_team_stats.name = 'Value'

    game_team_stats = game_team_stats.reset_index().groupby(['TimeFrame', 'TeamId']).apply(get_averages)
    game_team_stats = game_team_stats.unstack(level=[-2, -1])

    return game_team_stats, game_player_stats, players, home_team_id, away_team_id


def get_matchup_matrix(game, game_team_stats, game_player_stats, game_roster, home_team_id, away_team_id):
    g_data = game.copy()
    g_data.index = pd.MultiIndex.from_tuples([('Attr:Game', c) for c in g_data.index])
    rows = []

    # .sort_index is important for performance (a "Lex sort depth" warning is thrown otherwise)
    game_player_stats = game_player_stats.set_index(['PlayerLink', 'TeamId', 'TimeFrame']).sort_index()

    for i, player in game_roster.iterrows():
        player['IsHome'] = player['TeamId'] == home_team_id
        opp_team_id = away_team_id if player['IsHome'] else home_team_id
        player.index = pd.MultiIndex.from_tuples([('Attr:Player', c) for c in player.index])

        if ('Past', opp_team_id) not in game_team_stats.index:
            continue
        opp_stats = game_team_stats.loc[('Past', opp_team_id)]
        assert isinstance(opp_stats, pd.Series), 'Opposition stats not found as series'
        opp_stats.index = pd.MultiIndex.from_tuples([('Stat:Opp', c[0]+':'+c[1]) for c in opp_stats.index])

        res = player.append(opp_stats)
        res = res.append(g_data)

        # Fetch this particular player's past performance so far this season
        key = (player[('Attr:Player', 'PlayerLink')], player[('Attr:Player', 'TeamId')], 'Past')
        if key in game_player_stats.index:
            game_stats = game_player_stats.loc[key]
            game_stats = game_stats.groupby('Metric')['Value'].sum() / game['GameNumber']
            game_stats = game_stats.reset_index()
            idx = pd.MultiIndex.from_tuples([('Stat:Player', m) for m in game_stats['Metric']])
            res = res.append(pd.Series(game_stats['Value'].values, index=idx))

        # Fetch this particular player's actual performance for this game
        key = (player[('Attr:Player', 'PlayerLink')], player[('Attr:Player', 'TeamId')], 'Current')
        if key in game_player_stats.index:
            game_stats = game_player_stats.loc[key]
            idx = pd.MultiIndex.from_tuples([('Stat:Result', m) for m in game_stats['Metric']])
            res = res.append(pd.Series(game_stats['Value'].values, index=idx))

        if len(pd.Series(res.index.values).value_counts().value_counts()) > 1:
            print('Found player sports_math set with repeated keys:')
            print(pd.Series(res.index.values).value_counts())
            raise ValueError('Found player sports_math set with repeated keys (game = {})'.format(game))
        rows.append(res)
    return pd.DataFrame(rows)


def get_player_stats(games, year):
    stat_cols = [c for c in nfl_db.get_collections(DB) if c.startswith(COLL_STATS) and 'team_stats' not in c]
    parts = []
    for stat_col in stat_cols:
        stats = nfl_db.get_data(DB, stat_col, {'Year': year})
        if len(stats) == 0:
            continue
        stat_type = stat_col.replace(COLL_STATS+'_', '')
        id_cols = ['TeamId', 'GameType', 'OutcomeLink', 'RowLink', 'RowName', 'Date', 'Year']
        game_stats = pd.merge(games[['TeamId', 'OutcomeLink', 'GameType', 'Date']], stats, on=['OutcomeLink'])
        game_stats = game_stats.set_index(id_cols)
        game_stats.columns.name = 'Metric'
        game_stats = game_stats.rename(columns=lambda x: stat_type + ':' + x).stack()
        game_stats.name = 'Value'
        parts.append(game_stats.reset_index())
    res = functools.reduce(pd.DataFrame.append, parts)
    return res.rename(columns={'RowLink': 'PlayerLink', 'RowName': 'PlayerName'})


def get_games(year):
    def get_game_number(x):
        x = x.sort('Date')
        x['GameNumber'] = np.arange(len(x)) + 1
        return x
    games = nfl_db.get_data(DB, COLL_GAMES, {'Year': year})
    return games.groupby(['TeamId', 'Year', 'GameType'], group_keys=False).apply(get_game_number)


import functools
def get_game_matrix(games, player_stats, rosters, year, min_game_num=6):
    game_matrix = []    
    training_games = games[(games['GameType'] == 'Regular Season') & (games['GameNumber'] >= min_game_num)]

    n_games = len(training_games)
    i = 0; game_links = set()
    for _, game in training_games.iterrows():
        i += 1
        if i % 10 == 0:
            print('\tProcessing game {} of {}'.format(i, n_games))

        # Each game appears twice, once for each team involved, so duplicates
        # should not be reprocessed here 
        if game['OutcomeLink'] in game_links:
            continue
        game_links.add(game['OutcomeLink'])

        #parts = nfl_feat.get_game_context(game, player_stats, rosters, year)
        parts = get_game_context(game, player_stats, rosters, year)
        #game_team_stats, game_player_stats, game_roster, home_team_id, away_team_id = parts
        if i > 20:
            break
        part = nfl_feat.get_matchup_matrix(game, *parts)
        #print(len(part))
        game_matrix.append(part)
    return functools.reduce(pd.DataFrame.append, game_matrix)


year = 2013
rosters = nfl_db.get_data(DB, COLL_ROSTR, {'Year': year})
games = nfl_feat.get_games(year)
games = games[(~games['OutcomeLink'].isnull()) & (games['GameType'] == 'Regular Season')]
player_stats = nfl_feat.get_player_stats(games, year)
game_matrix = get_game_matrix(games, player_stats, rosters, year)
training_games = games[(games['GameType'] == 'Regular Season') & (games['GameNumber'] >= 6)]
game = training_games.iloc[0]
get_game_context(game, player_stats, rosters, year)
