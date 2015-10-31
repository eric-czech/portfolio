__author__ = 'eczech'

import pandas as pd
import numpy as np
from nfl import storage as nfl_db
from nfl.fdb.const import *
import functools


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


# def get_averages(x):
#     n_games = len(x['OutcomeLink'].unique())
#
#     def get_game_aggregate(d):
#         metric = d['Metric'].iloc[0]
#         for m in [':Avg', ':Rate', ':Lg', ':YPA']:
#             if metric.endswith(m):
#                 return d['Value'].mean()
#         return d['Value'].sum()
#
#     # x = x.groupby(['OutcomeLink', 'Pos', 'Metric'])['Value'].sum()
#     x = x.groupby(['OutcomeLink', 'Pos', 'Metric']).apply(get_game_aggregate)
#     x.name = 'Value'
#     return x.reset_index().groupby(['Pos', 'Metric'])['Value'].sum() / n_games

# TeamId, OpponentTeamId, Date, Metric, Value


def swap_direction(matchup_stats):
    rename_func = lambda c: 'OpponentTeamId' if c == 'TeamId' else ('TeamId' if c == 'OpponentTeamId' else c)
    return matchup_stats.rename(columns=rename_func)


def get_second_degree_stats(game, team, games, player_stats, rosters, aggregator):
    """
    Returns statistics for the teams that played against the teams involved in the given game

    IMPORTANT: This expects that mirrored game records ARE given (ie there should be two
        game records per OutcomeLink)

    :param game: Game with two teams to get transitive statistics for
    :param games: Full set of known games
    :param player_stats: Player stats to be used to create team stats
    :param team_stat_calculator: Method used to roll individual player statistics up to the team level
    :return:
    """
    opp_teams = games[games['TeamId'] == team]['OpponentTeamId'].unique()
    player_stats = player_stats[['Year', 'Date', 'PlayerLink', 'TeamId', 'OpponentTeamId', 'OutcomeLink', 'Metric', 'Value']]

    mask = player_stats['TeamId'].isin(opp_teams)
    mask &= player_stats['OpponentTeamId'] == team
    player_stats = player_stats[mask].copy()

    player_stats['TimeFrame'] = player_stats['Date'].apply(get_time_frame, args=(game,))

    join_cols = ['Year', 'TeamId', 'PlayerLink']
    player_stats = pd.merge(player_stats, rosters[join_cols + ['Pos']], on=join_cols)
    player_stats = swap_direction(player_stats)
    player_stats = player_stats[['TimeFrame', 'TeamId', 'OpponentTeamId',
                                 'OutcomeLink', 'PlayerLink', 'Pos', 'Year', 'Date', 'Metric', 'Value']]

    # Group per-team, per-time-frame results by time frame only, and create mean values
    return aggregator(game, player_stats, agg_type=AGG_TYPE_TEAM_SECOND)


def get_game_context(game, games, player_stats, rosters, year, aggregator):
    opponent_id = game['OpponentTeamId']
    home_team_id = opponent_id if '@' in game['Opponent'] else game['TeamId']
    away_team_id = game['TeamId'] if '@' in game['Opponent'] else opponent_id

    # For now limit game and statistics data to only the past and current time frames,
    # for the sake of performance (future calculations are slow and irrelevant)
    games = games[games['Date'] <= game['Date']]
    player_stats = player_stats[player_stats['Date'] <= game['Date']].drop('PlayerName', axis=1)

    players = rosters[(rosters['TeamId'].isin([home_team_id, away_team_id])) & (rosters['Year'] == year)]

    game_player_stats = pd.merge(players, player_stats, on=['Year', 'TeamId', 'PlayerLink'])
    game_player_stats['TimeFrame'] = game_player_stats['Date'].apply(get_time_frame, args=(game,))

    game_team_1st_stats = aggregator(game, game_player_stats, agg_type=AGG_TYPE_TEAM_FIRST)
    game_team_1st_stats.columns = [(c[0], c[1]+':1st') for c in game_team_1st_stats]

    p1 = get_second_degree_stats(game, home_team_id, games, player_stats, rosters, aggregator=aggregator)
    p2 = get_second_degree_stats(game, away_team_id, games, player_stats, rosters, aggregator=aggregator)


    try:
        game_team_2nd_stats = pd.DataFrame.append(
            pd.DataFrame() if len(p1) == 0 else p1.reset_index(),
            pd.DataFrame() if len(p2) == 0 else p2.reset_index()
        ).set_index(p1.index.names if len(p1) > 0 else p2.index.names)
    except:
        raise

    game_team_2nd_stats.columns = [(c[0], c[1]+':2nd') for c in game_team_2nd_stats]
    game_team_stats = pd.concat([game_team_1st_stats, game_team_2nd_stats], axis=1)

    return game_team_stats, game_player_stats, players, home_team_id, away_team_id


def get_team_agg_stats(game, game_team_stats, team_id, col_prefix):
    if ('Past', team_id) not in game_team_stats.index:
        raise ValueError('Failed to find past stats for team {} for game {}'.format(team_id, game))
    stats = game_team_stats.loc[('Past', team_id)]
    assert isinstance(stats, pd.Series), 'Opposition stats not found as series'
    stats.index = pd.MultiIndex.from_tuples([(col_prefix, c[0]+':'+c[1]) for c in stats.index])
    return stats


def get_matchup_matrix(game, game_team_stats, game_player_stats, game_roster, home_team_id, away_team_id, aggregator):
    g_data = game.copy()
    g_data.index = pd.MultiIndex.from_tuples([('Attr:Game', c) for c in g_data.index])
    rows = []

    # .sort_index is important for performance (a "Lex sort depth" warning is thrown otherwise)
    game_player_stats = game_player_stats.set_index(['PlayerLink', 'TeamId', 'TimeFrame']).sort_index()

    for i, player in game_roster.iterrows():
        player_team_id = player['TeamId']
        player['IsHome'] = player_team_id == home_team_id
        opp_team_id = away_team_id if player['IsHome'] else home_team_id
        player.index = pd.MultiIndex.from_tuples([('Attr:Player', c) for c in player.index])


        # if ('Past', opp_team_id) not in game_team_stats.index:
        #     print('Failed to find past stats for opposing team {} for game {}'.format(opp_team_id, game['OutcomeLink']))
        #     continue
        # opp_stats = game_team_stats.loc[('Past', opp_team_id)]
        # assert isinstance(opp_stats, pd.Series), 'Opposition stats not found as series'
        # opp_stats.index = pd.MultiIndex.from_tuples([('Stat:Opp', c[0]+':'+c[1]) for c in opp_stats.index])
        # res = player.append(opp_stats)

        res = player\
            .append(get_team_agg_stats(game, game_team_stats, player_team_id, 'Stat:Team'))\
            .append(get_team_agg_stats(game, game_team_stats, opp_team_id, 'Stat:Opp'))

        res = res.append(g_data)

        player_key = [player[('Attr:Player', 'PlayerLink')], player[('Attr:Player', 'TeamId')]]

        # Fetch this particular player's past performance so far this season
        key = tuple(player_key + ['Past'])
        if key in game_player_stats.index:
            game_stats = game_player_stats.loc[key]
            # Determine the number of games in the current season for which this player has
            # made it onto the box score, in one way or another
            n_contrib = len(game_stats[game_stats['Year'] == game['Year']]['Date'].unique())
            game_stats = aggregator(game, game_stats.reset_index(), agg_type=AGG_TYPE_PLAYER_FIRST)

            # game_stats now has columns like ('Pos', 'Metric')
            idx = pd.MultiIndex.from_tuples([('Stat:Player', m[1]) for m in game_stats])
            res = res.append(pd.Series(game_stats.iloc[0, :].values, index=idx))


            res[('Stat:Player', 'ContribG')] = n_contrib

        # Fetch this particular player's actual performance for this game
        key = tuple(player_key + ['Current'])
        if key in game_player_stats.index:
            game_stats = game_player_stats.loc[key]
            idx = pd.MultiIndex.from_tuples([('Stat:Result', m) for m in game_stats['Metric']])
            res = res.append(pd.Series(game_stats['Value'].values, index=idx))

        # Verify that no stats added to result were added twice (there was originally an edge
        # case with a player that played for 2 teams in the same season and in a game
        # where those teams where playing each other, that makes this worth doing
        if len(pd.Series(res.index.values).value_counts().value_counts()) > 1:
            print('Found player sports_math set with repeated keys:')
            print(pd.Series(res.index.values).value_counts())
            raise ValueError('Found player sports_math set with repeated keys (game = {})'.format(game))
        rows.append(res)
    return pd.DataFrame(rows)



def clean_player_stats(stats):
    """
    Attempts to convert string values like '30t' or '23,38' into single floating point
    numbers based on their associated metric (and its implicit interpretation)
    :param stats: DataFrame from #get_player_stats
    :return: Same data frame as given with 'Value' converted to float
    """
    stats['Value'] = [convert_number(t) for t in stats[['Metric', 'Value']].values]
    return stats


def get_player_stats(games, years):
    stat_cols = [c for c in nfl_db.get_collections(DB) if c.startswith(COLL_STATS) and 'team_stats' not in c]
    parts = []
    for stat_col in stat_cols:
        stats = nfl_db.run_query(DB, stat_col, years)
        if len(stats) == 0:
            continue
        id_cols = ['OutcomeLink', 'RowLink', 'RowName', 'Year']
        stats = stats.rename(columns=lambda c: c if c in id_cols else 'stat::'+c)
        stat_type = stat_col.replace(COLL_STATS+'_', '')

        game_stats = pd.merge(games, stats, on=['OutcomeLink', 'Year'])
        id_cols = [c for c in game_stats if not c.startswith('stat::')]
        game_stats = game_stats.set_index(id_cols)
        game_stats = game_stats.rename(columns=lambda c: c.replace('stat::', ''))
        game_stats.columns.name = 'Metric'
        game_stats = game_stats.rename(columns=lambda x: stat_type + ':' + x).stack()
        game_stats.name = 'Value'
        parts.append(game_stats.reset_index())
    res = functools.reduce(pd.DataFrame.append, parts)
    res = res.rename(columns={'RowLink': 'PlayerLink', 'RowName': 'PlayerName'})
    return clean_player_stats(res)


def get_games(years, remove_mirrored_records=True, query={}):
    """
    Returns game records for the given years
    :param years: List, Range, or Scalar year filter
    :param remove_mirrored_records: All game records are stored twice, once for each team
        involved in the game.  By default, this flag is on which means only one of those
        records will be returned (arbitrarily).
    :return:
    """
    def get_game_number(x):
        x = x.sort('Date')
        x['GameNumber'] = np.arange(len(x)) + 1
        return x
    games = nfl_db.run_query(DB, COLL_GAMES, years, other=query)

    # Ignore records with a NULL box score link, but report how
    # many such records are being deleted
    null_mask = games['OutcomeLink'].isnull()
    if null_mask.sum() > 0:
        null_counts = games[null_mask].groupby(['Year', 'GameType']).size()
        print('Removing {} game records of {} due to "OutcomeLink"'
              'being null.  Null record counts by year:\n{}'.format(null_mask.sum(), len(games), null_counts))
        games = games[~null_mask]

    # Select at most one record per outcome link
    if remove_mirrored_records:
        games = games.groupby('OutcomeLink').apply(lambda x: x.head(1)).reset_index(drop=True)

    games = games.groupby(['TeamId', 'Year', 'GameType'], group_keys=False).apply(get_game_number)
    games['OpponentTeamId'] = games['OpponentLink'].str.split('/').str.get(3)
    return games


def get_rosters(year):
    """
    Returns rosters for the given year, after adding cumulative game counts to it for each player
    :param year: Year to fetch roster data for
    """
    rosters = nfl_db.get_data(DB, COLL_ROSTR, {'Year': {'$lte': year}})

    # Pull out past roster data and use it to create season-specific
    # summaries of games played and games started
    cum_numbers = rosters[rosters['Year'] < year].copy()
    cum_numbers['TimeRange'] = year - rosters['Year']
    cum_numbers['TimeRange'] = cum_numbers['TimeRange']\
        .where(cum_numbers['TimeRange'] <= 3, '4+')\
        .apply(lambda x: '{}{}'.format(x, 'yr'))
    cum_numbers = cum_numbers.groupby(['Player', 'PlayerLink', 'TimeRange'])[['G', 'GS']].sum().unstack()
    number_cols = [c[0]+':'+c[1] for c in cum_numbers]
    cum_numbers.columns = number_cols
    cum_numbers = cum_numbers.reset_index()

    # Join seasonal counts to main dataset and return result with nan's coalesced to 0
    rosters = rosters[rosters['Year'] == year].rename(columns={'G': 'G:0yr', 'GS': 'GS:0yr'})
    res = pd.merge(rosters, cum_numbers, on=['Player', 'PlayerLink'], how='left')
    res[number_cols] = res[number_cols].fillna(0)
    return res


def get_airport_weather():
    weather = nfl_db.get_data(DB, COLL_WEATHER_WU)
    if weather['Temp'].isnull().sum() > 0:
        print('Encountered missing temperatures for the following airports '
              '(they will be replaced with national averages):')
        print(weather[weather['Temp'].isnull()]['AirportCode'].value_counts())
        temp_means = weather.groupby('Date').apply(lambda x: x['Temp'].dropna().mean())
        weather['Temp'] = weather\
            .apply(lambda x: x['Temp'] if not pd.isnull(x['Temp']) else temp_means[x['Date']], axis=1)
    return weather


def get_stadium_weather():
    stadiums = nfl_db.get_data(DB, COLL_STADIUM)
    weather = get_airport_weather()

    stadium_weather = pd.merge(stadiums[['AirportCode', 'LocationString']], weather, on='AirportCode', how='inner')
    stadium_weather = stadium_weather.drop('AirportCode', axis=1).rename(columns=lambda c: 'Attr:Weather:'+c)
    return stadium_weather
