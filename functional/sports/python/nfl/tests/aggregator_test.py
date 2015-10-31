__author__ = 'eczech'

from nfl.fdb import aggregator
from nfl.fdb import const
import pandas as pd


def test_aggregator():
    cols = [
        'TimeFrame', 'TeamId', 'OpponentTeamId', 'OutcomeLink',
        'Year', 'Date', 'Pos', 'Metric', 'Value', 'PlayerLink'
    ]
    date1 = pd.to_datetime('2014-09-01')
    game_player_stats = pd.DataFrame([
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'passing:YPA', 10, 'player1'),
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'passing:YPA', 20, 'player2'),
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'passing:Att', 1, 'player1'),
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'passing:Att', 3, 'player2'),

        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'passing:Yds', 100, 'player1'),
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'passing:Yds', 50, 'player2'),

        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'rushing:Avg', 10, 'player1'),
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'rushing:Avg', 20, 'player2'),
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'rushing:Att', 1, 'player1'),
        ('Past', 'arizona', 'seattle', 'link1', 2014, date1, 'QB', 'rushing:Att', 3, 'player2'),
    ], columns=cols)
    res = aggregator.get_game_aggregate(game_player_stats, const.AGG_TYPE_TEAM_FIRST)
    print(res)
    # Expected:
    # TimeFrame  TeamId   OpponentTeamId  OutcomeLink  Year  Date        Pos  Metric
    # Past       arizona  seattle         link1        2014  2014-09-01  QB   passing:Att      4.0
    #                                                                         passing:YPA     17.5
    #                                                                         passing:Yds    150.0
    #                                                                         rushing:Att      4.0
    #                                                                         rushing:Avg     17.5

    res = aggregator.get_game_aggregate(game_player_stats, const.AGG_TYPE_PLAYER_FIRST)
    print(res)
    # Expected:
    # TimeFrame  TeamId   OpponentTeamId  OutcomeLink  Year  Date        Pos  Metric
    # Past       arizona  seattle         link1        2014  2014-09-01  QB   passing:Att      4
    #                                                                         passing:YPA     30
    #                                                                         passing:Yds    150
    #                                                                         rushing:Att      4
    #                                                                         rushing:Avg     30
test_aggregator()