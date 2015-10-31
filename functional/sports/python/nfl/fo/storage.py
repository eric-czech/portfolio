__author__ = 'eczech'


from nfl import storage as nfl_db
from nfl.fo import const as fo_const


def get_team_dvoa():
    team_dvoa = nfl_db.get_data(fo_const.DB, fo_const.COLL_TEAM_DVOA)
    dvoa_cols = [c for c in team_dvoa if c.endswith('DVOA')]
    team_dvoa[dvoa_cols] = team_dvoa[dvoa_cols].applymap(lambda x: float(x.replace('%', '')))
    return team_dvoa.set_index(['Year', 'Week', 'Team', 'Opponent'])


def get_derived_game_dvoa(years, query=None):
    res = nfl_db.run_query(fo_const.DB, fo_const.COLL_DERIVED_DVOA, years, query)
    return res.set_index(['Year', 'Date', 'TeamId', 'OpponentTeamId']).drop('Week', axis=1)

