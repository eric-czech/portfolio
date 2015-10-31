__author__ = 'eczech'

DB = 'nfl'
COLL_PREFIX = 'fdb'
COLL_TEAMS = COLL_PREFIX + '_teams'
COLL_GAMES = COLL_PREFIX + '_games'
COLL_STATS = COLL_PREFIX + '_stats'
COLL_ROSTR = COLL_PREFIX + '_roster'
COLL_STADIUM = COLL_PREFIX + '_stadium'

COLL_TRAIN_V1 = 'dat_train_v1'

COLL_SOS_V1 = 'sos_v1'
#COLL_SOS_V1 = 'sos_v2'

COLL_WEATHER_WU = 'weather_wu_apt_code'


AGG_TYPE_TEAM_FIRST = 'team:first'
AGG_TYPE_TEAM_SECOND = 'team:second'
AGG_TYPE_PLAYER_FIRST = 'player:first'