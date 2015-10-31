__author__ = 'eczech'


def get_sos_type(pos):
    if pos == 'QB':
        return 'passing:defense'
    return None


def get_sos_aggregator(year):
    sos, sos_def = sos_db.get_sos_data([year], damping=.5)
    def sos_aggregator(game, game_player_stats, agg_type):
        # Sum by position and game
        res = game_player_stats.groupby([
                'TimeFrame', 'TeamId', 'OpponentTeamId', 'OutcomeLink',
                'Year', 'Date', 'Pos', 'Metric'])['Value'].sum()
        res.name = 'Value'

        def sos_averages(x):
            n_games = len(x['OutcomeLink'].unique())
            sos_type = get_sos_type(x['Pos'].iloc[0])

            if sos_type is not None and agg_type == nfl_feat.AGG_TYPE_PLAYER_FIRST:
                def get_sos_ranks(r):
                    key = (sos_type, game['Year'], game['Date'], r['OpponentTeamId'])

                    if key not in sos.index:
                        return sos_def.loc[(sos_type, game['Year'])]
                    else:
                        return sos.loc[key]
                x['Rank'] = x.apply(get_sos_ranks, axis=1)
                def get_sos_values(x):
                    return {
                        #'v': x['Value'].sum() / n_games,
                        'v': list(x['Value'].values),
                        'r': list(x['Rank'].values),
                        'n': n_games,
                        'a': np.dot(x['Rank'], x['Value']) / n_games
                    }
                r = x.groupby(['Pos', 'Metric']).apply(get_sos_values)
                return r
            return x.groupby(['Pos', 'Metric'])['Value'].sum() / n_games

        res = res.reset_index().groupby(['TimeFrame', 'TeamId']).apply(sos_averages)


        # Unstacking is only necessary when there was more than 1 group above
        if not isinstance(res, pd.DataFrame):
            res = res.unstack(level=[-2, -1])

        #if agg_type == nfl_feat.AGG_TYPE_PLAYER_FIRST and 'Michael Vick' in game_player_stats['Player'].unique():
            #print(res)

        return res
    return sos_aggregator


