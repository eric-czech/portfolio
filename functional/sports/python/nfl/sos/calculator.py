__author__ = 'eczech'

from sports_math import scoring as sm_scoring
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from nfl.sos.profile import PROFILE_PASSING_DEFENSE


def preprocess_scores(data, date, weights, standardize):
    """
    Creates a single weighted score value for pairwise matchups
    :param data: DataFrame with at least 'Date' and columns by names
        specified in keys of #weights
    :param weights: Dict with keys equal to statistics columns to be combined
        and values equal to weight assigned to each (weights must sum to 1)
    :return DataFrame identical to original with all statistics columns removed
        and instead replaced by a single 'Value' column

    Example:

    input data:
    Year	Date	TeamId	OpponentTeamId	passing:Rate	passing:YPA	passing:Yds
    0	1996	1996-09-01	arizona-cardinals	indianapolis-colts	82.900000	6.200000	237
    1	1996	1996-09-01	indianapolis-colts	arizona-cardinals	94.148485	7.509091	249
    2	1996	1996-09-01	atlanta-falcons	carolina-panthers	65.800000	6.100000	215
    3	1996	1996-09-01	carolina-panthers	atlanta-falcons	95.900000	6.400000	198
    4	1996	1996-09-01	cincinnati-bengals	st-louis-rams	61.000000	5.700000	226

    output data: Same as above with all passing:* columns replace by 'Value'
    """
    # Limit to only data BEFORE the target date
    d = data[data['Date'] < date].copy()

    if len(d) == 0:
        return pd.DataFrame([], columns=['Year', 'Value'])

    # Scale all statistics so that weighted sum is valid
    scaler = StandardScaler()
    stat_cols = list(weights.keys())
    if standardize:
        d[stat_cols] = d[stat_cols].apply(scaler.fit_transform)

    # Create a single weighted value from all statistics
    def get_weighted_value(x):
        return np.sum(x[k]*w for k, w in weights.items())
    v = d[stat_cols].apply(get_weighted_value, axis=1)
    d = d.drop(stat_cols, axis=1)
    d['Value'] = v

    return d


def get_scores(data, year, date, profile, season_threshold=3, standardize=True):
    d = preprocess_scores(data, date, profile['weights'], standardize)
    prior_d = d[d['Year'] < year - season_threshold]
    curr_d = d[d['Year'].between(year - season_threshold, year)]
    #print(len(curr_d), len(prior_d))

    # If no games have occurred this season prior to date, return
    # a score of 0 for each team that played on date

    if len(curr_d) == 0:
        res = data[(data['Year'] == year) & (data['Date'] == date)].copy()
        res['Score'] = 0
        res = res.set_index(['TeamId', 'OpponentTeamId'])
        res.name = 'Score'
        return res

    def get_team_scores(x):
        def get_opp_scores(y):
            # Just override for now
            return y['Value'].mean()

            raw_v = y['Value']
            w = np.array([year - y_past for y_past in y['Year']])
            w = np.max(w) - w + 1
            rep_v = []
            for i in range(len(raw_v)):
                #rep_v.extend(np.repeat(raw_v.iloc[i], 1))
                rep_v.extend(np.repeat(raw_v.iloc[i], w[i]*10))
            #print(y, rep_v)
            prior_vs = prior_d[prior_d['TeamId'] == y['TeamId'].iloc[0]]
            prior_vs = prior_vs[prior_vs['OpponentTeamId'] == y['OpponentTeamId'].iloc[0]]['Value']
            if len(prior_vs) == 0:
                prior_vs = raw_v
            #print(len(prior_vs['Value']))
            M0, L0, A0, B0 = sm_scoring.ngamma_posterior(rep_v, prior_vs)
            score = sm_scoring.ngamma_marginal_mu_ppf(M0, L0, A0, B0, [.25]).iloc[0]
            assert not np.isnan(score), 'Nan score created for raw input values: {}, prior values: {}'\
                .format(raw_v.values, prior_vs.values)
            return score
        return x.groupby('OpponentTeamId').apply(get_opp_scores)
    res = curr_d.groupby('TeamId').apply(get_team_scores)
    res.name = 'Score'
    return res





def get_matchup_statistics(data, profile=PROFILE_PASSING_DEFENSE):
    stats = data[data['Metric'].isin(profile['input_fields'])]
    id_cols = ['Year', 'Date', 'GameType', 'OutcomeLink', 'TeamId', 'OpponentTeamId', 'PlayerLink', 'Metric']
    stats = stats.set_index(id_cols)['Value'].unstack()
    stats.index = stats.index.droplevel('PlayerLink')

    def calc_stats(x):
        return pd.Series({k: f(x) for k, f in profile['output_fields'].items()})
    stats = stats.reset_index().groupby(stats.index.names).apply(calc_stats)
    return stats