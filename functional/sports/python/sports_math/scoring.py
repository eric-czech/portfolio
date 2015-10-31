__author__ = 'eczech'

import numpy as np
import pandas as pd
from scipy import stats


def ngamma_posterior(data, prior_data):
    """
    Returns Normal-Gamma posterior parameters using the given data and prior sample

    :param data: Observations
    :param prior_data: Prior observations used to construct empirical bayes prior
    :return:
    """

    # Initialize gamma params; this sets the mean of the gamma distribution
    # to the the estimated precision in the prior data sample (i.e. 1/variance)
    a0, b0 = 1, np.std(prior_data)**2

    # Initialize the normal params; this sets the mean of the mean distribution
    # to the estimated mean in the prior data sample
    l0, m0 = 1, np.mean(prior_data)

    # Extract sufficient statistics from observation sample
    n = len(data)
    x = np.mean(data)
    s = np.sum([(v - x)**2 for v in data])

    # Create and return the updated posterior params
    M0 = (l0*m0 + n*x) / (l0 + n)
    L0 = l0 + n
    A0 = a0 + n/2
    B0 = b0 + .5*s + (l0 * n * (x - m0)**2)/(2*(l0 + n))
    return M0, L0, A0, B0


def rngamma(m0, l0, a0, b0, size):
    """
    Draw random samples from normal gamma distribution
    :param m0: NG location parameter
    :param l0: NG lambda parameter
    :param a0: NG alpha parameter
    :param b0: NG beta parameter
    :param size: Number of samples to draw
    :return: DataFrame with #size rows and columns 'Mean' and 'Variance'
    """
    sample = []
    for i in range(size):
        r = np.random.gamma(shape=a0, scale=1/b0, size=1)[0]
        m = np.random.normal(loc=m0, scale=(1/(l0*r))**.5, size=1)[0]
        sample.append((m, 1/r))
    return pd.DataFrame(sample, columns=['Mean', 'Variance'])


def ngamma_marginal_mu_ppf(m0, l0, a0, b0, p):
    """
    Returns quantile sports_math for Normal-Gamma marginal mu distribution
    :param m0: NG location parameter
    :param l0: NG lambda parameter
    :param a0: NG alpha parameter
    :param b0: NG beta parameter
    :param p: List of percentiles to get quantiles for all in range [0, 1]
    :return: Series containing quantiles indexed by percentiles
    """
    s = (b0 / (l0 * a0)) ** .5
    t_dist = stats.t(2*a0, loc=m0, scale=s)
    return pd.Series(t_dist.ppf(p), index=p)


def get_normalized_scores(scores, linear=False, less_is_better=True, damping_factor=.15):
    def normalize(x):
        v = x['Score'].values

        # If there is only one value present, return normalized scores
        # equal to 1 over the number of teams
        if len(np.unique(v)) == 1:
            return pd.Series(1/len(v), index=x['OpponentTeamId'])

        if linear:
            # Scale range to [0, 1]
            v = (v - np.min(v)) / (np.max(v) - np.min(v))

            # Determine weights
            v = v / np.sum(v)
            if less_is_better:
                v = 1 - v
        else:
            # Standardize inputs for softmax
            v = (v - np.mean(v)) / np.std(v)
            if less_is_better:
                v = -v
            v_sum = np.sum(np.exp(v))
            v = np.exp(v) / v_sum
        return pd.Series(v, index=x['OpponentTeamId'])
    res = scores.groupby('TeamId').apply(normalize).unstack().fillna(0)
    res = damping_factor + (1 - damping_factor) * res
    res = res.apply(lambda x: x/x.sum(), axis=1)
    return res
