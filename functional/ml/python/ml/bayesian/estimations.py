
import numpy as np
from scipy import stats


def get_normal_gamma_variance_posteriors(x, u0, a0, b0):
    """
    Get posterior predictive distribution for normal samples with known mean and unknown variance.

    See https://en.wikipedia.org/wiki/Conjugate_prior section on "Normal with known mean u"

    :param x: Sample
    :param u0: Known distribution mean
    :param a0: Hyperparameter for alpha of precision distribution (shape)
    :param b0: Hyperparameter for beta of precision distribution (rate, not scale)
    :return: Gamma Distribution (for precision), T Distribution (posterior predictive for samples)
    """
    x = np.array(x, dtype=np.float64)
    n = len(x)
    a = a0 + n/2.
    b = b0 + (np.sum((x - u0)**2)/2.)

    # Beta for gamma is a rate up until this point but scipy gamma distribution only accepts scale (ie 1/rate)
    precision_dist = stats.gamma(a, scale=1/b)
    # print(a, b, u0, b/a, (np.sum((x - u0)**2))/2.)
    predictive_dist = stats.t(df=2*a, loc=u0, scale=b/a)
    return precision_dist, predictive_dist


def get_normal_gamma_posterior_predictive(x, u0, k0, a0, b0):
    """
    Get posterior predictive for vector sample with unknown mean and variance.

    The returned Student-t distribution for the predictive posterior can be seen as derived in [1], [2], and [3]
    1: https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/reading/NG.pdf (page 5)
    2: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf (page 9)
    3: https://en.wikipedia.org/wiki/Conjugate_prior (Normal-gamma conjugate prior section)

    :param x: Sample to estimate distribution for (any non-zero length sequence convertible to float)
    :param u0: Hyperparameter for mean of mean distribution
    :param k0: Hyperparameter for inverse variance of mean distribution
    :param a0: Hyperparameter for alpha of precision distribution (shape)
    :param b0: Hyperparameter for beta of precision distribution (rate, not scale)
    :return: T-Distribution (posterior predictive for samples)
        *Note that posterior on parameters is not returned here simply because python has no normal-gamma implementation
    """
    x = np.array(x, dtype=np.float64)
    n = len(x)
    x_bar = np.mean(x)
    u = (k0 * u0 + n * x_bar) / (k0 + n)
    k = k0 + n
    a = a0 + n / 2.
    b = b0 + .5 * np.sum((x - x_bar)**2) + (k0 * n * (x_bar - u0)**2) / (2 * (k0 + n))
    # print(u, k, a, b, (b * (k + 1))/(a * k))
    predictive_dist = stats.t(df=2*a, loc=u, scale=(b * (k + 1))/(a * k))
    return predictive_dist


def get_dirichlet_mle(x):
    """
    Get maximum likelihood estimation for dirichlet concentration parameters

    :param x: Data array of size nxp where n is the number of observations and p is the dimensionality of the
        dirichlet distribution to estimate
    :return: Dirichlet distribution (with `alpha` parameter set to MLE inferred from given data)
    """
    assert len(x.shape) == 2, 'Data array must be two dimensional'
    assert np.allclose(x.sum(axis=1), 1), 'Sum of observations across rows must equal 1'
    from ml.bayesian.dirichlet import dirichlet
    alpha = dirichlet.mle(x)
    return stats.dirichlet(alpha)


def get_dirichlet_multinomial_posterior(x, alpha):
    """
    Return posterior dirichlet distribution with dirichlet prior and multinomial likelihood

    Posterior for dirichlet distribution with parameters equal to (alpha1 + x1, alpha2 + x2, ..., alphap + xp)
    where p is the number of categories (i.e. number of columns in x)

    :param x: Integer array where index indicates category (must have length > 1)
    :param alpha: Concentration parameter for dirichlet prior; can be scalar or array of size len(x)
    :return: Dirichlet distribution
    """
    assert len(x) > 1, 'Number of categories for dirichlet/multinomial model must be > 1'
    return stats.dirichlet(x + alpha)
