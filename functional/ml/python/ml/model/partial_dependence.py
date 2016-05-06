

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.utils import check_array
from sklearn.tree._tree import DTYPE
import numbers

from sklearn.ensemble.gradient_boosting import BaseGradientBoosting


def get_partial_dependence(gbrt, X, features, feature_names=None,
                            label=None, n_cols=3, grid_resolution=100,
                            percentiles=(0.05, 0.95), n_jobs=1,
                            verbose=0, ax=None, line_kw=None,
                            contour_kw=None, **fig_kw):
    """Partial dependence data for ``features``.

    Parameters
    ----------
    gbrt : BaseGradientBoosting
        A fitted gradient boosting model.
    X : array-like, shape=(n_samples, n_features)
        The data on which ``gbrt`` was trained.
    features : seq of tuples or ints
        If seq[i] is an int or a tuple with one int value, a one-way
        PDP is created; if seq[i] is a tuple of two ints, a two-way
        PDP is created.
    feature_names : seq of str
        Name of each feature; feature_names[i] holds
        the name of the feature with index i.
    label : object
        The class label for which the PDPs should be computed.
        Only if gbrt is a multi-class model. Must be in ``gbrt.classes_``.
    n_cols : int
        The number of columns in the grid plot (default: 3).
    percentiles : (low, high), default=(0.05, 0.95)
        The lower and upper percentile used create the extreme values
        for the PDP axes.
    grid_resolution : int, default=100
        The number of equally spaced points on the axes.
    n_jobs : int
        The number of CPUs to use to compute the PDs. -1 means 'all CPUs'.
        Defaults to 1.
    verbose : int
        Verbose output during PD computations. Defaults to 0.
    ax : Matplotlib axis object, default None
        An axis object onto which the plots will be drawn.
    line_kw : dict
        Dict with keywords passed to the ``pylab.plot`` call.
        For one-way partial dependence plots.
    contour_kw : dict
        Dict with keywords passed to the ``pylab.plot`` call.
        For two-way partial dependence plots.
    fig_kw : dict
        Dict with keywords passed to the figure() call.
        Note that all keywords not recognized above will be automatically
        included here.

    Returns
    -------
    fig : figure
        The Matplotlib Figure object.
    axs : seq of Axis objects
        A seq of Axis objects, one for each subplot.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> X, y = make_friedman1()
    >>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
    >>> fig, axs = plot_partial_dependence(clf, X, [0, (0, 1)]) #doctest: +SKIP
    ...
    """

    if not isinstance(gbrt, BaseGradientBoosting):
        raise ValueError('gbrt has to be an instance of BaseGradientBoosting')
    if gbrt.estimators_.shape[0] == 0:
        raise ValueError('Call %s.fit before partial_dependence' %
                         gbrt.__class__.__name__)

    X = check_array(X, dtype=DTYPE, order='C')
    if gbrt.n_features != X.shape[1]:
        raise ValueError('X.shape[1] does not match gbrt.n_features')

    # convert feature_names to list
    if feature_names is None:
        # if not feature_names use fx indices as name
        feature_names = [str(i) for i in range(gbrt.n_features)]
    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    def convert_feature(fx):
        if isinstance(fx, six.string_types):
            try:
                fx = feature_names.index(fx)
            except ValueError:
                raise ValueError('Feature %s not in feature_names' % fx)
        return fx

    # convert features into a seq of int tuples
    tmp_features = []
    for fxs in features:
        if isinstance(fxs, (numbers.Integral,) + six.string_types):
            fxs = (fxs,)
        try:
            fxs = np.array([convert_feature(fx) for fx in fxs], dtype=np.int32)
        except TypeError:
            raise ValueError('features must be either int, str, or tuple '
                             'of int/str')
        if not (1 <= np.size(fxs) <= 2):
            raise ValueError('target features must be either one or two')

        tmp_features.append(fxs)

    features = tmp_features

    names = []
    try:
        for fxs in features:
            l = []
            # explicit loop so "i" is bound for exception below
            for i in fxs:
                l.append(feature_names[i])
            names.append(l)
    except IndexError:
        raise ValueError('features[i] must be in [0, n_features) '
                         'but was %d' % i)

    # compute PD functions
    pd_result = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(partial_dependence)(gbrt, fxs, X=X,
                                    grid_resolution=grid_resolution, percentiles=percentiles)
        for fxs in features)

    return pd_result
