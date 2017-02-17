
import numpy as np
from sklearn.feature_selection.univariate_selection import check_X_y, f_classif, _BaseFilter, _clean_nans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection.base import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import logging

logger = logging.getLogger(__name__)


class MultiOutputSelect(BaseEstimator, SelectorMixin):
    """Select features for multivariate outcomes based on a per-outcome selection strategy.

    This feature selector works by taking a given "base" feature selector and uses this
    to determine retained features on a per-outcome basis.  Given this determination
    per outcome, an aggregation is then drawn across all outcomes to given one single set
    of resulting features to keep (available via \code{get_support}).

    Note that the base selector given is generally specific to regression or classification methods
    while this selector itself is applicable to both, as an abstraction over more specific techniques.

    Parameters
    ----------
    selector : SelectorMixin
        Selector to be used for determining feature "support" for each outcome/response.

    strategy : 'any' or 'all'
        Method with which to determine resulting feature support across all per-outcome feature
        support vectors:
        - any: Any feature found to be significant by per-outcome selector will be retained
        - all: A single feature must be found to be significant by ALL per-outcome selectors to be retained

    Attributes
    ----------
    support_matrix_ : array-like, shape=(n_outcomes, n_features)
        Feature support for each response/outcome

    See also
    --------
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    SelectPercentile: Select features based on percentile of the highest scores.
    GenericUnivariateSelect: Univariate feature selector with configurable mode.
    """

    def __init__(self, selector=SelectKBest(), strategy='any'):
        super(MultiOutputSelect, self).__init__()
        self.selector = selector
        self.strategy = strategy
        self.support_matrix_ = None

    def _check_params(self, X, y):
        if not isinstance(self.selector, SelectorMixin):
            raise ValueError("Selector must be of type 'SelectorMixin' (got '{}')".format(type(self.selector)))
        if len(y.shape) < 2:
            raise ValueError("Invalid Y shape ({}); multivariate selector is meant to be used with multidimensional "
                             "responses (i.e. y must have 2 dimensions)".format(y.shape))
        if self.strategy not in ['any', 'all']:
            raise ValueError('Support resolution strategy "{}" is not valid '
                             '(should be one of ["any", "all"])'.format(self.strategy))

    def fit(self, X, y, **kwargs):
        """Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # print('MultiOutputSelect: ', type(y))
        # X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)
        self._check_params(X, y)
        n_cols = y.shape[1]

        # Generate support vectors for each column, stored here as a new row per support vector;
        # E.g. a 2 column response could have support like:
        #              Feat 1  Feat 2  Feat 3  Feat 4
        # response 1   True    False   False   True
        # response 2   False   True    False   True
        self.support_matrix_ = np.array([self.selector.fit(X, y[:, j]).get_support() for j in range(n_cols)])

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'support_matrix_')

        if self.strategy == 'any':
            mask = np.any(self.support_matrix_, axis=0)
        elif self.strategy == 'all':
            mask = np.all(self.support_matrix_, axis=0)
        else:
            raise ValueError('Support resolution strategy "{}" is not valid '
                             '(should be one of ["any", "all"])'.format(self.strategy))

        # Throw an error if no features would be retained since this will inevitably be a problem downstream,
        # and the context around that problem will likely be less clear (as will possible corrective actions)
        if not np.any(mask):
            raise ValueError('No features were retained in selection.  Consider changing selection strategies '
                             'across outcomes or using a less restrictive base selector method')

        return mask


class SelectAtMostKBest(_BaseFilter):
    """Select features according to the k highest scores.

    This is an adaptation of the native SelectKBest class that allows for the specification
    of the number of features to select (k) to be equal to or greater than the number
    of features present.  This is useful when selection follows several other operations
    that may remove unnecessary features (such as variance filters) in a pipeline, and is
    configurable through a "validate" flag that will enable/disable this feature.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.

    k : int or "all", optional, default=10
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.

    validate : bool
        Whether or not to ensure that k <= n_features

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.

    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continious target.
    SelectPercentile: Select features based on percentile of the highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable mode.
    """

    def __init__(self, score_func=f_classif, k=10, validate=True):
        super(SelectAtMostKBest, self).__init__(score_func)
        self.k = k
        self.validate = validate

    def _check_params(self, X, y):
        max_k = X.shape[1] if self.validate else np.inf
        if not (self.k == "all" or 0 <= self.k <= max_k):
            raise ValueError("k should be >=0, <= n_features; got %r."
                             "Use k='all' to return all features."
                             % self.k)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        if self.k == 'all':
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            k = self.k
            if k > len(mask):
                if self.validate:
                    logger.warning('Configured k ({}) is larger than number of features ({}) '
                                   'so none will be eliminated'.format(k, len(mask)))
                k = len(mask)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            mask[np.argsort(scores, kind="mergesort")[-k:]] = 1
            return mask
