
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

import numpy as np


def get_feature_selector(X, regex, negate=False):
    from sklearn.preprocessing import FunctionTransformer
    cols = X.columns.tolist()
    feats = X.filter(regex=regex).columns.tolist()
    idx = [cols.index(c) for c in feats]
    if negate:
        selector = lambda X: X[:, [i for i in range(X.shape[1]) if i not in idx]]
    else:
        selector = lambda X: X[:, idx]
    return FunctionTransformer(selector)


def get_static_feature_selector(all_features, target_features):
    """
    Returns a feature selector that will only choose specific features by name.

    This is useful for comparing models to some baseline that only uses some feature subset.
    :param all_features: Full feature list
    :param target_features: Target features to be selected
    :return: Sklearn feature selector
    """
    c_diff = np.setdiff1d(target_features, all_features)
    assert len(c_diff) == 0, \
        'At least one of the target features does not actually exist in the set of all possible features.  '\
        'Features not present = {}'.format(c_diff)

    scores = np.array([int(c in target_features) for c in all_features])

    def selector(*_):
        # Return (scores, pvalues) where scores is binary indicator for target features
        # and pvalues is opposite of scores
        return scores, np.abs(1 - scores)

    return SelectKBest(selector, k=np.sum(scores))


def _identity(X):
    """The identity function.
    """
    return X


class FnTransformer(BaseEstimator, TransformerMixin):
    """Constructs a transformer from an arbitrary callable.

    ** This is class is from scikit-learn 17.1 and is included here for use in previous versions

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    A FunctionTransformer will not do any checks on its function's output.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    .. versionadded:: 0.17

    Parameters
    ----------
    func : callable, optional default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.

    validate : bool, optional default=True
        Indicate that the input X array should be checked before calling
        func. If validate is false, there will be no input validation.
        If it is true, then X will be converted to a 2-dimensional NumPy
        array or sparse matrix. If this conversion is not possible or X
        contains NaN or infinity, an exception is raised.

    accept_sparse : boolean, optional
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.

    pass_y: bool, optional default=False
        Indicate that transform should forward the y argument to the
        inner callable.

    """
    def __init__(self, func=None, validate=True,
                 accept_sparse=False, pass_y=False):
        self.func = func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.pass_y = pass_y

    def fit(self, X, y=None):
        if self.validate:
            check_array(X, self.accept_sparse)
        return self

    def transform(self, X, y=None):
        if self.validate:
            X = check_array(X, self.accept_sparse)
        func = self.func if self.func is not None else _identity

        return func(X, *((y,) if self.pass_y else ()))
