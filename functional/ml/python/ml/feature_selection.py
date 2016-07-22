
from sklearn.feature_selection import SelectKBest
import numpy as np


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