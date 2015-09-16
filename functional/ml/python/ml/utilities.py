
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


def get_one_hot_features(df, **kwargs):
    """
    Returns a one-hot encoded version of the given data
    :param df:
    :param kwargs:
    :return:
    """

    # The column labeling cannot be done using the 'sparse' option
    # so make sure that is overriden in the given arguments for OneHotEncoder
    kwargs['sparse'] = False
    ohe = OneHotEncoder(**kwargs)
    df_ohe = ohe.fit_transform(df)
    feats = df.columns.tolist()
    cols = []
    fi = ohe.feature_indices_
    for i in range(len(fi)-1):
        feat = feats[i]
        n = fi[i+1] - fi[i]
        cols.extend(['{}.{}'.format(feat, j) for j in range(n)])
    return pd.DataFrame(df_ohe, columns=cols)


def get_polynomial_features(df, interaction_sign=' x ', **kwargs):
    """
    Gets polynomial features for the given data frame using the given sklearn.PolynomialFeatures arguments
    :param df: DataFrame to create new features from
    :param kwargs: Arguments for PolynomialFeatures
    :return: DataFrame with labeled polynomial feature values
    """
    pf = PolynomialFeatures(**kwargs)
    feats = _get_polynomial_features(df.columns.tolist(), pf.fit(df), interaction_sign=interaction_sign)
    return pd.DataFrame(pf.transform(df), columns=feats)


def _get_polynomial_features(cols, pf, interaction_sign=' x '):
    """
    Determines names for new polynomial features based on power and interaction depth
    :param cols: Original list of columns
    :param pf: Fitted sklearn.PolynomialFeatures instance
    :return: List of new feature names
    """
    res = []
    for poly_feat in pf.powers_:
        feat_idx = np.argwhere(poly_feat > 0).ravel()
        feat_names = []
        for i in feat_idx:
            name = cols[i]
            if poly_feat[i] > 1:
                name = '{}^{}'.format(name, poly_feat[i])
            feat_names.append(name)
        if len(feat_names) == 0:
            feat_names = ['Intercept']
        res.append(interaction_sign.join(feat_names))
    return res