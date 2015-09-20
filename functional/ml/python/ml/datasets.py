__author__ = 'eczech'

import pandas as pd
import numpy as np
from sklearn import datasets


def get_test_classifier_data(multinomial=False):
    d = datasets.load_iris()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    if multinomial:
        df['target'] = d.target
    else:
        df['target'] = np.where(d.target == 1, 0, 1)
    return df[d.feature_names], df['target']


def get_test_regressor_data():
    d = datasets.load_boston()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df['target'] = d.target
    return df[d.feature_names], df['target']