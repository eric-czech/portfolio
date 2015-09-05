__author__ = 'eczech'

from sklearn.metrics import average_precision_score


def avg_precision_score(estimator, X, y):
    y_proba = estimator.predict_proba(X)[:, 1]
    r = average_precision_score(y, y_proba)
    return r