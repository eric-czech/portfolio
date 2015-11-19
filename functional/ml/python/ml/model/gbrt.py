__author__ = 'eczech'

# Utility methods for working with GradientBoosting[Classifier|Regressor]

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid


# def get_loss_grid(gbrt, X, y, param_grid, test_size=.3, **kwargs):
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, **kwargs)
#
#     gbrt.fit(X_train, y_train)
#     n_estimators = len(gbrt.estimators_)
#
#     test_loss = np.empty(n_estimators)
#     for i, pred in enumerate(gbrt.staged_predict(X_test)):
#         test_loss[i] = gbrt.loss_(y_test, pred)
#     return pd.DataFrame({
#         'n_estimators': np.arange(n_estimators) + 1,
#         'training_loss': gbrt.train_score_,
#         'test_loss': test_loss
#     })


def get_loss(gbrt, X, y, test_size=.3, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, **kwargs)

    gbrt.fit(X_train, y_train)
    n_estimators = len(gbrt.estimators_)

    test_loss = np.empty(n_estimators)
    for i, pred in enumerate(gbrt.staged_predict(X_test)):
        test_loss[i] = gbrt.loss_(y_test, pred)
    return pd.DataFrame({
        'n_estimators': np.arange(n_estimators) + 1,
        'training_loss': gbrt.train_score_,
        'test_loss': test_loss
    })


def plot_loss(gbrt_loss, figsize=(12,6)):
    ax = gbrt_loss.set_index('n_estimators').plot(figsize=figsize, alpha=.5)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Loss')
    return ax


# def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6',
#
#                   test_color='#d7191c', alpha=1.0):
#     """Deviance plot for ``est``, use ``X_test`` and ``y_test`` for test error. """
#     test_dev = np.empty(n_estimators)
#
#     for i, pred in enumerate(est.staged_predict(X_test)):
#         test_dev[i] = est.loss_(y_test, pred)
#
#     if ax is None:
#         fig = plt.figure(figsize=(12, 6))
#         ax = plt.gca()
#
#     ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label,
#              linewidth=2, alpha=alpha)
#     ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color,
#              label='Train %s' % label, linewidth=2, alpha=.2)
#     ax.set_ylabel('Error')
#     ax.set_xlabel('n_estimators')
#     #ax.set_ylim((0., 1.))
#     return test_dev, ax
#
# test_dev, ax = deviance_plot(est, X_test.drop('date', axis=1), y_test)
# ax.legend(loc='upper right')