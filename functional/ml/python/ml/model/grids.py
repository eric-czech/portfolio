__author__ = 'eczech'


from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import RandomizedSearchCV, ParameterGrid
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np


def get_svr_search_estimators(scaling=True, cv=3, n_iter_svr=1E4, n_iter_grid=50):
    svr_lin = ParameterGrid({'kernel': ['linear'], 'C': np.logspace(-8, -8, 12)})
    svr_rbf = ParameterGrid({'kernel': ['rbf'],    'C': np.logspace(-8, 8, 12), 'gamma': np.logspace(-8, 8, 12)})
    svr_pol = ParameterGrid({'kernel': ['poly'],   'C': np.logspace(-8, 8, 12), 'gamma': np.logspace(-8, 8, 12), 'degree': [2, 3]})
    svr_est = SVR(verbose=0, max_iter=n_iter_svr)

    def n_iter(grid, max_iter_grid):
        return np.min(len(grid), max_iter_grid)

    def pipeline(est):
        return Pipeline([('scale', StandardScaler()), ('est', est)])

    clfs = {
        'svr_lin': RandomizedSearchCV(svr_est, svr_lin.param_grid, cv=cv, n_iter=n_iter(svr_lin, n_iter_grid)),
        'svr_rbf': RandomizedSearchCV(svr_est, svr_rbf.param_grid, cv=cv, n_iter=n_iter(svr_rbf, n_iter_grid)),
        'svr_pol': RandomizedSearchCV(svr_est, svr_pol.param_grid, cv=cv, n_iter=n_iter(svr_pol, n_iter_grid))
    }
    if scaling:
        for name in clfs:
            clfs[name] = pipeline(clfs[name])

    return clfs


def get_gbr_search_estimator(cv=3, n_iter_grid=50, base_est=None):
    gbr_grid = {
        'learning_rate': [.1, .05, .02, .01, .005, .001],
        'max_depth': [3, 5, 7, 9, 13],
        'min_samples_leaf': [1, 3, 5, 9, 17],
        'max_features': [1., .3, .1, None],
        'subsample': [1., .5, .3],
        'n_estimators': [3000]
    }
    base_est = GradientBoostingRegressor() if base_est is None else base_est
    return {'gbr': RandomizedSearchCV(base_est, gbr_grid, n_iter=n_iter_grid, cv=cv)}


# http://www.slideshare.net/PyData/gradient-boosted-regression-trees-in-scikit-learn-gilles-louppe

GBR_STANDARD_1 = {
    'learning_rate': [.1, .05, .02, .01],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'max_features': [1., .3, .1],
    'n_estimators': [3000]
}


