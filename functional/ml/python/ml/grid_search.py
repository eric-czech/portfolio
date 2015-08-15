__author__ = 'eczech'

from sklearn.grid_search import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from sklearn import base

# This was a start an attempt to emulate GridSearchCV
# to deal with multithreading issues when using custom scoring functions

class GridSearch(object):

    def __init__(self, estimator, param_grid, scorer=None, cv=None):
        self.estimator = estimator
        self.param_grid = ParameterGrid(param_grid)
        self.scorer = scorer
        self.cv = cv

    def _fit(self, arg):
        X, y, param, train, test = arg
        clf = base.clone(self.estimator)
        print(param)
        clf.set_params(**param)
        clf.fit(X.iloc[train], y.iloc[train])
        return (self.scorer(clf, X.iloc[test], y.iloc[test]), clf)

    def fit(self, X, y=None):
        args = []
        for param in list(self.param_grid):
            for train, test in self.cv:
                args.append((X, y, param, train, test))
        self.res = Parallel(n_jobs=3, backend='multiprocessing', verbose=100, pre_dispatch=1)(
            delayed(self._fit)(arg) for arg in args)
        return self.res


# Client code for above:

# clf = LogisticRegression(penalty='l1')
# clf.set_params()
# grid = {
#     'class_weight': class_weights,
#     'C': np.logspace(-3, 0, num=3)
# }
# X, y = d_in[features], d_in[response]
# cv = StratifiedKFold(y, 3)
# def avg_prec_score(estimator, X, y):
#     print('using scorer')
#     y_proba = estimator.predict_proba(X)[:, 1]
#     return average_precision_score(y, y_proba)
# gclf = GridSearch(clf, grid, scorer=avg_prec_score, cv=cv)
# res = gclf.fit(X, y)