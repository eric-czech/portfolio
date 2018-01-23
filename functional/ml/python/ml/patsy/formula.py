#  Helper functions for formula construction

from ml.patsy.models import PatsyTransformer
from patsy import ContrastMatrix
import numpy as np


class OneHotContrast(object):
    def __init__(self, reference=0):
        self.reference = reference

    def code_with_intercept(self, levels):
        return ContrastMatrix(np.eye(len(levels)), ["[O.%s]" % (level,) for level in levels])

    def code_without_intercept(self, levels):
        return self.code_with_intercept(levels)


def quote(cols):
        return [("Q('{}')".format(c) if "Q(" not in c else c) for c in cols]


def build(cols, terms=None, add_terms=None, sub_terms=None):
    formula = ' + '.join(quote(cols))
    if terms is not None:
        formula += ' '.join(quote(terms))
    if add_terms is not None:
        formula = '{} + {}'.format(formula, ' + '.join(quote(add_terms)))
    if sub_terms is not None:
        formula = '{} - {}'.format(formula, ' - '.join(quote(sub_terms)))
    return formula


def transformer(formula, **kwargs):
    if 'return_type' not in kwargs:
        kwargs['return_type'] = 'dataframe'
    return PatsyTransformer(formula,  **kwargs)


def apply(formula, df, **kwargs):
    if 'return_type' not in kwargs:
        kwargs['return_type'] = 'dataframe'
    return PatsyTransformer(formula,  **kwargs).fit_transform(df)


