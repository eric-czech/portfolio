
import numpy as np


def sigmoid(x):
    """Computes sigmoid of `x` element-wise.

    Specifically, `y = 1 / (1 + exp(-x))`.

    :param x: 1-D numpy array
    :return: 1-D array of same length as `x`
    """
    return 1. / (1. + np.exp(-x))


def sigmoid(x):
    """ Numerically-stable, vectorized sigmoid function.

    This mirrors how sigmoid evaluations are made within scipy, and works well for extremely large or small inputs.

    Sources:
        - http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        - https://github.com/scipy/scipy/pull/3386/commits/47b73a3a87b86b4e18a7f21482eeaa66e26467d0#diff-3e18974596db8da1b9e7e036caae042a
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
