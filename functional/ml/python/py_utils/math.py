
import numpy as np


def sigmoid(x):
    """Computes sigmoid of `x` element-wise.

    Specifically, `y = 1 / (1 + exp(-x))`.

    :param x: 1-D numpy array
    :return: 1-D array of same length as `x`
    """
    return 1. / (1. + np.exp(-x))
