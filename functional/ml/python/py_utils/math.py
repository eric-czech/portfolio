
import numpy as np


def sigmoid(x, clip=False, clip_val=256):
    """ Numerically-stable, vectorized sigmoid function.

    This mirrors how sigmoid evaluations are made within scipy, and works well for extremely large or small inputs.

    Sources:
        - http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        - https://github.com/scipy/scipy/pull/3386/commits/47b73a3a87b86b4e18a7f21482eeaa66e26467d0#diff-3e18974596db8da1b9e7e036caae042a

    :param x: array-like or scalar value
    :param clip: flag indicating whether or not x values should be clipped to a certain range; this has the effect of:
        1. when true, avoids numpy warnings for floating point overflow in np.exp(x) evaluation (though a stable version
            of sigmoid calculations isn't really necessary when inputs are clipped)
        2. when false, results will still be stable however warnings are common for input x values over 500 or so
            because of floating point overflow/underflow (though the stability of the calculation will still produce
            correct results using infinite values)
    :param clip_val: max absolute values of input values
    :return: sigmoid of x, as array with same shape
    """
    if clip:
        x = np.clip(x, -clip_val, clip_val)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

