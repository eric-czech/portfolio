
import numpy as np
import pandas as pd

import numpy as np

def to_data_frame(v, index=None, columns=None):
    """ Converts a given vector or matrix into a DataFrame

     This conversion results in the following transformations:
        - Given: 1-D numpy array -> One column data frame with arbitrary columns and index (unless either is provided)
        - Given: 2-D numpy array -> Data frame with arbitrary columns and index (unless either is provided)
        - Given: Series -> Data frame with the same index and a single column having the same name as the series
        - Given: DataFrame -> Nothing, the given value is returned as is
        - Anything else -> ValueError

    :param v: Vector or matrix
    :param index: Index to assign to a converted numpy array
        (will be ignored if `v` is a Series or DataFrame already)
    :param columns: Columns to assign to a converted numpy array
        (will be ignored if `v` is a Series or DataFrame already)
    :return: DataFrame
    """
    if v is None:
        return None

    if len(v.shape) > 2:
        raise ValueError('Shape of given data "{}" cannot have greater than 2 dimensions'.format(v.shape))

    if isinstance(v, np.ndarray):
        # If `v` is a 1-D array, reshape to single column matrix
        if len(v.shape) == 1:
            v = np.reshape(v, (-1, 1))
        return pd.DataFrame(v, index=index, columns=columns)
    elif isinstance(v, pd.Series):
        return v.to_frame()
    elif isinstance(v, pd.DataFrame):
        return v
    else:
        raise ValueError('Expected numpy array or pandas object, but was given "{}" instead'.format(type(v)))


def to_tabular_array(v):
    """
    Conform array to 2D structure (if already 1 or 2 dimensional)

    This is useful when dealing with arrays that may be 1 or 2 dimensional and it is more desirable
    to work with the array as either a 2D array (if already 2 dimensional) or a Nx1 2D array if 1 dimensional

    :param v: Array to convert
    :return: Array with 2 dimensions
    """
    # If the array is 1D, convert to 2D
    if len(v.shape) == 1:
        return np.reshape(v, (-1, 1))
    assert len(v.shape) == 2, 'Array should not have more than 2 dimensions'
    return np.array(v)
