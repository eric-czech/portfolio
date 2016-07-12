
import pandas as pd
import numpy as np
from unbalanced_dataset import UnderSampler


def downsample(d, response, random_state=None, preserve_index=False, verbose=True):
    """ Downsample data frame
    :param d: Data frame to be downsampled
    :param response: Field within data frame to use for downsampling (must contain
        only two unique values; an error will be thrown otherwise)
    :param random_state: Random state to use for downsampling
    :param preserve_index: Determines whether or not the index associated with the given
        data frame will be reattached to the downsampled result; if true, the names of all
        index fields must be non-null
    :param verbose: Flag indicating whether or not summaries of class frequencies should be printed
    :return: Data frame idential to "d" with some rows removed, and the values in "response"
        occurring with an equal frequency
    """
    sampler = UnderSampler(random_state=random_state, replacement=False, verbose=verbose)
    idx = None

    # If index preservation is requested, store the index field names before reseting it
    # on the input data frame (and make sure none of the names are null)
    if preserve_index:
        assert not np.any(pd.isnull(d.index.names)), \
            'When downsampling with "preserve_index=True", index field names must all be non-null.  ' \
            'At least one name was null for the given index.  Index names given: {}'.format(d.index.names)
        idx = list(d.index.names)
        d = d.reset_index()

    # Capture original data frame types and column names
    dtypes = d.dtypes.to_dict()
    cols = d.columns

    # Ensure that the field to be used for downsampling is present
    assert response in cols, \
        'Given response to use for downsampling "{}" was not found in dataset to be downsampled'.format(response)

    # Downsample dataset (as numpy arrays)
    ds, _ = sampler.fit_transform(d.values, d[response].values)

    # Re-conform resampled frame to original (add cols + index)
    d = pd.DataFrame(ds, columns=cols)
    for c in d:
        d[c] = d[c].astype(dtypes[c])
    if preserve_index:
        d = d.set_index(idx)

    # Return result
    return d