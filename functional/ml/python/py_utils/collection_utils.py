

def to_batches(sequence, batch_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i+batch_size]


def subset(d, subset_func, subset_op=None, log=True):
    """
    Applies a filter/subset function to a data structure and summarizes reduction in size
    :param d: Data structure to subset (anything with "len")
    :param subset_func: Function to use for subsetting; function should expect a single data structure,
        subset that structure, and return a result of the same typpe
    :param subset_op: Optional name of subset operation to show in summary (for convenience)
    :param log:
        - True/False - If boolean, then summary will be logged to stdout if True
        - logging.Logger - Logger to use for summary
        - None - No summary will be printed/logged
    :return: Original data structure
    """

    # Apply subset function and record size before and after
    n_before = len(d)
    d = subset_func(d)
    n_after = len(d)
    n_diff = n_before - n_after

    if log:
        # Create summary message
        subset_op = subset_op if subset_op else 'Subset Operation'
        msg = '{}Records before = {}, Records after = {}, Records removed = {} (%{:.2f})'\
            .format(
                '[{}] '.format(subset_op),
                n_before, n_after, n_diff,
                100. * n_diff / n_before if n_before > 0 else 0.
            )

        # Print summary
        if isinstance(log, bool):
            print(msg)
        else:
            log.info(msg)

    return d


def transform(d, field, mask, value, trans_op=None, log=True, target_field=None, log_no_change=False):
    """
    Applies a conditional transformation and logs number of records affected

    Equivalent to: d[target_field] = np.where(mask, value, d[field])

    :param d: DataFrame
    :param field: Field to apply transformation to
    :param mask: Mask to use for conditional transformation; should be true where values WILL be transformed
    :param trans_op: Optional name of subset operation to show in summary (for convenience)
    :param log:
        - True/False - If boolean, then summary will be logged to stdout if True
        - logging.Logger - Logger to use for summary
        - None - No summary will be printed/logged
    :param target_field: Field to which result will be attached in data frame; defaults to `field`
    :param log_no_change: Flag to indicate whether message should be logged even if no transformation occurs; defaults
        to False (i.e. a message will show that 0 rows have changed)
    :return: Original data structure
    """
    import numpy as np
    assert mask.dtype == np.bool

    # Apply subset function and record size before and after
    n = len(d)
    n_chg = mask.sum()
    target_field = target_field if target_field else field

    # Return immediately if nothing further should occur (either logging or transformation)
    if n_chg == 0 and not log_no_change:
        return d

    # Short-circuit transformation if no values would be changed
    if n_chg > 0:
        d[target_field] = np.where(mask, value, d[field])

    if log:
        # Create summary message
        trans_op = trans_op if trans_op else 'Transform {}'.format(field)

        msg = '{}Records changed = {} of {} (%{:.2f})'\
            .format(
                '[{}] '.format(trans_op),
                n_chg, n, 100. * n_chg / n if n > 0 else 0.
            )

        # Print summary
        if isinstance(log, bool):
            print(msg)
        else:
            log.info(msg)

    return d


def nested_dictionary():
    """ Get empty dictionary with values that default to new dictionaries if keys are absent

    :examples:
    d = nested_dictionary()
    d[3]['z'][9] = ('a', 'b', 'c')

    :return: Empty dictionary
    """
    import collections
    nested_dict = lambda: collections.defaultdict(nested_dict)
    return nested_dict()


def cross_product(x, y):
    """
    Return cross product of 1D item collection as 2 column numpy array

    :param x: First sequence of items
    :param y: Second sequence of items
    :return: 2-D numpy array with two columns and len(x) * len(y) rows (every combination of x and y values); x values
        will be in first column and y values in second
    """
    import numpy as np
    if not isinstance(x, np.array):
        x = np.array(x)
    if not isinstance(y, np.array):
        y = np.array(y)
    assert x.ndim == 1, 'x values must be one dimensional'
    assert y.ndim == 1, 'y values must be one dimensional'
    r = np.hstack([np.expand_dims(x.ravel(), 1) for x in np.meshgrid(x, y)])
    assert r.shape[0] == len(x) * len(y), \
        'Number of cross product combinations found ({}) did not match the number expected ({})'\
        .format(r.shape[0], len(x) * len(y))
    return r
