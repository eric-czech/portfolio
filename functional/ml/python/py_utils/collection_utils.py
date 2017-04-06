

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
    :return:
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
