# Date Utilities

import pandas as pd
from pandas.tseries.frequencies import to_offset


def anchored_date_rounder(freq):
    """
    Returns function for rounding ANCHORED dates using semantics like "Weeks beginning on X" or "Business Days".

    Note the the given frequency must have isAnchored = True or an error will be thrown

    :param freq: pd.tseries.offset instance or frequency string (must be of some anchored type)
    :return: Date rounding function
    """
    if isinstance(freq, str):
        freq = to_offset(freq)
        freq.normalize = True
    else:
        assert freq.normalize, \
            'Date rounding frequencies should have normalize = True for rounding'
    assert freq.isAnchored(), \
        'Date rounding frequencies must be "anchored" to work correctly ' \
        '(eg weeks starting on mondays -- Week(n=1, normalize=True, weekday=0))'
    return lambda x: pd.NaT if pd.isnull(x) else x + freq - freq


def date_rounder(freq):
    """
    Returns function for rounding dates/times with constant frequencies (eg Days, Hours, 15 Minutes)

    This function does not apply to more complicated rounding schemes like business days (see anchored_date_rounder).

    :param freq: pd.tseries.offset instance or frequency string
    :return: Date rounding function
    """
    if isinstance(freq, str):
        freq = to_offset(freq)
    return lambda x: pd.NaT if pd.isnull(x) else pd.Timestamp((x.value // freq.delta.value) * freq.delta.value)

