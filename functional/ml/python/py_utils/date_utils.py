# Date Utilities

import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np

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


def get_delay_between_events(events):
    """
    Returns time between events in unit increments.

    :param events: A boolean series with adjacent, evenly incremented index (the time between events returned
        will be in number of index increments -- not a specific timescale)
    :return: Integer series with time between events
    """
    assert events.dtype == np.bool, 'Event series must be boolean'
    assert np.all(events.notnull()), 'Event timeline cannot have null values'

    # Find the cumulative sum of the absence of events
    x1 = (~events).cumsum()

    # Replace the sum above with when not on a positive diff (i.e. when there is not an event),
    # and then forward fill the cumulative some over those newly introduced na's
    x2 = x1.where(events, np.nan).ffill()

    # Subtract the cumulative sum of non-events from the cumulative frozen in a point-in-time on each event
    # to give the correct spacing between events in index increments
    return x1 - x2
