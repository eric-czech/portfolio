from unittest import TestCase

import pandas as pd
from pandas import Timestamp
from py_utils import date_utils


class TestDateUtils(TestCase):

    def test_quarterly_rounding(self):
        dates = {
            Timestamp('2015-12-31 00:00:00'): Timestamp('2015-10-01 00:00:00'),
            Timestamp('2016-01-01 00:00:00'): Timestamp('2016-01-01 00:00:00'),
            Timestamp('2016-03-31 00:00:00'): Timestamp('2016-01-01 00:00:00'),
            Timestamp('2016-04-01 00:00:00'): Timestamp('2016-04-01 00:00:00'),
            Timestamp('2016-05-01 00:00:00'): Timestamp('2016-04-01 00:00:00'),
            Timestamp('2016-06-01 00:00:00'): Timestamp('2016-04-01 00:00:00'),
            Timestamp('2016-06-30 00:00:00'): Timestamp('2016-04-01 00:00:00'),
            Timestamp('2016-07-01 00:00:00'): Timestamp('2016-07-01 00:00:00')
        }
        freq = pd.tseries.offsets.QuarterBegin(normalize=True, startingMonth=1)
        rounder = date_utils.anchored_date_rounder(freq)
        for d in dates:
            self.assertEqual(dates[d], rounder(d))

