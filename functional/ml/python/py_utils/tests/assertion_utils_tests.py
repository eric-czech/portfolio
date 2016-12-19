from unittest import TestCase

import pandas as pd
from py_utils import assertion_utils


class TestAssertionUtils(TestCase):

    def test_object_type_assertion(self):
        df = pd.DataFrame({'x': [2, 'v'], 'y': ['x', 'y']})

        # Ensure that the integer value in x raises an exception
        self.assertRaises(AssertionError, lambda: assertion_utils.assert_object_types(df))

        # Ensure that when allowing strings and integers, no exception is raised
        assertion_utils.assert_object_types(df, {'x': [str, int]})
