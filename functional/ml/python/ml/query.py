__author__ = 'eczech'

import re


def feature_regex_query(d, regex):
    """
    Returns the columns in the given data frame with names that match the given regular expression

    :param d: DataFrame to select features from
    :param regex: Regular expression to use to match features
    :return: DataFrame with same number of rows as #d, and possibly fewer columns

    Example:
        d_subset = feature_regex_query(d, '(age)|(gender)|(.*p10$)')
        # Returns all features exactly equal to 'age' or 'gender' or that end with 'p10'
    """
    regex = re.compile(regex)
    cols = [c for c in d if regex.match(c) is not None]
    return d[cols]