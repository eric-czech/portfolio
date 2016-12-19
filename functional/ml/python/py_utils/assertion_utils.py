
import numpy as np
import pandas as pd


def assert_object_types(d, m_type=None, na_ok=True):
    """ Ensures that the type associated with each value in a Series with dtype "object" is in an expected set of types
    :param d: DataFrame to validate
    :param m_type: Dict of field name to type or list of types; if no type is specified in this
        map for a field, "str" type will be assumed and similarly if the map itself is not provided
        then it will be assumed that object fields should be validated as strings
    :param na_ok: If true, null/na values will be ignored (as will their types)
    :raises: AssertionError if any object does not match the expected types
    """
    if m_type is None:
        m_type = {}

    # Loop through each field in the given data frame
    for c in d:
        # If this field isn't of object type then there's nothing to validate
        if d[c].dtype != np.object:
            continue

        # Get the type filter for this field
        typ = m_type.get(c, str)
        if not isinstance(typ, list):
            typ = [typ]

        # Ensure types given are actually types (and not strings or something of the like)
        for t in typ:
            assert isinstance(t, type), \
                'Type given for column "{}" must be a python type.  Value given = {}'.format(c, typ)

        # Validate field values
        typ_fn = np.vectorize(lambda v: True if pd.isnull(v) and na_ok else type(v) in typ)
        i_assert = typ_fn(d[c])
        if not np.all(i_assert):
            d_ex_v = d[c][~i_assert].head(10)
            d_ex_v.name = 'InvalidValue'
            d_ex_t = d[c].apply(type)[~i_assert].head(10)
            d_ex_t.name = 'InvalidType'
            msg = 'Found at least one value for field "{}" not matching type filter "{}".  '\
                'First 10 offending values:\n{}'.format(c, typ, pd.concat([d_ex_v, d_ex_t], axis=1))
            raise AssertionError(msg)
