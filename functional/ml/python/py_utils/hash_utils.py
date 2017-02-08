

import hashlib


def _hash(hash_fn, byte_value):
    return hash_fn(byte_value).hexdigest()


def _hash_string(hash_fn, str_value, encoding):
    if str_value is None:
        return None
    if not isinstance(str_value, str):
        raise ValueError(
                '{} hash function input value must be a string (received "{}" instead)'
                .format(hash_fn.__name__, type(str_value))
        )
    return hash_fn(str_value.encode(encoding)).hexdigest()


def md5(value, encoding='utf-8'):
    return _hash_string(hashlib.md5, value, encoding)


def sha1(value, encoding='utf-8'):
    return _hash_string(hashlib.sha1, value, encoding)


def sha256(value, encoding='utf-8'):
    return _hash_string(hashlib.sha256, value, encoding)
