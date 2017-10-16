
import os
import dill
import logging
logger = logging.getLogger(__name__)


def _obj_name(obj_name):
    return '"{}" '.format(obj_name) if obj_name else ''


def cache(operation, file_path, obj_name=None, overwrite=False):
    """
    Cache an operation by storing its results as a pickle file

    :param operation: Function to cache results from (presumably this is expensive and slow to run)
    :param file_path: File path in which pickled results will be stored
    :param obj_name: Optional name of object to associate with object for logging message
    :param overwrite: Force operation to run and overwrite previous results (if any) (default False)
    :return: Result of `operation` function
    """
    if not os.path.exists(file_path) or overwrite:
        to_pickle(operation(), file_path, obj_name=obj_name)
    return from_pickle(file_path, obj_name=obj_name)


def to_pickle(obj, file_path, obj_name=None):
    """
    Serialize object to disk as pickle file

    * Workaround taken from [1] for serializing larger objects.

    [1] - http://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb

    :param obj: Object to serialize (as a pickle file)
    :param file_path: Absolute path of serialized (pickle) object; will be written as gzip stream if path ends with .gz
    :param obj_name: Optional name to associate with object for logging message
    :return: File path to pickle file
    """

    logger.debug('Serializing object {}to location "{}"'.format(_obj_name(obj_name), file_path))

    n_bytes = 2**31
    max_bytes = 2**31 - 1
    bytes_out = dill.dumps(obj)

    def open_file(fp):
        if fp.endswith('.gz'):
            import gzip
            return gzip.open(fp, 'wb')
        else:
            return open(fp, 'wb')

    with open_file(file_path) as fd:
        for idx in range(0, n_bytes, max_bytes):
            fd.write(bytes_out[idx:idx+max_bytes])
    return file_path


def from_pickle(file_path, obj_name=None):
    """
    Restore pickled object from disk

    * Workaround taken from [1] for deserializing larger objects

    [1] - http://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb

    :param file_path: Absolute path of serialized (pickle) object (can be gzip compressed if path ends with .gz)
    :param obj_name: Optional name to associate with object for logging message
    :return: Deserialized object
    """
    logger.debug('Restoring serialized object {}from location "{}"'.format(_obj_name(obj_name), file_path))

    # If given gzip file, read in one chunk (this limits source gzip files to ~2GB since
    # there is no way to get the uncompressed size of gzip data over 2GB directly)
    if file_path.endswith('.gz'):
        import gzip
        with gzip.open(file_path, 'rb') as f_in:
            return dill.load(f_in)

    # Otherwise, read arbitrarily large uncompressed file
    bytes_in = bytearray(0)
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as fd:
        for _ in range(0, input_size, max_bytes):
            bytes_in += fd.read(max_bytes)
    return dill.loads(bytes_in)


def gzip(src_file_path, dest_file_path=None):
    import gzip
    import shutil

    if dest_file_path is None:
        dest_file_path = src_file_path + '.gz'

    logger.debug('Compressing "{}" as "{}"'.format(src_file_path, dest_file_path))
    with open(src_file_path, 'rb') as f_in, gzip.open(dest_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
