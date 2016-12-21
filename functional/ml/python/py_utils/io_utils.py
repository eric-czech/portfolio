
import dill
import logging
logger = logging.getLogger(__name__)


def _obj_name(obj_name):
    return '"{}" '.format(obj_name) if obj_name else ''


def to_pickle(obj, file_path, obj_name=None):
    obj_name = '"{}" to'.format(obj_name) if obj_name else 'to'
    logger.debug('Serializing object {}to location "{}"'.format(_obj_name(obj_name), file_path))
    with open(file_path, 'wb') as fd:
        dill.dump(obj, fd)
    return file_path


def from_pickle(file_path, obj_name=None):
    logger.debug('Restoring serialized object {}from location "{}"'.format(_obj_name(obj_name), file_path))
    with open(file_path, 'rb') as fd:
        return dill.load(fd)
