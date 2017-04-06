import logging
from multiprocessing import Lock

_HR = '-' * 80
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DEFAULT_LOG_FILE = '/tmp/ml-models-exec.log'
LOG_FILE_HANDLER = logging.FileHandler(DEFAULT_LOG_FILE)
LOGGER.addHandler(LOG_FILE_HANDLER)

DEFAULT_LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'

_LOG_LOCK = Lock()


def set_logger(log_file, log_format=DEFAULT_LOG_FORMAT):
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    global LOG_FILE_HANDLER
    LOGGER.removeHandler(LOG_FILE_HANDLER)
    LOG_FILE_HANDLER = logging.FileHandler(log_file)
    LOG_FILE_HANDLER.setFormatter(logging.Formatter(log_format))
    LOGGER.addHandler(LOG_FILE_HANDLER)


def log_hr():
    log(_HR)


def log(msg, level=logging.INFO):
    _LOG_LOCK.acquire()
    LOGGER.log(msg, level)
    _LOG_LOCK.release()
