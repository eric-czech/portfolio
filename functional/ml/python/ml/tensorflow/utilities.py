
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def tf_eval(exp):
    sess = tf.InteractiveSession()
    res = sess.run(exp)
    sess.close()
    return res


def tf_print(t, transform=None):
    def log_value(x):
        logger.info('{} - {}'.format(t.name, x if transform is None else transform(x)))
        return x
    log_op = tf.py_func(log_value, [t], [t.dtype], name=t.name.split(':')[0])[0]
    with tf.control_dependencies([log_op]):
        r = tf.identity(t)
    return r


def tf_devices():
    """ Returns devices available for computations """
    from tensorflow.python.client import device_lib
    return device_lib.list_local_devices()
