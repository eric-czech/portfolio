
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class TensorCollector(object):
    """ Collector class to be used for retrieving individual tensor values"""

    def __init__(self, enabled=True):
        self.data = {}
        self.enabled = enabled

    def register(self, t, extraction_fn, name=None):
        """
        Register a tensor for collection
        :param t: Tensor to collect data for
        :param extraction_fn: Function with signature fn(x) returning any value, though this value will always
            be appended to the end of a list on each training iteration so typically only a list, array or scalar
            value is useful
        :param name: Name to key tensor values by; if not given, tensor name will be used
        :return: Original tensor (t)
        """
        if not self.enabled:
            return t

        if name is None:
            n = t.name.split(':')[0]
        else:
            n = name

        if n not in self.data:
            self.data[n] = []

        def extract_value(x):
            self.data[n].append(extraction_fn(x))
            return x
        collect_op = tf.py_func(extract_value, [t], [t.dtype], name=n)[0]
        with tf.control_dependencies([collect_op]):
            r = tf.identity(t)
        return r


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


def pearson_correlation(self, Y1, Y2):
    p_top = tf.reduce_mean((Y1 - tf.reduce_mean(Y1, axis=0)) * (Y2 - tf.reduce_mean(Y2, axis=0)), axis=0)
    p_bottom = self._std(Y1) * self._std(Y2)
    return p_top / p_bottom


# def _std(Y):
#     devs_squared = tf.square(Y - tf.reduce_mean(Y, axis=0))
#     return tf.sqrt(tf.reduce_mean(devs_squared, axis=0))
#
# def _pearson(Y1, Y2):
#     p_top = tf.reduce_mean((Y1 - tf.reduce_mean(Y1, axis=0)) * (Y2 - tf.reduce_mean(Y2, axis=0)), axis=0)
#     p_bottom = _std(Y1) * _std(Y2)
#     return p_top / p_bottom
#
# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# y = x + np.random.randn(3, 3)
#
# top = np.mean((x - np.mean(x, axis=0)) * (y - np.mean(y, axis=0)), axis=0)
# bottom = np.std(x, axis=0) * np.std(y, axis=0)
# top / bottom
#
# from ml.tensorflow.utilities import tf_eval
# tf.reset_default_graph()
# tf_eval(_pearson(tf.constant(x, dtype=tf.float64), tf.constant(y, dtype=tf.float64)))
#
# [pd.Series(x[:, i]).corr(pd.Series(y[:, i])) for i in range(3)]
