import collections
import os
import edward as ed
import numpy as np
import tensorflow as tf
import logging
from sklearn.base import BaseEstimator
from edward.util import graphs
from edward.models import Normal, Laplace, PointMass
logger = logging.getLogger(__name__)


# Notes on session configuration for Edward:
# At TOW, the following changes were necessary in edward.util.graphs to set default session configurations -
# def set_default_config(config):
#   global _ED_SESSION_CONFIG
#   _ED_SESSION_CONFIG = config
#
# def get_default_config():
#   global _ED_SESSION_CONFIG
#   return _ED_SESSION_CONFIG
#
#
# def get_session():
#   """Get the globally defined TensorFlow session.
#
#   If the session is not already defined, then the function will create
#   a global session.
#
#   Returns
#   -------
#   _ED_SESSION : tf.InteractiveSession
#   """
#   global _ED_SESSION
#   if tf.get_default_session() is None:
#     config = get_default_config()
#     _ED_SESSION = tf.InteractiveSession(config=config)
#   else:
#     _ED_SESSION = tf.get_default_session()
#
#   if have_keras:
#     K.set_session(_ED_SESSION)
#
#   return _ED_SESSION


class ConvergenceError(Exception):
    pass


class BayesianModel(object):

    def inference_args(self, data):
        raise NotImplementedError('Method should be implemented by subclass')

    def criticism_args(self, sess, var_map):
        raise NotImplementedError('Method should be implemented by subclass')

    def prediction_fn_key(self):
        return 'pred_fn'


def set_default_gpu_config(device_count=1, memory_fraction=.8):
    """ Convenience method for configuring gpu usage
    :param device_count: Set to 0 to disable GPU training
    :param memory_fraction: Fraction of GPU memory to use for Tensorflow
    """
    config = tf.ConfigProto(device_count={'GPU': device_count})
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    set_default_tf_sess_config(config)


def set_default_tf_sess_config(config):
    """ Proxy method to set default configuration used for interactive Edward sessions
    :param config: None or tf.ConfigProto
    """
    graphs.set_default_config(config)


class ModelBuilder(object):

    def __init__(self, inference_fn, add_summaries=True):
        self.map = inference_fn == ed.MAP
        self.latent_map = {}
        self.tensor_map = {}
        self.add_summaries = add_summaries

    def add(self, dist, loc, scale, name,
            loc_transform=tf.identity, scale_transform=tf.nn.softplus,
            scale_coef=.1):

        shape = loc.get_shape().as_list()

        # Parameter prior distribution (eg Normal(0, 1))
        model = dist(loc, scale * tf.ones_like(loc))

        if self.map:
            # Add point mass model for MAP estimation
            q = PointMass(params=loc_transform(tf.Variable(tf.random_normal(shape, stddev=scale * scale_coef))))
            self.latent_map[model] = q
            self.tensor_map[name] = model
            self.tensor_map[name + '.q'] = q.params
            if self.add_summaries:
                tf.summary.histogram(name + '.q', self.tensor_map[name + '.q'])
        else:
            # Add inference model for all other estimations
            q = dist(
                loc_transform(tf.Variable(tf.random_normal(shape, stddev=scale*scale_coef))),
                scale_transform(tf.Variable(tf.random_normal(shape, stddev=scale*scale_coef)))
            )
            self.latent_map[model] = q
            self.tensor_map[name] = model
            if dist in [Normal, Laplace]:
                self.tensor_map[name + '.q'] = q.loc
                self.tensor_map[name + '.s'] = q.scale
            else:
                raise ValueError('Distribution "{}" not yet supported'.format(dist))

            if self.add_summaries:
                tf.summary.histogram(name + '.q', self.tensor_map[name + '.q'])
                tf.summary.histogram(name + '.s', self.tensor_map[name + '.s'])


class BayesianModelEstimator(BaseEstimator):

    def __init__(self, model, n_collect=1,
                 n_print_progress=None, max_steps=1000, tol=1e-1,
                 random_state=None, fail_if_not_converged=True,
                 n_samples=1, n_loss_buffer=10, save_tf_model=True,
                 inference_fn=ed.KLqp, optimizer=None):

        self.max_steps = max_steps
        self.tol = tol
        self.fail_if_not_converged = fail_if_not_converged
        self.n_samples = n_samples
        self.n_loss_buffer = n_loss_buffer
        self.save_tf_model = save_tf_model
        assert n_loss_buffer > 1, 'Loss buffer size must be > 1'
        assert max_steps > n_loss_buffer, 'Max steps must be > n_loss_buffer'

        self.model = model

        self.n_collect = n_collect
        self.n_print_progress = n_print_progress
        self.random_state = random_state
        self.inference_fn = inference_fn
        self.optimizer = optimizer
        self.log_dir = None

        self.losses_ = None
        self.converged_ = False
        self.tensor_map_ = None
        self.criticism_args_ = None

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        return self

    def _init(self):
        ed.get_session().close()
        tf.reset_default_graph()
        self.losses_ = None
        self.converged_ = False
        self.tensor_map_ = None
        self.criticism_args_ = None

    def fit(self, X, Y, **kwargs):
        return self.train({'X': X, 'Y': Y}, **kwargs)

    def train(self, data, **kwargs):
        self._init()
        ed.set_seed(self.random_state)

        sess = ed.get_session()

        input_fn, latent_vars, self.tensor_map_ = self.model.inference_args(data, **kwargs)

        # Initialize inference engine
        inference = self.inference_fn(latent_vars, data=input_fn(data))
        inference_kwargs = {} if self.inference_fn == ed.MAP else {'n_samples': self.n_samples}
        inference.initialize(
                logdir=self.log_dir, n_print=self.n_print_progress,
                optimizer=self.optimizer, **inference_kwargs
        )
        tf.global_variables_initializer().run()

        self.losses_ = []
        self.converged_ = False
        loss_buffer = collections.deque(maxlen=self.n_loss_buffer + 1)
        loss_change = None
        for t in range(self.max_steps):
            info_dict = inference.update()
            loss = info_dict['loss']
            # tf.summary.scalar('Loss', )

            if self.n_print_progress and t % self.n_print_progress == 0:
                logging.info(
                    'On iteration {} of at most {} (loss = {}, loss change = {})'
                    .format(t, self.max_steps, loss, loss_change)
                )

            # Check for convergence if at least one step has already been run
            if len(loss_buffer) > 1:
                loss_change = np.mean(np.diff(loss_buffer))
                if abs(loss_change) < self.tol:
                    self.converged_ = True
                    if self.n_print_progress:
                        logging.info(
                            'Converged on iteration {} (loss = {}, loss change = {})'\
                            .format(t, loss, loss_change)
                        )

            loss_buffer.append(loss)

            if t % self.n_collect == 0 or t == self.max_steps - 1 or self.converged_:
                # Collect loss as well as all parameter values
                self.losses_.append(info_dict['loss'])

            if self.converged_:
                break

        # If convergence was not reached, either log a warning or throw an error depending on which was configured
        if not self.converged_:
            msg = 'Failed to reach convergence after {} steps.  '\
                'Consider increasing max_steps or altering learning rate / optimizer parameters'\
                .format(self.max_steps)
            if self.fail_if_not_converged:
                raise ConvergenceError(msg)
            else:
                logger.warning(msg)

        # Save model results if a log/event directory was set
        if self.log_dir is not None and self.save_tf_model:
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(self.log_dir, 'model.ckpt'))

        # Extract criticism arguments where first is always prediction function
        self.criticism_args_ = self.model.criticism_args(sess, self.tensor_map_)

        inference.finalize()
        sess.graph.finalize()
        return self

    def get_tensor(self, tensor):
        sess = ed.get_session()
        if not isinstance(tensor, tf.Tensor):
            tensor = self.tensor_map_[tensor]
        return sess.run(tensor)

    def get_all_tensors(self, filter_fn=None):
        sess = ed.get_session()
        tensor_values = {}
        for t in self.tensor_map_:
            if filter_fn is not None and not filter_fn(t):
                continue
            v = self.tensor_map_[t]
            if not isinstance(v, tf.Tensor):
                continue
            tensor_values[t] = sess.run(v)
        return tensor_values

    def predict(self, X, **kwargs):
        return self.criticism_args_[self.model.prediction_fn_key()](X)

