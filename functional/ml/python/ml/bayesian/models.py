import collections
import os
import edward as ed
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import logging_ops
from edward.models import Normal
from edward.stats import bernoulli
from py_utils import math as py_math
import logging
logger = logging.getLogger(__name__)


def _sum(items):
    # Remove shape if 1-item, 1-D tensor
    # * Note that this will fail if any item has more than one value (as should be expected)
    items = [tf.reshape(i, []) for i in items]
    return tf.reduce_sum(tf.pack(items))


def _get_slice(X, m):
    # Extract list of feature index numbers
    idx = m.get_feature_index()

    assert isinstance(idx, list), 'Feature index must be a list of integers (got "{}" instead)'.format(idx)
    assert np.all(np.array([isinstance(i, int) for i in idx]))

    # Extract feature values from observed data
    return tf.transpose(tf.gather(tf.transpose(X), idx))


def _flatten_map(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten_map(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_prior_lp(v, p):
    return p['dist'].logpdf(v, **p['params'])


class FeatureGroupModel(object):

    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.name = None

    def get_feature_index(self):
        return self.feature_map['index']

    def get_feature_names(self):
        return self.feature_map['names']

    def add_histogram_summary(self, v, name):
        logging_ops.histogram_summary("{}:{}".format(self.name, name), v)

    def initialize(self, X, Y):
        """Intialize/observe data prior to building inference tensors

         This is useful for building any per-group sampling distributions based
         on certain values present in the data itself (e.g. in hierarchical models)

        :param X: Observed data *array* (not a tensor yet)
        :param Y: Observed responses
        """
        pass

    def get_parameter_map(self, X, Y):
        """Get sampling distribution tensors for latent variables
        :param X: Observed data *array* (not a tensor yet)
        :param Y: Observed responses
        :return: Dict containing inference tensors keyed by string names
        """
        raise NotImplementedError()

    def get_parameter_values(self, sess, Z):
        """Get parameter values for latent variables
        :param sess: Tensorflow session used to fetch values in current graph
        :param Z: Latent variables
        :return: Dict containing parameter values (as floats, not tensors) keyed by name
        """
        raise NotImplementedError()

    def get_prior_log_proba(self, X, Y, Z):
        """Get log probability contribution to total likelihood due to priors
        :param X: Observed data *tensor*
        :param Y: Observed responses
        :param Z: Latent variables
        :return: Scalar float tensor containing log probability value
        """
        raise NotImplementedError()

    def get_prediction_tf(self, X, Z):
        """Get partial prediction on link scale from this group
        :param X: Observed data *tensor*
        :param Z: Latent variables
        :return: Tensor with shape [n_samples, n_outputs] (usually [N,])
        """
        raise NotImplementedError()

    def get_prediction_py(self, X, Z):
        """Get partial prediction on link scale from this group outside of Tensorflow graph

        This is useful for composing prediction formulae that execute much faster in pure python
        than they do within Tensorflow

        :param X: Observed data *array*
        :param Z: Latent variables (this will be the dictionary returned by "get_parameter_values")
        :return: Matrix with shape [n_samples, n_outputs] (usually [N,])
        """
        raise NotImplementedError()

    def set_name(self, name):
        self.name = name

    def get_name(self):
        assert self.name, 'Name has not been set yet'
        return self.name


class LinearGroupModel(FeatureGroupModel):

    def __init__(self, feature_map, priors):
        self.priors = priors
        super(LinearGroupModel, self).__init__(feature_map)

    def get_parameter_map(self, X, Y):
        D = len(self.feature_map['index'])
        w = Normal(
            mu=tf.Variable(self.priors['params']['mu']),
            sigma=tf.nn.softplus(tf.Variable(self.priors['params']['sigma']))
        )
        return {'w': w}

    def get_prior_log_proba(self, X, Y, Z):
        # w = tf_print(Z['w'], lambda x: 'Weights: {}'.format(x))
        w = Z['w']
        # self.add_histogram_summary(w, 'w')
        lp = _get_prior_lp(w, self.priors)
        return tf.reduce_sum(lp)

    def get_parameter_values(self, sess, Z):
        w = Z['w']
        w_mu, w_sigma = sess.run([w.mu, w.sigma])
        names = self.feature_map['names']
        w_mu = dict(zip(['l:{}:mu'.format(n) for n in names], list(w_mu)))
        w_sigma = dict(zip(['l:{}:sigma'.format(n) for n in names], list(w_sigma)))
        w_mu.update(w_sigma)
        return w_mu

    def get_prediction_tf(self, X, Z):
        return ed.dot(X, Z['w'])

    def get_prediction_py(self, X, Z):
        names = self.feature_map['names']
        w = np.array([Z['l:{}:mu'.format(name)] for name in names])
        return np.matmul(X, w)


class SquareGroupModel(FeatureGroupModel):

    def __init__(self, feature_map, linear_prior, square_prior):
        self.square_prior = square_prior
        self.linear_prior = linear_prior
        super(SquareGroupModel, self).__init__(feature_map)

    def get_parameter_map(self, X, Y):
        D = len(self.feature_map['index'])
        w1 = Normal(
            mu=tf.Variable(self.linear_prior['params']['mu']),
            sigma=tf.nn.softplus(tf.Variable(self.linear_prior['params']['sigma']))
        )
        w2 = Normal(
            mu=tf.Variable(self.square_prior['params']['mu']),
            sigma=tf.nn.softplus(tf.Variable(self.square_prior['params']['sigma']))
        )
        return {'w1': w1, 'w2': w2}

    def get_prior_log_proba(self, X, Y, Z):
        w1, w2 = Z['w1'], Z['w2']
        lp1 = _get_prior_lp(w1, self.linear_prior)
        lp2 = _get_prior_lp(w2, self.square_prior)
        return tf.reduce_sum(lp1) + tf.reduce_sum(lp2)

    def get_parameter_values(self, sess, Z):
        w1, w2 = Z['w1'], Z['w2']
        w1_mu, w1_sigma = sess.run([w1.mu, w1.sigma])
        w2_mu, w2_sigma = sess.run([w2.mu, w2.sigma])
        names = self.feature_map['names']
        p = {}
        p.update(dict(zip(['l:{}:mu'.format(n) for n in names], list(w1_mu))))
        p.update(dict(zip(['l:{}:sigma'.format(n) for n in names], list(w1_sigma))))
        p.update(dict(zip(['s:{}:mu'.format(n) for n in names], list(w2_mu))))
        p.update(dict(zip(['s:{}:sigma'.format(n) for n in names], list(w2_sigma))))
        return p

    def get_prediction_tf(self, X, Z):
        return ed.dot(X, Z['w1']) + ed.dot(X**2, Z['w2'])

    def get_prediction_py(self, X, Z):
        names = self.feature_map['names']
        w1 = np.array([Z['l:{}:mu'.format(name)] for name in names])
        w2 = np.array([Z['s:{}:mu'.format(name)] for name in names])
        return np.matmul(X, w1) + np.matmul(X**2, w2)


class SigmoidGroupModel(FeatureGroupModel):

    def __init__(self, feature_map, linear_priors, sigmoid_prior, bias_prior):
        self.linear_priors = linear_priors
        self.sigmoid_prior = sigmoid_prior
        self.bias_prior = bias_prior

        super(SigmoidGroupModel, self).__init__(feature_map)

    def get_parameter_map(self, X, Y):
        D = len(self.feature_map['index'])
        w = Normal(
            mu=tf.Variable(self.linear_priors['params']['mu']),
            sigma=tf.nn.softplus(tf.Variable(self.linear_priors['params']['sigma']))
        )
        b = Normal(
            mu=tf.Variable(self.bias_prior['params']['mu']),
            sigma=tf.nn.softplus(tf.Variable(self.bias_prior['params']['sigma']))
        )
        s = Normal(
            mu=tf.Variable(self.sigmoid_prior['params']['mu']),
            sigma=tf.nn.softplus(tf.Variable(self.sigmoid_prior['params']['sigma']))
        )
        return {'w': w, 's': s, 'b': b}

    def get_prior_log_proba(self, X, Y, Z):
        w, s, b = Z['w'], Z['s'], Z['b']
        lpw = _get_prior_lp(w, self.linear_priors)
        lps = _get_prior_lp(s, self.sigmoid_prior)
        lpb = _get_prior_lp(b, self.bias_prior)
        return tf.reduce_sum(tf.concat(0, [lpw, tf.reshape(lps, [1]), tf.reshape(lpb, [1])]))

    def get_parameter_values(self, sess, Z):
        w, s, b = Z['w'], Z['s'], Z['b']
        w_mu, w_sigma = sess.run([w.mu, w.sigma])
        s_mu, s_sigma = sess.run([s.mu, s.sigma])
        b_mu, b_sigma = sess.run([b.mu, b.sigma])
        names = self.feature_map['names']

        p = {}
        p.update(dict(zip(['l:{}:mu'.format(n) for n in names], list(w_mu))))
        p.update(dict(zip(['l:{}:sigma'.format(n) for n in names], list(w_sigma))))
        p.update({'s:mu': s_mu, 's:sigma': s_sigma})
        p.update({'b:mu': b_mu, 'b:sigma': b_sigma})
        return p

    def get_prediction_tf(self, X, Z):
        w, s, b = Z['w'], Z['s'], Z['b']

        return s * tf.tanh(ed.dot(X, w) + b)

    def get_prediction_py(self, X, Z):
        names = self.feature_map['names']
        w = np.array([Z['l:{}:mu'.format(name)] for name in names])
        s = Z['s:mu']
        b = Z['b:mu']
        return s * np.tanh(np.matmul(X, w) + b)


class InterceptGroupModel(FeatureGroupModel):

    def __init__(self, prior):
        self.prior = prior
        feature_map = {'index': [0], 'names': ['__Intercept__']}
        super(InterceptGroupModel, self).__init__(feature_map)

    def get_feature_names(self):
        return []

    def get_parameter_map(self, X, Y):
        b = Normal(
            mu=tf.Variable(self.prior['params']['mu']),
            sigma=tf.nn.softplus(tf.Variable(self.prior['params']['sigma']))
        )
        return {'b': b}

    def get_prior_log_proba(self, X, Y, Z):
        # b = tf_print(Z['b'], lambda x: 'Intercept: {}'.format(x))
        b = Z['b']
        return _get_prior_lp(b, self.prior)

    def get_prediction_tf(self, X, Z):
        n = int(X.get_shape()[0])
        return tf.fill([n], Z['b'])

    def get_parameter_values(self, sess, Z):
        b = Z['b']
        b_mu, b_sigma = sess.run([b.mu, b.sigma])
        return {'b:mu': b_mu, 'b:sigma': b_sigma}

    def get_prediction_py(self, X, Z):
        return np.repeat(Z['b:mu'], X.shape[0])


class ConvergenceError(Exception):
    pass


class BayesianModel(object):

    def __init__(self, model_config, optimizer_fn=None, n_collect=1,
                 n_print_progress=None, max_steps=1000, tol=1e-1,
                 random_state=None, fail_if_not_converged=True,
                 n_samples=1, n_loss_buffer=10, save_tf_model=True):

        self.max_steps = max_steps
        self.tol = tol
        self.fail_if_not_converged = fail_if_not_converged
        self.n_samples = n_samples
        self.n_loss_buffer = n_loss_buffer
        self.save_tf_model = save_tf_model
        assert n_loss_buffer > 1, 'Loss buffer size must be > 1'
        assert max_steps > n_loss_buffer, 'Max steps must be > n_loss_buffer'

        self.model_config = model_config
        self.n_collect = n_collect
        self.n_print_progress = n_print_progress
        self.optimizer_fn = optimizer_fn
        self.random_state = random_state
        self.log_dir = None

        self.params_ = None
        self.losses_ = None
        self.converged_ = False

        self.inv_link_tf = tf.sigmoid
        self.inv_link_py = py_math.sigmoid

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        return self

    def log_prob(self, xs, zs):
        X, Y = xs['X'], xs['Y']

        m_prior = {}
        m_pred = {}
        for g, m in self.model_config.items():
            x = _get_slice(X, m)

            # Select unobserved variables for this group and place them into a dict
            # keyed by the variable name with the group prefix removed
            z = {k.replace(g + ':', ''): v for k, v in zs.items() if k.startswith(g + ':')}

            # Get contribution to prior log probability from this group
            m_prior[g] = m.get_prior_log_proba(x, Y, z)

            # Get contribution to prediction equation from this group
            m_pred[g] = m.get_prediction_tf(x, z)

        # Compute total log probability sum from priors
        lp_prior = _sum(list(m_prior.values()))
        # lp_prior = tf_print(lp_prior, lambda x: x)

        # Compute log probability for sum of predictions on link scale
        y_logit = tf.reduce_sum(tf.pack(list(m_pred.values()), axis=1), 1)
        # y_logit = tf_print(y_logit, lambda x: [np.min(x), np.max(x), np.all(np.isfinite(x))])

        y_proba = self.inv_link_tf(y_logit)

        # Clip probability predictions to avoid log(0) in pmf calculation
        y_proba = tf.clip_by_value(y_proba, 1E-6, 1-1E-6)
        # y_proba = tf_print(y_proba, lambda x: [np.min(x), np.max(x), np.all(np.isfinite(x))])
        lp_data = tf.reduce_sum(bernoulli.logpmf(Y, p=y_proba))

        return lp_prior + lp_data

    def _init(self):
        ed.get_session().close()
        tf.reset_default_graph()
        self.params_ = None
        self.losses_ = None

    def fit(self, X, y, **kwargs):
        self._init()
        ed.set_seed(self.random_state)

        model = {}
        for g, m in self.model_config.items():
            # Set the name associated with this group
            # (this is useful for naming tensors and should only be done once)
            m.set_name(g)

            # Slice observed data to only fields relevant to this group
            x = X[:, m.get_feature_index()]

            # Initialize the feature group model and retrieve parameters
            # to be optimized for group
            with tf.name_scope(g):
                m.initialize(x, y)
                model[g] = m.get_parameter_map(x, y)

        sess = ed.get_session()
        data = {'X': X, 'Y': y}

        # Initialize inference engine
        inference = ed.MFVI(_flatten_map(model, sep=':'), data, self)
        optimizer = self.optimizer_fn() if self.optimizer_fn else None
        inference.initialize(optimizer=optimizer, n_print=self.n_print_progress, n_samples=self.n_samples)

        # It would be much better if inference instances exposed the ability to set the log dir
        # directly but at the moment it's only set in inference.run, and this is how it's used:
        if self.log_dir is not None:
            summary_writer = tf.train.SummaryWriter(self.log_dir, tf.get_default_graph())
            # inference.train_writer = tf.train.SummaryWriter(self.log_dir, tf.get_default_graph())

        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        init.run()

        self.params_ = []
        self.losses_ = []

        self.converged_ = False
        loss_buffer = collections.deque(maxlen=self.n_loss_buffer + 1)
        loss_change = None
        for t in range(self.max_steps):
            info_dict = inference.update()
            loss = info_dict['loss']

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
                # Collect and write out any summary ops
                if self.log_dir is not None and summary_op is not None:
                    summary_writer.add_summary(sess.run([summary_op])[0], t)

                # Collect loss as well as all parameter values
                self.losses_.append(info_dict['loss'])
                params = {g: m.get_parameter_values(sess, model[g]) for g, m in self.model_config.items()}
                self.params_.append(params)

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

        sess.graph.finalize()
        return self

    def get_unique_feature_names(self):
        feats = []
        for m in self.model_config.values():
            feats.extend(m.get_feature_names())
        return list(set(feats))

    def get_fit_model_parameters(self):
        assert self.params_, 'Fit must be called before getting model parameters'
        return [_flatten_map(x, sep=':') for x in self.params_]

    def predict(self, X):
        assert self.params_, 'Fit must be called before predict'
        params = self.params_[-1]

        y_logit = np.repeat(0., X.shape[0])
        for g, m in self.model_config.items():
            x = X[:, m.get_feature_index()]
            y_logit += m.get_prediction_py(x, params[g])
        y_proba = self.inv_link_py(y_logit)
        return y_proba

