
import collections

import edward as ed
from edward.models import Normal
from edward.stats import norm, bernoulli

from ml.tensorflow.utilities import tf_print
import tensorflow as tf
import numpy as np


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


class FeatureGroupModel(object):

    def __init__(self, feature_map):
        self.feature_map = feature_map

    def get_feature_index(self):
        return self.feature_map['index']

    def get_parameter_map(self):
        raise NotImplementedError()

    def get_parameter_values(self, sess):
        raise NotImplementedError()


class LinearGroupModel(FeatureGroupModel):

    def __init__(self, feature_map, priors):
        self.priors = priors
        self.prior_mu = [p['mu'] for p in priors]
        self.prior_sigma = [p['sigma'] for p in priors]
        super(LinearGroupModel, self).__init__(feature_map)

    def get_parameter_map(self):
        D = len(self.priors)
        w = Normal(
            mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D])))
        )
        return {'w': w}

    def get_prior_log_proba(self, X, Y, Z):
        w = tf_print(Z['w'], lambda x: 'Weights: {}'.format(x))
        lp = norm.logpdf(w, self.prior_mu, self.prior_sigma)
        return tf.reduce_sum(lp)

    def get_prediction(self, X, Y, Z):
        return ed.dot(X, Z['w'])

    def get_parameter_values(self, sess, Z):
        w = Z['w']
        w_mu, w_sigma = sess.run([w.mu, w.sigma])
        names = self.feature_map['names']
        w_mu = dict(zip(['{}:mu'.format(n) for n in names], list(w_mu)))
        w_sigma = dict(zip(['{}:sigma'.format(n) for n in names], list(w_sigma)))
        w_mu.update(w_sigma)
        return w_mu


class InterceptGroupModel(FeatureGroupModel):

    def __init__(self, prior):
        self.prior = prior
        feature_map = {'index': [0], 'names': ['Intercept']}
        super(InterceptGroupModel, self).__init__(feature_map)

    def get_parameter_map(self):
        b = Normal(
            mu=tf.Variable(tf.random_normal([])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([])))
        )
        return {'b': b}

    def get_prior_log_proba(self, X, Y, Z):
        b = tf_print(Z['b'], lambda x: 'Intercept: {}'.format(x))
        return norm.logpdf(b, self.prior['mu'], self.prior['sigma'])

    def get_prediction(self, X, Y, Z):
        n = int(X.get_shape()[0])
        return tf.fill([n], Z['b'])

    def get_parameter_values(self, sess, Z):
        b = Z['b']
        b_mu, b_sigma = sess.run([b.mu, b.sigma])
        return {'b:mu': b_mu, 'b:sigma': b_sigma}


class BayesianModel(object):

    def __init__(self, model_config, n_collect=1, n_print_progress=100, n_iter=500):
        self.model_config = model_config
        self.inv_link = tf.sigmoid
        self.n_collect = n_collect
        self.n_print_progress = n_print_progress
        self.n_iter = n_iter
        self.params_ = None
        self.sess_ = None
        self.inference_ = None

    def log_prob(self, xs, zs):
        X, Y = xs['X'], xs['Y']

        m_prior = {}
        m_pred = {}
        for g, m in self.model_config.items():
            x = _get_slice(X, m)

            # Select unobserved variables for this group and place them into a dict
            # keyed by the variable name with the group prefix removed
            zg = {k.replace(g + ':', ''): v for k, v in zs.items() if k.startswith(g + ':')}

            # Get contribution to prior log probability from this group
            m_prior[g] = m.get_prior_log_proba(x, Y, zg)

            # Get contribution to prediction equation from this group
            m_pred[g] = m.get_prediction(x, Y, zg)

        # Compute total log probability sum from priors
        lp_prior = _sum(list(m_prior.values()))
        #lp_prior = tf_print(lp_prior, lambda x: x)

        # Compute log probability for sum of predictions on link scale
        y_logit = tf.reduce_sum(tf.pack(list(m_pred.values()), axis=1), 1)
        #y_logit = tf_print(y_logit, lambda x: [np.min(x), np.max(x), np.all(np.isfinite(x))])

        y_proba = self.inv_link(y_logit)

        # Clip probability predictions to avoid log(0) in pmf calculation
        y_proba = tf.clip_by_value(y_proba, 1E-6, 1-1E-6)
        #y_proba = tf_print(y_proba, lambda x: [np.min(x), np.max(x), np.all(np.isfinite(x))])
        lp_data = tf.reduce_sum(bernoulli.logpmf(Y, p=y_proba))

        return lp_prior + lp_data

    def _init(self):
        if self.sess_ is not None:
            self.sess_.close()
            self.params_ = None
            self.sess_ = None
            self.inference_ = None

    def fit(self, X, y):
        self._init()

        model = {}
        for g, m in self.model_config.items():
            with tf.name_scope(g):
                model[g] = m.get_parameter_map()

        sess = ed.get_session()
        data = {'X': X, 'Y': y}

        inference = ed.MFVI(_flatten_map(model, sep=':'), data, self)
        inference.initialize(n_print=self.n_print_progress, n_iter=self.n_iter)

        init = tf.initialize_all_variables()
        init.run()

        self.params_ = []
        for t in range(inference.n_iter):
            info_dict = inference.update()
            inference.print_progress(info_dict)

            if t % inference.n_print == 0:
                print('On iteration {} of {}'.format(t, inference.n_iter))

            if t % self.n_collect == 0 or t == inference.n_iter - 1:
                params = {g: m.get_parameter_values(sess, model[g]) for g, m in self.model_config.items()}
                params = _flatten_map(params, sep=':')
                self.params_.append(params)

        self.sess_ = sess
        self.inference_ = inference
        return self

