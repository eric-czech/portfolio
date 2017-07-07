

import numpy as np
import pystan
from theano import shared
import pymc3 as pm
from sklearn.base import BaseEstimator
import logging
logger = logging.getLogger(__name__)


class PYMCModel(object):

    def __init__(self, seed=None, n_chain=1, sample_kwargs=None):
        self.seed = seed
        self.sample_kwargs = {} if sample_kwargs is None else sample_kwargs
        self.n_chain = n_chain

    def get_model(self, X, Y, **kwargs):
        """
        Establish shared Theano variables and PYMC Model
        """
        raise NotImplementedError()

    def _add_step(self, args):
        """ Add step argument for sampling

        This is likely to be overriden so it is the only sampling argument added separately like this
        """
        start = pm.find_MAP()
        args['step'] = pm.NUTS(scaling=start)
        return args

    def get_posterior(self, model, **kwargs):
        """
        Run MCMC
        """
        if self.n_chain <= 1:
            args = dict(self.sample_kwargs)
            args['random_seed'] = self.seed
            with model:
                # args['step'] = pm.NUTS()
                # args['start'] = pm.find_MAP()
                # args['step'] = pm.NUTS(scaling=start)
                # args['step'] = pm.Metropolis()
                # args['init'] = 'advi_map'
                post = pm.sample(**args)
                return post

        # ### If n_chain > 1, assume that manual chain creation is desired
        # * See: https://stackoverflow.com/questions/27446738/how-to-sample-multiple-chains-in-pymc3
        # Note that in this case njobs will always be set to 1

        rs = np.random.RandomState(self.seed)
        seeds = rs.randint(1, int(1E9), size=self.n_chain)
        # seeds = np.repeat(1, self.n_chain)

        # with model:
        #     step = pm.NUTS()

        for k in ['random_seed', 'chain', 'step', 'njobs']:
            assert k not in self.sample_kwargs, 'Sampling arguments cannot contain key "{}"'.format(k)

        with model:
            traces = []
            for i in range(self.n_chain):
                logger.info('Running sampler for chain {} (seed = {})...'.format(i+1, seeds[i]))
                args = dict(self.sample_kwargs)
                args['random_seed'] = seeds[i]
                args['chain'] = i
                args['njobs'] = 1
                args = self._add_step(args)
                traces.append(pm.sample(**args))
            post = pm.backends.base.merge_traces(traces)
            return post

    def get_posterior_predictive(self, model, posterior, X, **kwargs):
        """
        Generate posterior predictive samples
        """
        raise NotImplementedError()


class PYMCEstimator(BaseEstimator):
    """
    A new sklearn estimator class derived for use with pystan.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        self.pymc_model_ = None
        self.posterior_ = None
        self.stats_ = None

    def fit(self, X, y, **kwargs):
        """
        Fits the estimator based on covariates X and observations y.
        """
        self.pymc_model_ = self.model.get_model(X, y, **kwargs)
        self.posterior_ = self.model.get_posterior(self.pymc_model_, **kwargs)
        return self

    def predict_posterior(self, X, **kwargs):
        """
        Generates a prediction based on X, the array of covariates.
        """
        return self.model.get_posterior_predictive(self.pymc_model_, self.posterior_, X, **kwargs)

    def predict(self, X, y_name='y', **kwargs):
        """
        Generates a prediction based on X, the array of covariates.
        """
        pp = self.predict_posterior(X, **kwargs)
        return pp[y_name].mean(axis=0)

    def posterior_summary(self, **kwargs):
        return pm.summary(self.posterior_, **kwargs)

    def posterior_rhat(self):
        return pm.gelman_rubin(self.posterior_)

    def posterior_effn(self):
        return pm.effective_n(self.posterior_)

    def posterior_traceplot(self, **kwargs):
        return pm.traceplot(self.posterior_, **kwargs)

    def posterior_forestplot(self, **kwargs):
        return pm.forestplot(self.posterior_, **kwargs)

    def posterior_plot(self, **kwargs):
        return pm.plot_posterior(self.posterior_, **kwargs)

    def compute_stats(self):
        self.stats_ = {
            'waic': pm.stats.waic(self.posterior_, model=self.pymc_model_),
            'dic': pm.stats.dic(self.posterior_, model=self.pymc_model_)
        }
        return self.stats_
