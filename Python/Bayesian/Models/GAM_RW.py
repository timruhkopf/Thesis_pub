from Python.Effects.bspline import get_design, diff_mat1D
from Python.Bayesian.RandomWalkPrior import RandomWalkPrior

from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class GAM_RW:
    def __init__(self, no_basis, activation='identity'):
        """:param input_shape: is number of basis! = dim of gamma"""
        identity = lambda x: x
        self.activation = {
            'relu': tf.nn.relu,
            'tanh': tf.math.tanh,
            'sigmoid': tf.math.sigmoid,
            'identity': identity}[activation]

        self.rw = RandomWalkPrior(no_basis)
        self.parameters = self.rw.parameters + ['sigma']
        self.bijectors = self.rw.bijectors
        self.bijectors.update({'sigma': tfb.Exp()})
        self.bijectors_list = [self.bijectors[k] for k in self.parameters]

    @tf.function
    def dense(self, X, W):
        return self.activation(tf.linalg.matvec(X, W))

    def likelihood_model(self, Z, param):
        return tfd.JointDistributionNamed(OrderedDict(
            sigma=tfd.InverseGamma(1., 1.),
            y=lambda sigma: tfd.Sample(
                tfd.Normal(loc=self.dense(Z, param['W']), scale=sigma)))
        )

    def _closure_log_prob(self, X, y):
        # @tf.function
        def GAM_RW_log_prob(*tensorlist):
            param = {k: v for k, v in zip(self.parameters, tensorlist)}
            likelihood = self.likelihood_model(X, param)
            return tf.reduce_sum(likelihood.log_prob(y=y, sigma=param['sigma'])) + \
                   self.rw.prior_log_prob(param)

        return GAM_RW_log_prob

    # INITIALIZATION RELATED FUNCTIONS
    @tf.function
    def OLS(self, X, y):
        y = tf.reshape(y, (y.shape[0], 1))
        XXinv = tf.linalg.inv(tf.linalg.matmul(X, X, transpose_a=True))
        Xy = tf.linalg.matmul(X, y, transpose_a=True)
        ols = tf.linalg.matmul(XXinv, Xy)
        return tf.reshape(ols, (ols.shape[0],))

    def sample_model(self, Z):
        """

        :param Z:
        :return: tuple: (the parameter set that created y, y)
        """
        s = self.rw.sample()
        likelihood = self.likelihood_model(Z, s)
        d = likelihood.sample()
        s.update({'sigma': d['sigma']})

        return s, d['y']


if __name__ == '__main__':
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC
    from Python.Bayesian.plot1d import plot1d_functions

    no_basis = 20
    gam_rw = GAM_RW(no_basis=no_basis)

    # (0) SETTING UP THE DATA

    n = 200
    X = tfd.Uniform(-10., 10.).sample(n)
    Z = tf.convert_to_tensor(
        get_design(X.numpy(), degree=2, no_basis=no_basis),
        tf.float32)

    # true / init / ols_init
    true_param, y = gam_rw.sample_model(Z)
    init_param, _ = gam_rw.sample_model(Z)
    ols_param = init_param.copy()
    ols_param['W'] = gam_rw.OLS(Z, y)

    gam_rw.unnormalized_log_prob = gam_rw._closure_log_prob(Z, y)
    print(gam_rw.unnormalized_log_prob(*init_param.values()))
    print(gam_rw.unnormalized_log_prob(*true_param.values()))

    # look at functions
    f_true = gam_rw.dense(Z, true_param['W'])
    f_init = gam_rw.dense(Z, init_param['W'])
    f_ols = gam_rw.dense(Z, gam_rw.OLS(Z, y))
    plot1d_functions(X, y, **{'ols': f_ols, 'true': f_true, 'init': f_init})

    adHMC = AdaptiveHMC(
        initial=list(ols_param.values()),  # list(init_param.values()),
        bijectors=gam_rw.bijectors_list,
        log_prob=gam_rw.unnormalized_log_prob)

    # FIXME: sample_chain has no y
    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(1 * 10e2),
        num_results=int(10e2),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    print(adHMC.rate_accepted)

    # prediction
    meanPost = adHMC.predict_mean()
    modePost = adHMC.predict_mode(gam_rw.unnormalized_log_prob)

    # plotting
    f_true = gam_rw.dense(Z, true_param['W'])
    f_init = gam_rw.dense(Z, init_param['W'])
    f_mean = gam_rw.dense(Z, meanPost[1])
    f_mode = gam_rw.dense(Z, modePost[1])

    plot1d_functions(X, y, **{
        'ols': f_ols, 'true': f_true, 'init': f_init,
        'post. mean': f_mean, 'post. mode': f_mode})

    # SAMPLED FUNCTIONS:
    from random import sample

    # TODO FILTER FOR ACCEPTED ONLY!!
    paramsets = [s for s in zip(*samples)]
    plot1d_functions(
        X, y, true=f_true,
        **{str(i): gam_rw.dense(Z, W) for i, (tau, W, sigma) in enumerate(
            sample(paramsets, k=20))})

    # EMPIRICAL CONDITIONAL PREDICTIVE POSTERIOR:
    # CAREFULL this is not a joint distribution,
    # get conditional cdfs (cond. on x --> Z i.e. at x's position)
    # paramsets = [s for s in zip(*samples)]
    paramsets = [s for s, accepted in zip(
        zip(*samples), traced.inner_results.is_accepted.numpy()) if accepted]
    # predictive = [gam_rw.likelihood_model(Z, param={'W': W, 'sigma': sigma})  # FIXME: NASTY
    #               for tau, W, sigma in sample(paramsets, k=2)]

    # FIXME: cannot access y's conditional distribution directly, as the
    #  likelihood has a prior for sigma and in return, y's distribution is
    #  a lambda function predictive[0].parameters['model']['y']: be aware of:
    # predictive[0].parameters['model']['y'](sigma = 0.1)

    # FIXME: nasty! this is the explicit distribution taken from likelihood_model
    predictive = [tfd.Normal(loc=gam_rw.dense(Z, W), scale=sigma) for tau, W, sigma in sample(paramsets, k=100)]
    # predictive[0].sample(100)
    # not implemented
    # [dist.quantiles(0.9) for dist in predictive]

    # FIXME: .sample(100) does not work because of sigma being in likelihood!
    # predictive_sample = [y for sigma, y in [likelihood.sample(2)['y'] for likelihood in predictive]]

    # predictive_sample = tf.concat([likelihoody.sample(100) for likelihoody in predictive], axis=2)
    # deprec
    predictive_sample = tf.stack([likelihood.sample(100) for likelihood in predictive], axis=-1)
    # (no. obs, no.chains, no.likelihoodsamples)
    # not yet implemented method:
    # empiricals = tfp.distributions.Empirical(predictive_sample).quantile(0.5)
    quantiles = tfp.stats.quantiles(predictive_sample, num_quantiles=10, axis=0)
    ten, ninety = tf.reduce_mean(quantiles[1], axis=1), tf.reduce_mean(quantiles[9], axis=1)

    sortorder = tf.argsort(X).numpy()
    plot1d_functions(
        X, y, true=f_true, confidence={
            'x': X.numpy()[sortorder],
            'y1': ten.numpy()[sortorder],
            'y2': ninety.numpy()[sortorder]})


    # x = [0.,  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.]
    #
    # tfp.stats.quantiles(x, num_quantiles=10, interpolation='nearest')
