from Python.Effects.bspline import get_design, diff_mat1D
from Python.Bayesian.RandomWalkPrior import RandomWalkPrior

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class GAM_RW:
    def __init__(self, no_basis, activation='identity'):
        """:param input_shape: is number of basis! = dim of gamma"""
        self.rw = RandomWalkPrior(no_basis)
        self.prior_sigma = tfd.InverseGamma(1., 1., name='sigma')

        self.bijectors = {'W': tfb.Identity(),
                          'sigma': tfb.Exp(),
                          'tau': tfb.Exp()}

        identity = lambda x: x
        self.activation = {
            'relu': tf.nn.relu,
            'tanh': tf.math.tanh,
            'sigmoid': tf.math.sigmoid,
            'identity': identity}[activation]

    def sample(self):
        s = self.rw.sample()
        s['sigma'] = self.prior_sigma.sample()
        return s

    def likelihood_model(self, Z, W, sigma):
        """W = tf.concat([tf.reshape(W0, (1,)), W], axis=0)"""
        return tfd.Sample(tfd.Normal(
            loc=self.dense(Z, W),  # mu
            scale=sigma, name='y'),
            sample_shape=1)

    def _closure_log_prob(self, X, y):
        @tf.function
        def GAM_RW_log_prob(tau, W, sigma):  # precise arg ordering as sample!
            likelihood = self.likelihood_model(X, W, sigma)
            return tf.reduce_sum(likelihood.log_prob(y)) + \
                   self.rw.log_prob(gamma=W, tau=tau) + \
                   self.prior_sigma.log_prob(sigma)

        return GAM_RW_log_prob

    @tf.function
    def dense(self, X, W):
        return self.activation(tf.linalg.matvec(X, W))

    @tf.function
    def OLS(self, X, y):
        XXinv = tf.linalg.inv(tf.linalg.matmul(X, X, transpose_a=True))
        Xy = tf.linalg.matmul(X, y, transpose_a=True)
        ols = tf.linalg.matmul(XXinv, Xy)
        return tf.reshape(ols, (ols.shape[0],))


if __name__ == '__main__':
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC
    from Python.Bayesian.plot1d import plot1d_functions

    no_basis = 20
    gam_rw = GAM_RW(no_basis=no_basis)

    # (0) SETTING UP THE DATA
    true_param = gam_rw.sample()

    n = 200
    X = tfd.Uniform(-10., 10.).sample(n)
    Z = tf.convert_to_tensor(
        get_design(X.numpy(), degree=2, no_basis=no_basis),
        tf.float32)

    likelihood = gam_rw.likelihood_model(Z, true_param['W'], true_param['sigma'])
    y = likelihood.sample()

    # (1) SETTING UP THE ESTIMATION
    init_param = gam_rw.sample()
    gam_rw.unnormalized_log_prob = gam_rw._closure_log_prob(Z, y)

    print(gam_rw.unnormalized_log_prob(**init_param))
    print(gam_rw.unnormalized_log_prob(**true_param))

    # look at functions
    f_true = gam_rw.dense(Z, true_param['W'])
    f_init = gam_rw.dense(Z, init_param['W'])
    f_ols = gam_rw.dense(Z, gam_rw.OLS(Z, y))
    plot1d_functions(X, y, **{'ols': f_ols, 'true': f_true, 'init': f_init})

    # ols_init:
    ols_param = init_param
    ols_param['W'] = gam_rw.OLS(Z, y)

    adHMC = AdaptiveHMC(
        initial=list(ols_param.values()),  # list(init_param.values()),
        bijectors=[gam_rw.bijectors[k] for k in init_param.keys()],
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

    # empirical posterior predictive:
    # get conditional cdfs (cond. on x --> Z i.e. at x's position)
    paramsets = [s for s in zip(*samples)]
    predictive = [gam_rw.likelihood_model(Z, W, sigma)
                  for tau, W, sigma in sample(paramsets, k=102)]

    predictive_sample = tf.concat([likelihood.sample(100) for likelihood in predictive], axis=2)
    # (no. obs, no.chains, no.likelihoodsamples)
    # not yet implemented method:
    # empiricals = tfp.distributions.Empirical(predictive_sample).quantile(0.5)
    quantiles = tfp.stats.quantiles(predictive_sample, num_quantiles=10, axis=0)
    ten, ninety = tf.reduce_mean(quantiles[1], axis=1), tf.reduce_mean(quantiles[9], axis=1)

    sortorder = tf.argsort(X).numpy()
    plot1d_functions(
    X, y, true = f_true, confidence = {
        'x': X.numpy()[sortorder],
        'y1': ten.numpy()[sortorder],
        'y2': ninety.numpy()[sortorder]})

    #
    # x = [0.,  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.]
    #
    # tfp.stats.quantiles(x, num_quantiles=10, interpolation='nearest')
