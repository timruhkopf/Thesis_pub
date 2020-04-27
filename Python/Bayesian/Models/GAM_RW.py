from Python.Effects.bspline import get_design, diff_mat1D
# from Python.Bayesian.Models.Regression import Regression
from Python.Bayesian.RandomWalkPrior import RandomWalkPrior

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class GAM_RW:
    def __init__(self, no_basis, no_units=1, activation='identity', *args, **kwargs):
        """:param input_shape: is number of basis! = dim of gamma"""
        self.rw = RandomWalkPrior(no_basis)
        self.prior_sigma = tfd.InverseGamma(1., 1.)

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
        # W = tf.concat([tf.reshape(W0, (1,)), W], axis=0)
        return tfd.Sample(tfd.Normal(
            loc=self.dense(Z, W),  # mu
            scale=sigma, name='y'),
            sample_shape=1)

    def _closure_log_prob(self, X, y):
        def GAM_RW_log_prob(W, sigma, tau):
            likelihood = self.likelihood_model(X, W, sigma)
            return likelihood.log_prob(y) + \
                   self.rw.log_prob(gamma=W, tau=tau) + \
                   self.prior_sigma.log_prob(sigma)

        return GAM_RW_log_prob

    @tf.function
    def dense(self, X, W):
        return self.activation(tf.linalg.matvec(X, W))


if __name__ == '__main__':
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC
    no_basis = 20
    gam_rw = GAM_RW(no_basis=no_basis)
    s_true = gam_rw.sample()

    n = 200
    Z = tf.convert_to_tensor(
        get_design(tfd.Uniform(-10., 10.).sample(n).numpy(),
                   degree=2, no_basis=no_basis),
        tf.float32)

    gam_rw.likelihood_model(Z, s_true['W'])

    s_init = gam_rw.sample()




