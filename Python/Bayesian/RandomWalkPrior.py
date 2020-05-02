from Python.Effects.bspline import get_design, diff_mat1D

import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from numpy import pi

pi = tf.constant(pi)
tfd = tfp.distributions
tfb = tfp.bijectors


class RandomWalkPrior:
    def __init__(self, no_basis):
        """A RandomWalkPrior of degree one"""
        self.K = tf.convert_to_tensor(diff_mat1D(no_basis, order=1)[1], tf.float32)
        self.order = tf.cast(tf.linalg.matrix_rank(self.K), tf.float32)

        # for drawing only:
        self.cov = tf.cast(tf.cast(
            tf.linalg.inv(self.K[1:, 1:]), tf.float16), tf.float32)
        self.cov_cholesky = tf.linalg.cholesky(self.cov)

        # \gamma_0: use Uniform in relevant area
        # \gamma_{d-1} vector: multivariate normal with K[1:,1:]
        # TODO sample conditionally on Uniform W0 draw!
        self.joint = tfd.JointDistributionNamed(OrderedDict(
            # Spline model
            tau=tfd.InverseGamma(1., 1., name='tau'),
            W0=tfd.Uniform(-1., 1., name='w0'),
            W=lambda tau: tfd.MultivariateNormalTriL(
                loc=tf.repeat([0.], self.cov.shape[0]),
                scale_tril=tau * self.cov_cholesky,  # fixme: tau or tau **-1
                name='w')
        ))

        # Notice: the variable name W0 is hidden from public, to get a unique
        # interface for dense.
        self.parameters = list(self.joint._parameters['model'].keys())
        self.parameters.remove('W0')
        self.bijectors = {
            'tau': tfb.Exp(),
            'W': tfb.Identity()}

    def sample(self, *args, **kwargs):
        """sample from priors"""
        s = self.joint.sample(*args, **kwargs)
        W = tf.concat([tf.reshape(s['W0'], (1,)), s['W']], axis=0)
        del s['W0']
        s['W'] = W
        return s

    def prior_log_prob(self, param):
        # In contrast, when having a large variance \tau^2 , neighboring
        # coefficients are able to deviate from each other, which in turn leads
        # to a rough estimated function.
        # The proportionality sign (this is only proportional log prob
        # results from the flat prior for the first regression coefficient
        # \gamma_1 , which we did not specify exactly but only up to a
        # proportionality constant
        # flat prior corresponds to a constant vector representing the
        # level of function f . With the partially
        # improper prior, we define a flat prior for this level.
        # Although the joint prior distribution is partially improper,
        # we obtain a proper multivariate normal posterior distribution for \gamma.
        # parse input
        W, tau = param['W'], param['tau']

        constant = tf.math.log(2 * pi * tau) * self.order / 2
        Kw = tf.linalg.matvec(self.K, W)
        wKw = tf.reduce_sum(tf.multiply(W, Kw))
        kernel = (2 * tau) ** -1 * wKw
        return -constant - kernel

    def PLS_estimate(self, Z, y, lam):
        """estiamte the gamma hat under PLS
        the P-spline smoothing parameter is given by the ratio of the error
        variance and the variance of the RW1. This solidifies our previous
        conjectures regarding the influence of the variance tau^2 of the
        random walk and further leads to an interesting interpretation of the
        smoothing parameter \lambda: The larger the variance \tau² of the prior
        distribution is relative to the variance of the residuals \sigma²,
        the less the estimation will be penalized. Consequently, we always have
        to interpret the value of \tau² relative to the variance \sigma² that
        is associated with the measurement error ". We can then refer to \lambda
        as the noise-to-signal ratio."""
        ZZ = tf.matmul(Z, Z, transpose_a=True)
        Zy = tf.linalg.matmul(Z, y, transpose_a=True)
        return tf.reshape(tf.linalg.matmul(tf.linalg.inv(ZZ + lam * self.K), Zy), (self.K.shape[0],))


if __name__ == '__main__':
    # Setting it up
    no_basis = 20
    rw = RandomWalkPrior(no_basis)

    n = 200
    Z = tf.convert_to_tensor(
        get_design(tfd.Uniform(-10., 10.).sample(n).numpy(),
                   degree=2, no_basis=no_basis),
        tf.float32)

    # deprec: joint shall no longer be addressed immediately!
    param = rw.joint.sample()
    tau, W, W0 = param['tau'], param['W'], param['W0']
    gamma = tf.concat([tf.reshape(W0, (1,)), W], axis=0)

    mu = tf.linalg.matvec(Z, gamma)
    sigma = 1.
    lam = sigma / tau
    y = tfd.Normal(tf.reshape(mu, (Z.shape[0], 1)), sigma).sample()

    # test the functions
    pls = rw.PLS_estimate(Z, y, lam)

    print(gamma - pls)

    s = rw.sample()
    rw.prior_log_prob(s)
