# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.bspline import get_design, diff_mat
from Python.Effect_cases import *


class AdaptiveHMC:
    # FIXME: FAIL OF SAMPLING ALWAYS SAME  VALUE MUST LIE IN HERE NOT XBETA!!
    """intended to hold the adaptive HMC sampler and to be inherited by a model class
    object, that has unnormalized_log_prob, bijectors and initial self attributes"""

    def __init__(self, initial, bijectors, log_prob, num_burnin_steps=int(1e3), num_leapfrog_steps=3):
        self.initial = initial
        bijected_hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob,
            num_leapfrog_steps=num_leapfrog_steps,
            step_size=1.),
            bijector=bijectors)

        self.adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            bijected_hmc,
            num_adaptation_steps=int(num_burnin_steps * 0.8))


        # self.adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        #     tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=log_prob,
        #         num_leapfrog_steps=3,
        #         step_size=1.),
        #     num_adaptation_steps=int(num_burnin_steps * 0.8))

    @tf.function
    def sample_chain(self, num_burnin_steps=int(1e3), num_results=int(10e3)):
        samples = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=self.initial,
            kernel=self.adaptive_hmc)
            #trace_fn=lambda _, pkr: pkr.inner_results.is_accepted ) # increases number of outputs

        self.samples = samples

        # sample_mean = tf.reduce_mean(samples, axis=0)
        # sample_stddev = tf.math.reduce_std(samples, axis=0)
        # print(sample_mean.numpy(), sample_stddev.numpy())
        return samples


class Xbeta(AdaptiveHMC):

    def __init__(self, xgrid, n, beta=[-1., 2.]):
        # data generation
        # MAKE SURE, all of the below is TF Ready

        self.n = n
        self.X = tf.stack([tf.ones((self.n,)), tfd.Uniform(xgrid[0], xgrid[1]).sample((self.n,))], axis=1)
        self.beta = tf.constant(beta)
        self.mu = tf.linalg.matvec(self.X, self.beta)
        self.y = self.mu + tfd.Normal(loc=0, scale=1).sample((self.n,))

        # attributes for sampler
        self.initial = tf.constant([1., 1.])  # CAREFULL MUST BE FLOAT!
        self.bijectors = [tfb.Identity(), tfb.Identity()]

        AdaptiveHMC.__init__(self, initial=self.initial, bijectors=self.bijectors, log_prob=self.unnormalized_log_prob)

    def unnormalized_log_prob(self, beta):
        """
        A closure, to keep X & y from data generating instance
        :param beta: float tensor
        :return: log-posterior value
        """

        def xbeta_log_prob(beta, X, y):
            # setting up priors
            prior_beta = tfd.Normal(loc=[0., 0.], scale=[1., 1.])

            # setting up likelihood model
            mu = tf.linalg.matvec(X, beta)
            likelihood = tfd.Normal(loc=mu, scale=1.)

            # return log
            return (prior_beta.log_prob(beta) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return xbeta_log_prob(beta, X=self.X, y=self.y)


class ZGamma(AdaptiveHMC, Effects1D):

    def __init__(self, xgrid):
        Effects1D.__init__(self, xgrid=xgrid)

        # data generation
        # FIXME MAKE SURE, all of the below is TF Ready
        self.X = tfd.Uniform(0, 10).sample((100,))  # FIXME
        self.Z = get_design(self.X, degree=2)
        # self.Bspline = Bspline_K(self.xgrid)
        self.mu = self.Bspline.spl(self.X)
        self.y = None  # FIXME

        # attributes for sampler
        self.initial = None
        self.bijectors = None

        AdaptiveHMC.__init__(self, initial=self.initial, bijectors=self.bijectors)

    def unnormalized_log_prob(self, gamma):
        def xbeta_log_prob(gamma=gamma, X=self.X, y=self.y):
            value = gamma
            return value

        return xbeta_log_prob(gamma)


if __name__ == '__main__':
    xbeta = Xbeta(xgrid=(0, 10), n=1000, beta=[-1., 2.])

    # log posterior of true beta
    xbeta.unnormalized_log_prob(beta=tf.constant([-1., 2.]))

    samples, tup = xbeta.sample_chain()

    # print(sample_mean.numpy(), sample_stddev.numpy())
    print(samples)
