import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Bayesian.Samplers import AdaptiveHMC


class Xbetasigma(AdaptiveHMC):
    """ X @ \beta Regression with sigma prior on y, to check out """

    def __init__(self, xgrid, n, beta=[-1., 2.], sigma=[10.]):
        # data generation
        self.n = n
        self.X = tf.stack([tf.ones((self.n,)),
                           tfd.Uniform(xgrid[0], xgrid[1]).sample((self.n,))],
                          axis=1)
        self.beta = tf.constant(beta)
        self.sigma = tf.constant(sigma)
        self.mu = tf.linalg.matvec(self.X, self.beta)
        self.y = self.mu + tfd.Normal(loc=0, scale=self.sigma).sample((self.n,))

        # attributes for sampler
        self.initial = [tf.constant([1., -1.]), tf.constant([4.])]  # CAREFULL MUST BE FLOAT!
        self.bijectors = [tfb.Identity(), tfb.Exp()]
        # TODO look at traced[0] (TransformedTransitionKernelResults(transformed_state,...))

        AdaptiveHMC.__init__(self,
                             initial=self.initial,
                             bijectors=self.bijectors,
                             log_prob=self.unnormalized_log_prob)

    def unnormalized_log_prob(self, beta, sigma):
        """
        A closure, to keep X & y from data generating instance
        :param beta: float tensor
        :return: log-posterior value
        """

        def xbetasigma_log_prob(beta, sigma, X, y):
            # setting up priors
            prior_beta = tfd.Normal(loc=[0., 0.], scale=[1., 1.])
            prior_sigma = tfd.InverseGamma(0.001, 0.001)

            # setting up likelihood model
            mu = tf.linalg.matvec(X, beta)
            likelihood = tfd.Normal(loc=mu, scale=sigma)

            # return log posterior value
            return (prior_beta.log_prob(beta) +
                    prior_sigma.log_prob(sigma) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return xbetasigma_log_prob(beta, sigma, X=self.X, y=self.y)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    logdir = '/home/tim/PycharmProjects/Thesis/TFResults'

    # # (XBETA SIGMA CASE) -------------------------------------------------------
    xbetasigma = Xbetasigma(xgrid=(0, 10), n=1000, beta=[-1., 2.], sigma=[10.])
    print(xbetasigma.unnormalized_log_prob(beta=tf.constant([-1., 2.]), sigma=tf.constant([10.])))
    samples, traced = xbetasigma.sample_chain(num_burnin_steps=int(1e1), num_results=int(10e1), logdir=logdir,)
