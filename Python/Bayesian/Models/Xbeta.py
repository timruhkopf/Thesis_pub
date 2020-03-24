import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC


class Xbeta(AdaptiveHMC):

    def __init__(self, xgrid, n, beta=[-1., 2.], prior_beta=tfd.Normal(loc=[0., 0.], scale=[1., 1.])):
        # data generation
        # MAKE SURE, all of the below is TF Ready

        self.n = n
        self.X = tf.stack([tf.ones((self.n,)),
                           tfd.Uniform(xgrid[0], xgrid[1]).sample((self.n,))],
                          axis=1)
        self.beta = tf.constant(beta)
        self.prior_beta = prior_beta
        self.mu = tf.linalg.matvec(self.X, self.beta)
        self.y = self.mu + tfd.Normal(loc=0, scale=1).sample((self.n,))
        # TODO TRAIN TEST SPLIT?

        # attributes for sampler
        self.initial = tf.constant([1., 1.])  # CAREFULL MUST BE FLOAT!
        self.bijectors = tfb.Identity()  # [tfb.Identity(), tfb.Identity()]

        self.unnormalized_log_prob = self._closure_log_prob(
            self.X, self.y, self.prior_beta)

        AdaptiveHMC.__init__(self, initial=self.initial,
                             bijectors=self.bijectors,
                             log_prob=self.unnormalized_log_prob)

    def likelihood_model(self, X, beta):
        # setting up likelihood model
        mu_hat = tf.linalg.matvec(X, beta)
        y = tfd.Normal(loc=mu_hat, scale=1.)  # distributional assumption
        return y, mu_hat

    def _closure_log_prob(self, X, y, prior_beta):
        """A closure, to set X, y & the priors in this model"""

        @tf.function
        def xbeta_log_prob(beta):
            likelihood, _ = self.likelihood_model(X, beta)
            return (prior_beta.log_prob(beta) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return xbeta_log_prob


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    xbeta = Xbeta(xgrid=(0, 10), n=1000, beta=[-1., 2.])
    print(xbeta.unnormalized_log_prob(beta=tf.constant([-1., 2.])))
    samples, traced = xbeta.sample_chain(
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    # histogram of parameters
    is_accepted = xbeta.traced.inner_results.is_accepted
    samples = samples.numpy()

    plt.hist(samples[:, 0], bins=30, histtype="stepfilled")
    plt.title("Traces of unknown parameters")
    plt.hist(samples[:, 1], bins=30, histtype="stepfilled")
    plt.title("Traces of unknown parameters")
