import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC


class Xbeta:
    def __init__(self, p):
        """
        linear regression with fixed, homoscedastic sigma.
        :param p: number of parameters
        """
        self.p = p
        self.prior_beta = tfd.Normal(loc=tf.repeat(0., p), scale=tf.repeat(10., p))

    def _initialize_from_prior(self):
        self.beta = self.prior_beta.sample()

    def likelihood_model(self, X, beta):
        mu_hat = tf.linalg.matvec(X, beta)
        y = tfd.Normal(loc=mu_hat, scale=1.)  # distributional assumption
        return y, mu_hat

    def _closure_log_prob(self, X, y):
        """A closure, to set X, y & the priors in this model"""

        @tf.function
        def xbeta_log_prob(beta):
            likelihood, _ = self.likelihood_model(X, beta)
            return (self.prior_beta.log_prob(beta) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return xbeta_log_prob


class Data_Xbeta:
    def __init__(self, xgrid, n, beta):
        """plain linear regression data with unit & homoscedastic variance"""
        self.n = n
        self.X = tf.stack(
            values=[tf.ones((self.n,)),
                    tfd.Uniform(xgrid[0], xgrid[1]).sample((self.n,))],
            axis=1)

        self.beta = tf.constant(beta)
        self.y = self.true_likelihood().sample((self.n,))

    def true_likelihood(self, X, beta):
        self.mu = tf.linalg.matvec(X, beta)
        y = tfd.MultivariateNormalDiag(
            loc=self.mu,
            scale_diag=tf.repeat(1., self.mu.shape[0]))
        return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # (0) (generating data) ------------------------------
    # beta = tfd.Normal(loc=[0., 0.], scale=[1., 1.]).sample((1,))
    data = Data(xgrid=(0, 10), n=100, beta=[-1., 2.])

    # TODO TRAIN TEST SPLIT?

    # (1) (generating the model) ------------------------
    xbeta = Xbeta(p=2)
    xbeta.unnormalized_log_prob = xbeta._closure_log_prob(data.X, data.y)
    print(xbeta.unnormalized_log_prob(beta=tf.constant([-1., 2.])))

    xbeta._initialize_from_prior()

    # (2) (sampling the model) --------------------------
    adHMC = AdaptiveHMC(initial=xbeta.beta,  # tf.constant([1., 1.]) # CAREFULL MUST BE FLOAT!
                        bijectors=tfb.Identity(),  # [tfb.Identity(), tfb.Identity()]
                        log_prob=xbeta.unnormalized_log_prob)

    samples, traced = adHMC.sample_chain(num_burnin_steps=int(1e3), num_results=int(10e2),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    # (3) (evaluate the model) --------------------------
    # histogram of parameters
    is_accepted = traced.inner_results.is_accepted
    samples = samples.numpy()

    plt.hist(samples[:, 0], bins=30, histtype="stepfilled")
    plt.title("Traces of unknown parameters")
    plt.hist(samples[:, 1], bins=30, histtype="stepfilled")
    plt.title("Traces of unknown parameters")
