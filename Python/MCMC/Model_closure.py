import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# from respective__init__.py
from Python.Data import *
from Python.MCMC import *


class model(AdaptiveHMC):
    xgrid = (0, 10)

    def __init__(self,
                 X=tf.stack(axis=1, values=[tf.ones((100,)), tfd.Uniform(xgrid[0], xgrid[1]).sample((100,))]),
                 beta=None, sigma=None,
                 prior_beta=tfd.Normal(loc=[0., 0.], scale=[1., 1.]),
                 prior_sigma=tfd.InverseGamma(0.001, 0.001),
                 initial=[tf.constant([1., -1.]), tf.constant([4.])],  # CAREFULL MUST BE FLOAT!
                 bijectors=[tfb.Identity(), tfb.Exp()]
                 ):
        """

        :param X: Data matrix
        :param beta: floating point tf.tensor e.g.: tf.constant([-1.,2.]) or
        None. In this case, a vector from the prior is drawn!
        :param sigma: floating point tf.tensor e.g.: tf.constant([-1.,2.]) or
        None. In this case, a vector from the prior is drawn!
        :param prior_beta: tfd, used for likelihood model
        :param prior_sigma: tfd, used for likelihood model
        :param initial: list of tf.constants
        :param bijectors:
        """

        # argument "parsing"

        self.X = X
        self.n = self.X.shape[0]

        if beta is None:
            self.beta = prior_beta.sample((1,))

        if sigma is not None:
            self.sigma = tf.constant(sigma)
        else:
            self.sigma = prior_sigma.sample((1,))

        # data generation
        y, mu = self.likelihood(self.X, self.beta, self.sigma)
        self.y = y
        self.mu = mu

        # setting up the estimation model
        self.unnormalized_log_prob = self._closure_log_prob(
            X=self.X, y=self.y,
            prior_beta=self.prior_beta,
            prior_sigma=self.prior_sigma)

        # setting up the estimator
        AdaptiveHMC.__init__(self,
                             initial=initial,
                             bijectors=bijectors,
                             log_prob=self.unnormalized_log_prob)

    def likelihood(self, X, beta, sigma):
        """

        :param X:
        :param beta:
        :param sigma:
        :return: (y, mu) with y a specific tfd getting its sample() & logprob method
        mu is the X @ beta
        """
        assert(X.shape[1] == beta.shape[0])
        # setting up the likelihood
        mu = tf.linalg.matvec(X, beta)
        # FIXME: do dim work or must i repeat sigma? also type assertation
        y = tfd.MultivariateNormalDiag(loc=self.mu, scale=sigma)
        # self.y = self.mu + tfd.Normal(loc=0, scale=sigma).sample((self.n,))
        return y, mu


    def _closure_log_prob(self, X, y, prior_beta, prior_sigma):
        """A closure, to set X, y & the priors in this model"""

        @tf.function
        def _unnormalized_log_prob(beta, sigma):
            # setting up likelihood model
            likelihood, mu = self.likelihood(X, beta, sigma)

            # return log posterior value
            return (prior_beta.log_prob(beta) +
                    prior_sigma.log_prob(sigma) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return _unnormalized_log_prob


    def predict(self, X, beta, sigma):
        """
        Predict y & mu from the provided values
        :param X:
        :param beta:
        :param sigma:
        :return:
        """
        n = X.shape[1]
        assert (n == beta.shape[0])
        y, mu = self.likelihood(X, beta, sigma)
        return mu, y.sample((n,))


if __name__ == '__main__':
    pass