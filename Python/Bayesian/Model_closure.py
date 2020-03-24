import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# from respective__init__.py
from Python.Data import *
from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC


class Xbetasigma(AdaptiveHMC):
    xgrid = (0, 10)

    def __init__(self,
                 X=tf.stack(axis=1, values=[tf.ones((100,)), tfd.Uniform(xgrid[0], xgrid[1]).sample((100,))]),
                 beta=None, sigma=None,
                 prior_beta=tfd.Normal(loc=[0., 0.], scale=[1., 1.]),
                 prior_sigma=tfd.InverseGamma(0.001, 0.001),
                 initial=[tf.constant([1., -1.]), tf.constant([4.])],  # CAREFULL MUST BE FLOAT!
                 bijectors=[tfb.Identity(), tfb.Exp()]):
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

        # (0) argument "parsing"
        if beta is None:
            beta = prior_beta.sample((1,))

        if sigma is not None:
            sigma = prior_sigma.sample((1,))

        print('init: beta: {}, \n sigma {}'.format(beta, sigma))

        self.X = X
        self.n = self.X.shape[0]
        self.beta = beta
        self.sigma = sigma
        self.prior_beta = prior_beta
        self.prior_sigma = prior_sigma

        # (1)  data generation
        y, mu = self.likelihood_model(self.X, self.beta, self.sigma)
        self.tfd_y = y
        self.y = y.sample((self.n,))
        self.mu = mu

        # (2) setting up the estimation model
        self.unnormalized_log_prob = self._closure_log_prob(
            X=self.X, y=self.y,
            prior_beta=self.prior_beta,
            prior_sigma=self.prior_sigma)

        # setting up the estimator
        AdaptiveHMC.__init__(self,
                             initial=initial,
                             bijectors=bijectors,
                             log_prob=self.unnormalized_log_prob)

    # def priors(self):
    #     []

    def likelihood_model(self, X, beta, sigma):
        """
        :return: (y, mu) with y a specific tfd getting its sample() & logprob method
        mu is the X @ beta
        """
        assert (X.shape[1] == beta.shape[0])

        # setting up the likelihood
        mu = tf.linalg.matvec(X, beta)
        if sigma.shape != mu.shape:
            sigma = tf.repeat([sigma], mu.shape[0])

        y = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return y, mu

    def _closure_log_prob(self, X, y, prior_beta, prior_sigma):
        """A closure, to set X, y & the priors in this model"""

        #@tf.function
        def _unnormalized_log_prob(beta, sigma):
            # setting up likelihood model
            print('beta: {},\n sigma {}'.format(beta, sigma))
            likelihood, _ = self.likelihood_model(X, beta, sigma)   # FIXME: second iteration sigma is a string tensor

            # return log posterior value
            return (tf.reduce_sum(prior_beta.log_prob(beta)) +
                    prior_sigma.log_prob(sigma) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return _unnormalized_log_prob

    def predict(self, X, beta, *args):
        """
        Predict y & mu from the provided values
        :return: (mu, y) with mu, the location (expectation) and y, the sample
        from the likelihood at that location.
        """
        n = X.shape[1]
        assert (n == beta.shape[0])
        y, mu = self.likelihood_model(X, beta, *args)

        return mu, y.sample((n,))  # Fixme: y sampled!


if __name__ == '__main__':
    xgrid = (0,10)
    xbetasigma = Xbetasigma(
        X=tf.stack(axis=1,
                   values=[tf.ones((100,)), tfd.Uniform(xgrid[0], xgrid[1]).sample((100,))]),
        # True Parameters
        beta=tf.constant([-1., 2.]), sigma=tf.constant([10.]),

        # prior model
        prior_beta=tfd.Normal(loc=[0., 0.], scale=[1., 1.]),
        prior_sigma=tfd.InverseGamma(0.001, 0.001),

        #
        initial=[tf.constant([1., -1.]), tf.constant([4.])],  # CAREFULL MUST BE FLOAT!
        bijectors=[tfb.Identity(), tfb.Exp()])

    xbetasigma.unnormalized_log_prob(beta=tf.constant([-1., 2.]), sigma=tf.constant([10.]))
    samples, traced = xbetasigma.sample_chain(num_burnin_steps=int(1e1), num_results=int(10e1),
    logdir = '/home/tim/PycharmProjects/Thesis/TFResults')

    # predicting with the first parameter set of the chain
    xbetasigma.predict(xbetasigma.X, *samples[0])
