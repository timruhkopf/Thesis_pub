import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Data import get_design
from Python.Data.Effect1D import Effects1D
from Python.Data.Cases1D.Bspline_K import Bspline_K
from Python.Bayesian.Samplers import AdaptiveHMC


class ZGamma(AdaptiveHMC, Bspline_K):

    def __init__(self, xgrid, n, sigma, prior_gamma):
        Effects1D.__init__(self, xgrid=xgrid)

        # data generation
        self.n = n
        self.X = tfd.Uniform(xgrid[0], xgrid[1]).sample((n,))
        self.Z = get_design(self.X, degree=2)
        self.Bspline = Bspline_K(self.xgrid)
        self.gamma = tf.constant(self.Bspline.z)
        self.mu = self.Bspline.spl(self.X)
        assert (self.mu, tf.linalg.matvec(self.Z, self.gamma))
        self.y = self.mu + tfd.Normal(loc=0, scale=1).sample((self.n,))

        # attributes for sampler
        self.initial = tf.constant([1., 1.])  # CAREFULL MUST BE FLOAT!
        self.bijectors = tfb.Identity()  # [tfb.Identity(), tfb.Identity()]

        self.prior_gamma = prior_gamma
        self.unnormalized_log_prob = self._closure_log_prob(
            self.X, self.y, self.prior_gamma)

        AdaptiveHMC.__init__(self, initial=self.initial,
                             bijectors=self.bijectors,
                             log_prob=self.unnormalized_log_prob)

    def likelihood(self, X, beta, sigma):
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

    def closure_log_prob(self, X, y, sigma, prior_gamma):  # fixing sigma
        def zgamma_log_prob(gamma):
            # setting up likelihood model
            print('beta: {},\n sigma {}'.format(gamma))
            likelihood, _ = self.likelihood(X, gamma, sigma)  # FIXME: second iteration sigma is a string tensor

            # return log posterior value
            return (tf.reduce_sum(prior_gamma.log_prob(gamma)) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return zgamma_log_prob

    # def __repr__(self):
    #     """
    #     # Consider Print method for class instance:
    #     Can leverage the plot method on the data generation process
    #     and some basic statistics.
    #     :example:
    #         zgamma = Zgamma(...)
    #         print(zgamma)
    #
    #      """
    #     pass


if __name__ == '__main__':
    # Fixme requires K (Q) to be defined on xgrid! & Effect1D to hold log_prob function
    Bspline_K.log_prob()
    zgamma = ZGamma(xgrid = (0,10), n = 1000, sigma=1 ) # fixme set sigma to fix value!
                    #prior_gamma=)
