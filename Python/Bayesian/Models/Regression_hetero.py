import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from Python.Bayesian.Models.Regression import Regression


class Regression_hetero(Regression):
    def __init__(self, input_shape):
        """y is heteroscedastic"""
        # calling Hidden's prior model for beta
        Regression.__init__(self, input_shape)

        self.prior_sigma = tfd.Gamma(0.1, 0.2)

    def likelihood_model(self, X, beta, sigma):
        """distributional assumption
        :param sigma: Tensor, of length X.shape[0]"""
        mu = tf.linalg.matvec(X, beta)
        likelihood = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return likelihood, mu

    def _closure_log_prob(self, X, y):
        """A closure, to set X, y & the priors in this model"""

        @tf.function
        def xbeta_log_prob(beta, sigma):
            likelihood, _ = self.likelihood_model(X, beta, sigma)
            return (self.prior_beta.log_prob(beta) +
                    tf.reduce_sum(self.prior_sigma.log_prob(sigma)) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return xbeta_log_prob


class Regression_hetero_Data(Regression_hetero):
    def __init__(self, X):
        n, p = X.shape
        self.X = X

        # get prior for beta
        Regression_hetero.__init__(self, p)

        # sample priors
        self.beta = self.prior_beta.sample()
        self.sigma = self.prior_sigma.sample(n)

        # sample likelihood
        likelihod, mu = self.likelihood_model(X, self.beta, self.sigma)
        self.y = likelihod.sample()
        self.mu = mu

if __name__ == '__main__':
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    tfb = tfp.bijectors
    # (0) generate data ---------------------------------------
    xgrid = (0, 10)
    n = 100
    X = tf.stack(
        values=[tf.ones((n,)),
                tfd.Uniform(xgrid[0], xgrid[1]).sample((n,))],
        axis=1)

    reg_data = Regression_hetero_Data(X)

    # (1) setting up the model -------------------------------
    p = X.shape[1]
    reg = Regression_hetero(p)

    # (2) estimate the model ---------------------------------
    print(reg._closure_log_prob(reg_data.X, reg_data.y)(beta=tf.constant([1.,1.]), sigma=tf.repeat(1., n)))

    adHMC = AdaptiveHMC(initial=[tf.constant([1., 1.]), tf.repeat(1., n)],  # CAREFULL MUST BE FLOAT!
                        bijectors=[tfb.Identity(),tfb.Exp()],  # [tfb.Identity(), tfb.Identity()]
                        log_prob=reg._closure_log_prob(reg_data.X, reg_data.y))  # unnorm. log_prob


    samples, traced = adHMC.sample_chain(num_burnin_steps=int(1e3), num_results=int(10e2),
                                         logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    print(reg_data.sigma)
    print(reg_data.beta)


    rate_accepted = tf.reduce_mean(tf.cast( traced[1].is_accepted, tf.float32), axis=0)

    for chain_accepted in samples:
        sample_mean = tf.reduce_mean(chain_accepted, axis=0)
        sample_stddev = tf.math.reduce_std(chain_accepted, axis=0)
        tf.print('sample mean:\t', sample_mean,
                 '\nsample std: \t', sample_stddev,
                 '\nacceptance rate: ', rate_accepted)

print('')