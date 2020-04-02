import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from Python.Bayesian.layers.Hidden import Hidden


class Regression(Hidden):
    def __init__(self, input_shape):
        """Homoscedastic Regression"""

        # instantiation of self.prior_model, containing W
        Hidden.__init__(self, input_shape=input_shape, no_units=1, activation='identity')
        self.prior_beta = self.prior_stackedW

        # removing b (as it is constant)
        self.init_b_from_prior = None
        self.log_prob = None
        del self.prior_b

        # for not overwriting dense
        self.b = tf.repeat(0., input_shape)

    @staticmethod
    @tf.function
    def dense(X, beta):
        return tf.linalg.matvec(X, beta)

    def likelihood_model(self, X, beta):
        """distributional assumption"""
        # mu = tf.linalg.matvec(X, beta)

        mu = self.dense(X, beta)
        likelihood = tfd.Normal(loc=mu, scale=1.)
        return likelihood, mu

    def _closure_log_prob(self, X, y):
        """A closure, to set X, y & the priors in this model"""

        @tf.function
        def xbeta_log_prob(beta):
            likelihood, _ = self.likelihood_model(X, beta)
            return (self.prior_beta.log_prob(beta) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return xbeta_log_prob


class Regression_Data(Regression):
    def __init__(self, X):
        """Gaussian indepentent prior on beta (W)"""
        n, p = X.shape
        self.X = X

        Regression.__init__(self, input_shape=p)
        self.beta = self.prior_beta.sample()

        # consider overwrite likelihood_model for e.g. heteroscedasticity
        likelihood, mu = self.likelihood_model(X, beta=self.beta)
        self.likelihood = likelihood  # Consider make tfd. accessible
        self.y = likelihood.sample()
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

    reg_data = Regression_Data(X)

    # (1) specify model --------------------------------------
    p = X.shape[1]
    reg = Regression(p)

    # check unnormalized log_prob
    print(reg._closure_log_prob(reg_data.X, reg_data.y)(beta=tf.constant([1.,1.])))

    adHMC = AdaptiveHMC(initial=tf.constant([1., 1.]),  # CAREFULL MUST BE FLOAT!
                        bijectors=tfb.Identity(),  # [tfb.Identity(), tfb.Identity()]
                        log_prob=reg._closure_log_prob(reg_data.X, reg_data.y))  # unnorm. log_prob


    samples, traced = adHMC.sample_chain(num_burnin_steps=int(1e3), num_results=int(10e2),
                                         logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    # for var, var_samples in pooled_samples._asdict().items():
    #     plot_traces(var, samples=var_samples, num_chains=4)
    adHMC.plot_traces('beta', adHMC.chain, 2)
    print(reg_data.beta)
    print(tf.reduce_mean(samples, axis=0))
