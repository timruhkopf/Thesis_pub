import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# from respective__init__.py
from Python.Data import *
from Python.MCMC import *

# Cases
from Python.Data.Effect1D_cases import Bspline_K


class Xbeta(AdaptiveHMC):

    def __init__(self, xgrid, n, beta=[-1., 2.]):
        # data generation
        # MAKE SURE, all of the below is TF Ready

        self.n = n
        self.X = tf.stack([tf.ones((self.n,)), tfd.Uniform(xgrid[0], xgrid[1]).sample((self.n,))], axis=1)
        self.beta = tf.constant(beta)
        self.mu = tf.linalg.matvec(self.X, self.beta)
        self.y = self.mu + tfd.Normal(loc=0, scale=1).sample((self.n,))
        # TODO TRAIN TEST SPLIT?

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

        @tf.function
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


class Xbetasigma(AdaptiveHMC):
    """ X @ \beta Regression with sigma prior on y, to check out """

    def __init__(self, xgrid, n, beta=[-1., 2.], sigma=[10.]):
        # data generation
        # MAKE SURE, all of the below is TF Ready

        self.n = n
        self.X = tf.stack([tf.ones((self.n,)), tfd.Uniform(xgrid[0], xgrid[1]).sample((self.n,))], axis=1)
        self.beta = tf.constant(beta)
        self.sigma = tf.constant(sigma)
        self.mu = tf.linalg.matvec(self.X, self.beta)
        self.y = self.mu + tfd.Normal(loc=0, scale=self.sigma).sample((self.n,))

        # attributes for sampler
        self.initial = [tf.constant([1., -1.]), tf.constant([4.])]  # CAREFULL MUST BE FLOAT!
        self.bijectors = [tfb.Identity(), tfb.Identity(), tfb.Exp()]

        AdaptiveHMC.__init__(self, initial=self.initial, bijectors=self.bijectors, log_prob=self.unnormalized_log_prob)

    def unnormalized_log_prob(self, beta, sigma):
        """
        A closure, to keep X & y from data generating instance
        :param beta: float tensor
        :return: log-posterior value
        """

        def xbeta_log_prob(beta, sigma, X, y):
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

        return xbeta_log_prob(beta, sigma, X=self.X, y=self.y)


class ZGamma(AdaptiveHMC, Bspline_K):

    def __init__(self, xgrid):
        Effects1D.__init__(self, xgrid=xgrid)

        # data generation
        # FIXME MAKE SURE, all of the below is TF Ready
        self.X = tfd.Uniform(0, 10).sample((100,))  # FIXME
        self.Z = get_design(self.X, degree=2)
        self.Bspline = Bspline_K(self.xgrid)
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
    import matplotlib.pyplot as plt

    # # (XBETA CASE) -------------------------------------------------------------
    xbeta = Xbeta(xgrid=(0, 10), n=1000, beta=[-1., 2.])

    # log posterior of true beta
    xbeta.unnormalized_log_prob(beta=tf.constant([-1., 2.]))

    samples, tup = xbeta.sample_chain()

    # (PRINTING TRACE RESULTS)
    # # argument structure depending on the nesting:
    # # SimpleStepSizeAdaptation /
    # print(tup.__doc__)
    # # TransformedTransitionKernel /
    # print(tup.inner_results.__doc__)
    # print(tup.inner_results.transformed_state.__doc__)
    # # MetropolisHastingsKernelResults (HMC)
    # print('\t', tup.inner_results.inner_results.__doc__)
    # # (uncalibrated) HamiltonianMonteCarlo
    # print('\t\t', tup.inner_results.inner_results.accepted_results.__doc__)
    #
    # print(tup.inner_results.inner_results.is_accepted)

    # # (XBETA SIGMA CASE) -------------------------------------------------------
    # xbetasigma = Xbetasigma(xgrid=(0, 10), n=1000, beta=[-1., 2.], sigma=[10.])
    # xbetasigma.unnormalized_log_prob(beta=tf.constant([-1., 2.]), sigma=tf.constant([10.]))
    # samples, tup = xbetasigma.sample_chain()

    is_accepted = tup[0][1][1]
    lw = 0.3
    # plt.plot(is_accepted[:, 0], label="trace of beta 0", lw=lw)
    # plt.plot(is_accepted[:, 1], label="trace of beta 1", lw=lw)
    # plt.title("Traces of acceptance of unknown parameters")

    # plot parameter traces
    # TODO make this in sub plots!
    # TODO parameters' standard deviation trace
    samples = samples.numpy()
    plt.plot(samples[np.all(is_accepted, axis=1), :][:, 0], label="trace of beta 0", lw=lw)
    plt.plot(samples[np.all(is_accepted, axis=1), :][:, 1], label="trace of beta 1", lw=lw)
    plt.title("Traces of unknown parameters")

    # histogram of parameters
    plt.hist(samples[:, 0], bins=30, histtype="stepfilled")
    plt.title("Traces of unknown parameters")
    plt.hist(samples[:, 1], bins=30, histtype="stepfilled")
    plt.title("Traces of unknown parameters")

    # # autocorrelation of samples
    # def autocorr(x):
    #     # from http://tinyurl.com/afz57c4
    #     result = np.correlate(x, x, mode='full')
    #     result = result / np.max(result)
    #     return result[result.size // 2:]

    from matplotlib import pyplot
    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(samples[np.all(is_accepted, axis=1)][:, 0])
    pyplot.show()

    ##################################################################

    # (draw y with heteroscedasticity) -----------------------------------------
    # bspline_k1 = Bspline_K(xgrid, order=2, sig_Q=0.1, sig_Q0=0.1)
    # mu_sigma = bspline_k1.spl(x)  # FIXME: ensure positive values for variance
    # mu_sigma += 1
    # mu_sigma *= 0.2
    # mu = 0.2 * mu
    #
    # mu_sigma = 2 + x * 3
    # z = np.random.normal(loc=mu, scale=mu_sigma)
