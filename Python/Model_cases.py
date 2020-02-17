# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.bspline import get_design, diff_mat1D
from Python.Effect2D_K import *
from Python.Effect2D_distance import *
from Python.Effect1D import Effects1D, Bspline_K


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

    def predict(self):
        pass

    def plot_y1D(self):
        pass

    def plot_y2D(self, xgrid, ygrid, effectsurface):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_wireframe(meshx, meshy, fxy.surface(gridxy))
        ax.scatter(xs=self.X[:, 0], ys=self.X[:, 1], zs=self.y, alpha=0.3)
        ax.set_title('N(f(x,y), ...) = z')

        # plot mu
        (xmesh, ymesh), gridvec = Effects2D._generate_grid(self, xgrid, ygrid)
        gridmu = effectsurface.surface(gridvec)
        # ax.plot_surface(X=xmesh, Y=ymesh, Z=gridmu.reshape(xmesh.shape) ,facecolors=np.repeat('b', 100).reshape(xmesh.shape), linewidth=0)
        ax.plot_trisurf(xmesh.reshape(gridmu.shape), ymesh.reshape(gridmu.shape), gridmu, alpha=0.3,linewidth=0.2, antialiased=True)

        # ax1 = fig.add_subplot(222, projection='3d')
        # ax1.scatter(xs=self.X[:,0], ys=self.X[:,1], zs=self.mu, alpha=0.3)
        # ax1.set_title('mu')

        plt.show()


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
    import matplotlib.pyplot as plt


    # # (XBETA CASE) -------------------------------------------------------------
    # xbeta = Xbeta(xgrid=(0, 10), n=1000, beta=[-1., 2.])
    #
    # # log posterior of true beta
    # xbeta.unnormalized_log_prob(beta=tf.constant([-1., 2.]))
    #
    # samples, tup = xbeta.sample_chain()
    #
    # # # (XBETA SIGMA CASE) -------------------------------------------------------
    # # xbetasigma = Xbetasigma(xgrid=(0, 10), n=1000, beta=[-1., 2.], sigma=[10.])
    # # xbetasigma.unnormalized_log_prob(beta=tf.constant([-1., 2.]), sigma=tf.constant([10.]))
    # # samples, tup = xbetasigma.sample_chain()
    #
    # lw = 0.3
    # # plot acceptance
    # is_accepted = tup[0][1][1]
    # no_accepted = np.sum(is_accepted, axis=0)
    #
    # plt.plot(is_accepted[:, 0], label="trace of beta 0", lw=lw)
    # plt.plot(is_accepted[:, 1], label="trace of beta 1", lw=lw)
    # plt.title("Traces of acceptance of unknown parameters")
    #
    # sam = samples.numpy()
    #
    # # plot parameter traces
    # # TODO make this in sub plots!
    # # TODO parameters' standard deviation trace
    # plt.plot(sam[:, 0], label="trace of beta 0", lw=lw)
    # plt.plot(sam[:, 1], label="trace of beta 1", lw=lw)
    # plt.title("Traces of unknown parameters")
    #
    # # histogram of parameters
    # plt.hist(sam[:, 0], bins=30, histtype="stepfilled")
    # plt.title("Traces of unknown parameters")
    # plt.hist(sam[:, 1], bins=30, histtype="stepfilled")
    # plt.title("Traces of unknown parameters")
    #
    #
    # # autocorrelation of samples
    # def autocorr(x):
    #     # from http://tinyurl.com/afz57c4
    #     result = np.correlate(x, x, mode='full')
    #     result = result / np.max(result)
    #     return result[result.size // 2:]
    #
    #
    # x = np.arange(0, 10000)
    # plt.bar(x, autocorr(sam[:, 0])[1:], width=1, label="$y_t$")
    # # edgecolor=colors[0], color=colors[0])
    #
    # plt.legend(title="Autocorrelation")
    # plt.ylabel("measured correlation \nbetween $y_t$ and $y_{t-k}$.")
    # plt.xlabel("k (lag)")
    #
    # print(samples)

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

    # (plot y) -----------------------------------------------------------------
    class Exampledata(GMRF, AdaptiveHMC):
        def __init__(self, xgrid=(0, 10, 1), ygrid=(0, 10, 1), n=1000):
            self.X = np.stack([np.random.uniform(low=xgrid[0], high=xgrid[1]-xgrid[2], size=n), \
                               np.random.uniform(low=ygrid[0], high=ygrid[1]-ygrid[2], size=n)], axis=-1)

            # mu = bspline_k.spl(x) + bspline_cum.spl(y) + gmrf_k.surface(np.stack([x, y], axis=-1))
            self.gmrf =GMRF(xgrid, ygrid, lam=1, phi=40, delta=10, radius=10, tau=1, decomp='eigenB')
            self.mu = self.gmrf.surface(self.X)  # np.stack([x, y]

            self.y = np.random.normal(loc=self.mu, scale=0.1, size=n)

            self.plot_y2D(xgrid=xgrid, ygrid=ygrid, effectsurface=self.gmrf)


    example = Exampledata()
