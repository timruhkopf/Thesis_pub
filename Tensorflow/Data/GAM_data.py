import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from Tensorflow.Effects import get_design
from Tensorflow.Effects.Cases1D.Bspline_K import Bspline_K


class Data_GAM:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """Single B-Spline, Homoscedastic by default"""
        self.n = n
        self.grid = grid

        # true effect:
        self.prior()
        # self.mu = tf.constant(self.Bspline.spl(self.X))  # == matvec(Z, gamma)

        # Support & transformation of Data
        self.X = self.priorX(grid, n)
        self.Z = tf.convert_to_tensor(
            get_design(
                tf.reshape(self.X, (n,)).numpy(),
                degree=2,
                no_basis=self.gamma.shape[0]),
            tf.float32)

        y, mu = self.true_likelihood(self.Z, self.gamma)
        self.y = y.sample()
        self.mu = mu

    def priorX(self, grid, n):
        return tfd.Uniform(grid[0], grid[1]).sample((n, 1))

    def prior(self):
        # FIXME: should return a proper tfd. to allow easy log_prob:
        #  alternative, if improper:
        #  return Bspline_K
        #  with methods:
        #  self.Bspline_K.sample() und self.Bspline_K.log_prob()
        #  BOTH MUST RETURN TF TENSOR

        self.Bspline = Bspline_K(self.grid)
        self.gamma = tf.convert_to_tensor(self.Bspline.z, tf.float32)

    def true_likelihood(self, Z, gamma, sigma=tf.constant(1.)):
        mu = tf.linalg.matvec(Z, gamma)
        y = tfd.Normal(loc=mu, scale=sigma)
        return y, mu


if __name__ == '__main__':
    from Tensorflow.Bayesian.Models.GAM import GAM
    from Tensorflow.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    tfb = tfp.bijectors

    # (0) gam example ---------------------------------------
    data = Data_GAM(n=100)
    # gam_data.X
    # gam_data.Z
    # gam_data.gamma

    # no preset precision
    gam = GAM()  # precision @ default: diff_mat1D(dim=20, order=1)[1][1:, 1:]
    # gam.joint_prior.sample()
    # gam.unnormalized = gam._closure_log_prob(Z=data.Z, y=data.y)
    # gam.unnormalized(*list(gam.joint_prior.sample().values()))

    import functools
    import seaborn as sns

    unnormalized_posterior_log_prob = functools.partial(
        gam.joint_log_prob,
        y=data.y, Z=data.Z)  # y = tf.reshape(data.y, (data.y.shape[0], 1)

    sampled = gam.model(data.Z).sample()
    w = tf.concat([tf.reshape(sampled['w0'], (1,)), sampled['w']], axis=0)
    mu = gam.dense(data.Z, w)

    sns.scatterplot(tf.reshape(data.X, (data.X.shape[0],)), data.y,)
    sns.lineplot(tf.reshape(data.X, (data.X.shape[0],)), mu)
    # plt.title('Data & estimated function, log_prob={}'.format(unnormalized_posterior_log_prob(
    #     **{k: v for k, v in sampled.items() if k != 'y'})))


    # sampled['y']
    # FIXME!!!!: log posterior of this data easily diverges to infinity
    unnormalized_posterior_log_prob(
        **{k: v for k, v in sampled.items() if k != 'y'})

    # Parameters: tau, w, w0, sigma
    # get key-ordering
    # gam.joint_prior._parameters['model'].keys()

    bijectors = {'tau': tfb.Exp(),
                 'w0': tfb.Identity(),
                 'w': tfb.Identity(),
                 'sigma': tfb.Exp()}

    initial_state = sampled

    adHMC = AdaptiveHMC(
        initial=[initial_state[k] for k in ['tau', 'w0', 'w', 'sigma']],  # kick out y
        bijectors=[bijectors[k] for k in ['tau', 'w0', 'w', 'sigma']],
        # gam.gam1._parameters['model'].keys() if k != 'y'],
        # [bijectors[k] for k in gam.joint._parameters['model'].keys() if k != 'y']
        # [tfb.Exp(), tfb.Identity(), tfb.Identity(), tfb.Exp()]
        log_prob=unnormalized_posterior_log_prob)

    # adHMC = AdaptiveHMC(
    #     initial=list(gam.joint_prior.sample().values()),
    #     bijectors=[bijectors[k] for k in gam.joint_prior._parameters['model'].keys()],
    #     # [tfb.Exp(), tfb.Identity(), tfb.Identity(), tfb.Exp()]
    #     log_prob=gam._closure_log_prob(Z=data.Z, y=data.y))

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(1e3),
        num_results=int(10e2),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    print('')
