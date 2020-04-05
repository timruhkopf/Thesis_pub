import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from Python.Effects import get_design
from Python.Effects.Cases1D.Bspline_K import Bspline_K


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
        self.y = y.sample((self.n,))
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
    from Python.Bayesian.layers.GAM import GAM
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    tfb = tfp.bijectors

    # (0) gam example ---------------------------------------
    gam = Data_GAM(n=100)
    gam.X
    gam.Z
    gam.gamma

    AdaptiveHMC()

    print('')
