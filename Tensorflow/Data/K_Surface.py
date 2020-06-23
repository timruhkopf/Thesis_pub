import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict

tfb = tfp.bijectors
tfd = tfp.distributions


class K_surface:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """complex two dimensional effect, no main effects"""
        self.n = n
        self.grid = grid

        # X prior sample:
        self.X = tfd.Uniform(self.grid[0], self.grid[1]).sample((self.n, 2))

        # effect prior sample:
        from Tensorflow.Effects.Cases2D.K.GMRF_K import GMRF_K
        self.gmrf = GMRF_K(xgrid=self.grid, ygrid=self.grid)

        # evaluate the likelihood
        self.mu = tf.constant(self.gmrf.surface(self.X), dtype=tf.float32)
        self.likelihood = self.true_likelihood_model(self.mu)
        odict = self.likelihood.sample()
        self.y = odict['y']
        self.sigma = odict['sigma']

    def true_likelihood_model(self, mu):
        return tfd.JointDistributionNamed(OrderedDict(
            sigma=tfd.InverseGamma(0.5, 0.5),
            y=lambda sigma: tfd.Sample(
                tfd.Normal(loc=mu,
                           scale=sigma)))
        )


if __name__ == '__main__':
    pass
