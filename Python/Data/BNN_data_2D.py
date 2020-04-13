import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class Data_BNN_2D:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """complex two dimensional effect, no main effects"""
        self.n = n
        self.grid = grid

        self.X = self.prior_X()

        self.prior()  # Consider self.prior().sample()
        likelihood, mu = self.true_likelihood(self.X)
        self.y = likelihood.sample()
        self.mu = mu

    def prior_X(self):
        return tf.stack(
            values=[tfd.Uniform(self.grid[0], self.grid[1]).sample((self.n,)),
                    tfd.Uniform(self.grid[0], self.grid[1]).sample((self.n,))],
            axis=1)

    def prior(self):
        from Python.Effects.Cases2D.K.GMRF_K import GMRF_K
        self.gmrf = GMRF_K(xgrid=self.grid, ygrid=self.grid)

        # FIXME: should return a proper tfd. to allow easy log_prob:
        #  alternative, if improper:
        #  return gmrf
        #  with methods:
        #  self.gmrf.sample() und self.gmrf.log_prob()
        #  BOTH MUST RETURN TF TENSOR

    def true_likelihood(self, X):
        mu = tf.constant(self.gmrf.surface(X), dtype=tf.float32)
        likelihood = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=tf.repeat(1., mu.shape[0]))
        return likelihood, mu


if __name__ == '__main__':
    pass
