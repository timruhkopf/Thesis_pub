import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class Data_BNN_1D:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """One dimensional effect"""
        self.grid = grid
        self.n = n

        self.X = self.prior_X().sample((self.n,))

        # instantiate the data
        self.prior()  # Consider self.prior().sample()
        likelihood, mu = self.true_likelihood(self.X)
        self.y = likelihood.sample()
        self.mu = mu

    def prior_X(self):
        return tfd.Uniform(self.grid[0], self.grid[1])

    def prior(self):
        """HERE: drawing effect instance from prior"""
        from Python.Effects.Cases1D.Bspline_K import Bspline_K
        self.gmrf = Bspline_K(xgrid=self.grid)

        # FIXME: should return a proper tfd. to allow easy log_prob:
        #  alternative, if improper:
        #  return gmrf
        #  with methods:
        #  self.gmrf.sample() und self.gmrf.log_prob()
        #  BOTH MUST RETURN TF TENSOR

    def true_likelihood(self, X):
        """class of data, allows to generate data instances.
        :return <tuple> tfd for y, mu(X, gamma) """
        mu = tf.constant(self.gmrf.spl(X), dtype=tf.float32)
        likelihood = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=tf.repeat(1., mu.shape[0]))
        return likelihood, mu


class Data_BNN_2D:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """complex two dimensional effect, no main effects"""
        from Python.Effects.Cases2D.K.GMRF_K import GMRF_K
        self.gmrf = GMRF_K(xgrid=grid, ygrid=grid)

        self.n = n
        self.X = tf.stack(
            values=[tfd.Uniform(grid[0], grid[1]).sample((self.n,)),
                    tfd.Uniform(grid[0], grid[1]).sample((self.n,))],
            axis=1)
        self.y = self.true_likelihood(self.X).sample((self.n,))

    def true_likelihood(self, X):
        self.mu = tf.constant(self.gmrf.surface(X), dtype=tf.float32)
        y = tfd.MultivariateNormalDiag(
            loc=self.mu,
            scale_diag=tf.repeat(1., self.mu.shape[0]))
        return y


if __name__ == '__main__':
    from Python.Bayesian.Models.BNN import BNN
    # (0) (generating data) ----------------------------------
    data = Data_BNN_2D(n=1000, grid=(0, 10, 0.5))

    # (1) (setting up posterior model) -----------------------
    bnn = BNN(hunits=[2, 10, 9, 8, 1], activation='relu')
    bnn._initialize_from_prior()
    y = bnn.forward(X=tf.constant([3., 4]), Ws=bnn.Ws, bs=bnn.bs)

    bnn.unnormalized_log_prob = bnn._closure_log_prob(data.X, data.y)


    # (2) (sampling posterior) -------------------------------
    # from Python.Bayesian.Samplers import AdaptiveHMC
    # bnn._initialize_from_prior()
    # adHMC = AdaptiveHMC(initial=initial, # FIXME: input format
    #             bijectors=tfb.Identity(),
    #             log_prob=bnn.unnormalized_log_prob)
    #
    # samples, traced = adHMC.sample_chain(
    #     logdir='/home/tim/PycharmProjects/Thesis/TFResults')

print('')
