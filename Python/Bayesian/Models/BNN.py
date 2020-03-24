import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class Hidden:
    def __init__(self, no_units=10, activation='relu'):
        """create independent normal priors with MVN(0,I)"""

        self.prior_W = tfd.MultivariateNormalDiag(
            loc=tf.repeat(0., no_units),
            scale_diag=tf.repeat(1., no_units))
        self.prior_b = tfd.MultivariateNormalDiag(
            loc=tf.repeat(0., no_units),
            scale_diag=tf.repeat(1., no_units))

        self.activation = {'relu': tf.nn.relu,
                           'tanh': tf.math.tanh}[activation]

    @tf.function
    def dense(self, x, W, b):
        return self.activation(tf.linalg.matvec(W, x) + b)

    @tf.function
    def log_prob(self, W, b):
        return self.prior_W.log_prob(W) + self.prior_b.log_prob(b)


class BNN:
    # on how to write BNN models in Code (later example has even TB)
    # https://github.com/tensorflow/probability/issues/292 # but claimed to not work

    # explicit formulation (formula) of BNN modeling mean of y~N and indep.
    # gaussian priors for for weights and biases.
    # https://www.cs.tufts.edu/comp/150BDL/2018f/assignments/hw2.html

    def __init__(self, units=[10, 9, 8, 1], activation='relu'):
        self.units = units
        self.layers = [Hidden(u, activation) for u in units]

    def _initialize_from_prior(self):
        # FIXME: sample_shape!. Does list match Samplers.initial input format?
        # FIXME: list comprehension or tf.map_fn(lambda i: i ** 2 if i > 0 else i, x)
        self.Ws = [h.prior_W.sample((1,)) for h in self.layers]
        self.b = [h.prior_b.sample((1,)) for h in self.layers]

    @tf.function
    def forward(self, X, Ws, bs):
        """
        Evaluate the NN nested in BNN, once random variables are realized.
        :param X: Batch of x vectors
        :param Ws: list defining all of NN's weight matrices
        :param bs: list defining all of NN's bias vectors
        :return: f(x,w,b)
        """
        # FIXME: map dense accross all rows of X! (as dense operates on vectors)
        for h, W, b in zip(self.layers, Ws, bs):
            X = h.dense(X, W, b)
        return X

    @tf.function
    def likelihood_model(self, X, Ws, bs):
        mu = self.forward(X, Ws, bs)
        y = tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.repeat(1., mu.shape[0]))
        return y

    def _closure_log_prob(self, X, y):
        """A closure, to set X, y in this model and get HMC's expected model format"""

        @tf.function
        def BNN_log_prob(Ws, bs):
            likelihood, _ = self.likelihood_model(X, Ws, bs)
            # unnormalized log posterior value: log_priors + log_likelihood
            return (tf.reduce_sum(h.log_prob(W, b) for h, W, b in zip(self.layers, Ws, bs)) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return BNN_log_prob


if __name__ == '__main__':
    # (0) (generating data) ----------------------------------
    from Python.Data.Cases2D.K.GMRF_K import GMRF_K

    data = GMRF_K(xgrid=(0, 10, 0.5), ygrid=(0, 10, 0.5))

    # (1) (setting up posterior model) -----------------------
    bnn = BNN(units=[10, 9, 8, 1], activation='relu')
    bnn.unnormalized_log_prob = bnn._closure_log_prob(data.X, data.y)

    # (1.1) (sampling NN from priors) ------------------------


    # (2) (sampling posterior) -------------------------------
    # FIXME BNN initial
    # from Python.Bayesian.Samplers import AdaptiveHMC
    # bnn._initialize_from_prior()
    # adHMC = AdaptiveHMC(initial=initial,
    #             bijectors=tfb.Identity(),
    #             log_prob=bnn.unnormalized_log_prob)
    #
    # samples, traced = adHMC.sample_chain(
    #     logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    print('')
