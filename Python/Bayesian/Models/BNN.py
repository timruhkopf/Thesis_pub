import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class Hidden:
    def __init__(self, input_shape, no_units=10, activation='relu'):
        """create independent normal priors with MVN(0,I)"""

        self.input_shape = input_shape
        self.no_units = no_units

        # W matrix is drawn with stacked vector
        self.prior_stackedW = tfd.MultivariateNormalDiag(
            loc=tf.repeat(0., no_units * input_shape),
            scale_diag=tf.repeat(1., no_units * input_shape))
        self.prior_b = tfd.MultivariateNormalDiag(
            loc=tf.repeat(0., no_units),
            scale_diag=tf.repeat(1., no_units))

        identity = lambda x: x
        self.activation = {'relu': tf.nn.relu,
                           'tanh': tf.math.tanh,
                           'identity': identity}[activation]

    def init_W_from_prior(self):
        """initialize W matrix from stacked_prior"""
        return tf.reshape(self.prior_stackedW.sample(), shape=(self.no_units, self.input_shape))

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

    def __init__(self, hunits=[2, 10, 9, 8, 1], activation='relu'):
        """
        model specification:
        :param hunits: list: number of hidden units per layer. first element is
        input shape. last element is output_shape.
        :param activation: activation function for
        """
        self.hunits = hunits
        self.layers = [Hidden(i, u, activation) for i, u in zip(self.hunits[:-2], self.hunits[1:-1])]
        self.final_layer = Hidden(input_shape=self.hunits[-2],
                                  no_units=self.hunits[-1],
                                  activation='identity')

    def _initialize_from_prior(self):
        # TODO: list comprehension or tf.map_fn(lambda i: i ** 2 if i > 0 else i, x)
        # tf.map_fn(lambda x: x[0] + x[1], tf.stack([tf.constant([1., 2.]), tf.constant([3., 4.])], axis=1))

        self.Ws = [h.init_W_from_prior() for h in [*self.layers, self.final_layer]]
        self.bs = [h.prior_b.sample() for h in self.layers]

    # @tf.function
    def forward(self, X, Ws, bs):
        # TODO default arguments for Ws, bs - if already self.Ws and bs exist?
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

        X = self.final_layer.dense(X, Ws[-1], tf.constant(0.))
        return X

    # @tf.function
    def likelihood_model(self, X, Ws, bs):
        mu = self.forward(X, Ws, bs)
        y = tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.repeat(1., mu.shape[0]))
        return y

    def _closure_log_prob(self, X, y):
        """A closure, to preset X, y in this model and match HMC's expected model format"""

        # @tf.function
        def BNN_log_prob(Ws, bs):
            """unnormalized log posterior value: log_priors + log_likelihood"""
            # Fixme: BNN_log_prob(Ws, bs) sample_chain(current_state= is either
            #  tensor or list of tensors, but not multiple arguments!!!
            # Consider (*args) and argparser(*args)

            likelihood, _ = self.likelihood_model(X, Ws, bs)

            # FIXME: carefull with the final layers bs also stack W to vector!
            return (tf.reduce_sum(h.log_prob(W, b) for h, W, b in zip(self.layers, Ws, bs)) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return BNN_log_prob

    def argparser(self, paramlist):
        """parse the chain's results, because sample_chain(current_state= is
        either tensor or list of tensors, but not multiple arguments!!!"""
        Ws = paramlist[:len(self.layers)]
        bs = paramlist[len(self.layers):]
        return Ws, bs


class Data_BNN_1D:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """One dimensional effect"""
        from Python.Data.Cases1D.Bspline_GMRF import Bspline_GMRF
        self.gmrf = Bspline_GMRF(xgrid=grid)

        self.n = n
        self.X = tfd.Uniform(grid[0], grid[1]).sample((self.n,))
        self.y = self.true_likelihood(self.X).sample((self.n,))

    def true_likelihood(self, X):
        self.mu = tf.constant(self.gmrf.spl(X), dtype=tf.float32)
        y = tfd.MultivariateNormalDiag(
            loc=self.mu,
            scale_diag=tf.repeat(1., self.mu.shape[0]))
        return y


class Data_BNN_2D:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """complex two dimensional effect, no main effects"""
        from Python.Data.Cases2D.K.GMRF_K import GMRF_K
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
    h = Hidden(input_shape=2, no_units=3, activation='relu')
    h.dense(x=tf.constant([1., 2.]),
            W=tf.constant([[1., 1.], [1., 2.], [3., 4.]]),  # three hidden units
            b=tf.constant([0.5, 1., 1.]))

    # (0) (generating data) ----------------------------------
    # data = Data_BNN_2D(n=1000, grid=(0, 10, 0.5))

    # (1) (setting up posterior model) -----------------------
    bnn = BNN(hunits=[2, 10, 9, 8, 1], activation='relu')
    # data = Data_BNN_2D(n=1000, grid=(0, 10, 0.5))
    # bnn.unnormalized_log_prob = bnn._closure_log_prob(data.X, data.y)

    # (1.1) (sampling NN from priors) ------------------------
    bnn._initialize_from_prior()
    y = bnn.forward(X=tf.constant([3.,4]), Ws=bnn.Ws, bs=bnn.bs)

    # batches work naturally!
    y = bnn.forward(X=tf.constant([[1., 2.],[3.,4]]), Ws=bnn.Ws, bs=bnn.bs)


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
