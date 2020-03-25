import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Bayesian.layers.Hidden import Hidden

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

    @tf.function
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



if __name__ == '__main__':
    # (1) (setting up posterior model) -----------------------
    bnn = BNN(hunits=[2, 10, 9, 8, 1], activation='relu')

    # (1.1) (sampling NN from priors) ------------------------
    bnn._initialize_from_prior()
    y = bnn.forward(X=tf.constant([3.,4]), Ws=bnn.Ws, bs=bnn.bs)

    # batches work naturally!
    y = bnn.forward(X=tf.constant([[1., 2.],[3.,4]]), Ws=bnn.Ws, bs=bnn.bs)


    print('')
