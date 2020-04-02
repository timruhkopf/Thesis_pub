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

    # A TORCH impementation of BNN & SGMCMC can be found here:
    # https://sgmcmc.readthedocs.io/en/latest/index.html

    def __init__(self, hunits=[2, 10, 9, 8, 1], activation='relu'):
        """
        model specification:
        :param hunits: list: number of hidden units per layer. first element is
        input shape. last element is output_shape.
        :param activation: activation function for
        """
        self.hunits = hunits
        self.layers = [Hidden(i, u, activation) for i, u in zip(self.hunits[:-2], self.hunits[1:-1])]
        self.layers.append(Hidden(input_shape=self.hunits[-2],
                                  no_units=self.hunits[-1],
                                  activation='identity'))

    def _initialize_from_prior(self):
        """
        initialize the hidden units weights and biases from their priors.
        This allows
        """
        self.Ws = [h.init_W_from_prior() for h in self.layers]
        self.bs = [h.prior_b.sample() for h in self.layers]
        self.bs[-1] = tf.constant([0.]) # final layer bs is constant

    @tf.function
    def forward(self, X, Ws, bs):

        """
        Evaluate the NN nested in BNN, once random variables are realized.
        :param X: Batch of x vectors
        :param Ws: list defining all of NN's weight matrices
        :param bs: list defining all of NN's bias vectors
        :return: f(x,w,b)
        """
        # Consider default arguments for Ws, bs - if already self.Ws and bs exist?
        # Consider map dense accross all rows of X! (as dense operates on vectors)
        for h, W, b in zip(self.layers, Ws, bs):
            X = h.dense(X, W, b)
        return X


    def likelihood_model(self, X, Ws, bs):
        # CANNOT BE @tf.function, as it would have to return single tensor!
        mu = self.forward(X, Ws, bs)
        likelihood = tfd.MultivariateNormalDiag(
            loc=mu, scale_diag=tf.repeat(1., mu.shape[0]))
        return likelihood, mu

    def _closure_log_prob(self, X, y):
        """A closure, to preset X, y in this model and match HMC's expected model format"""

        @tf.function
        def BNN_log_prob(Ws, bs):
            """unnormalized log posterior value: log_priors + log_likelihood"""
            likelihood, _ = self.likelihood_model(X, Ws, bs)

            # Carefull: W STACKING: W is matrix, but prior was vector!!!!
            # notice: next generator statement is not allowed
            return (tf.reduce_sum(
                        [h.log_prob(tf.reshape(W, (h.no_units * h.input_shape,)), b)
                         for h, W, b in zip(self.layers, Ws, bs)]) +
                    tf.reduce_sum(likelihood.log_prob(y)))

        return BNN_log_prob

    # DEPREC: list of tensors is applicable! no need for parsing!
    # def argparser(self, paramlist):
    #     """parse the chain's results, because sample_chain(current_state= is
    #     either tensor or list of tensors, but not multiple arguments!!!"""
    #     Ws = paramlist[:len(self.layers)]
    #     bs = paramlist[len(self.layers):]
    #     return Ws, bs


if __name__ == '__main__':
    # (1 2D) (setting up posterior model) -----------------------
    bnn = BNN(hunits=[2, 10, 9, 8, 1], activation='relu')
    bnn._initialize_from_prior()

    # (1.1) (sampling NN from priors) ------------------------
    y = bnn.forward(X=tf.constant([3., 4.]), Ws=bnn.Ws, bs=bnn.bs)

    # batches work naturally!
    y = bnn.forward(X=tf.constant([[1., 2.], [3., 4.]]), Ws=bnn.Ws, bs=bnn.bs)

    # (1 1D) (setting up posterior model) -----------------------
    bnn = BNN(hunits=[1, 10, 9, 8, 1], activation='relu')
    bnn._initialize_from_prior()

    # (1.1) (sampling NN from priors) ------------------------
    y = bnn.forward(X=tf.constant([[3.], [4.]]), Ws=bnn.Ws, bs=bnn.bs)

    # (2) sample prior functions -----------------------------
    from Python.Data.BNN_data import Data_BNN_1D

    data = Data_BNN_1D(n=100)
    bnn.unnormalized_log_prob = bnn._closure_log_prob(
        X=tf.reshape(data.X, (100, 1)), y=data.y)

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=4, ncols=4)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Sampling BNN Functions from prior')

    for ax in axes.flatten():
        bnn._initialize_from_prior()
        y = bnn.forward(X=tf.reshape(tf.range(-1., 10., 0.1), (110, 1)), Ws=bnn.Ws, bs=bnn.bs)

        # evaluate bnn functions drawn from prior
        sns.lineplot(x=tf.range(-1., 10., 0.1), y=tf.reshape(y, (110,)).numpy(), ax=ax)
        ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(bnn.Ws, bnn.bs).numpy()))

    # (3) (fitting to data) ----------------------
    from Python.Data.BNN_data import Data_BNN_1D

    data = Data_BNN_1D(n=100)
    bnn.unnormalized_log_prob = bnn._closure_log_prob(X=data.X, y=data.y)

    print('')
