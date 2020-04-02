import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Hidden:
    def __init__(self, input_shape, no_units=10, activation='relu'):
        """create independent normal priors with MVN(0,I)"""

        self.input_shape = input_shape
        self.no_units = no_units

        self.prior_model(no_units, input_shape)

        identity = lambda x: x
        self.activation = {'relu': tf.nn.relu,
                           'tanh': tf.math.tanh,
                           'identity': identity}[activation]

    def prior_model(self, no_units, input_shape):
        # TODO Hyperparam for Variance.
        # W matrix is drawn with stacked vector
        self.prior_stackedW = tfd.MultivariateNormalDiag(
            loc=tf.repeat(0., no_units * input_shape),
            scale_diag=tf.repeat(1., no_units * input_shape))
        self.prior_b = tfd.MultivariateNormalDiag(
            loc=tf.repeat(0., no_units),
            scale_diag=tf.repeat(1., no_units))

    def init_W_from_prior(self):
        """initialize W matrix from stacked_prior"""
        self.W = tf.reshape(self.prior_stackedW.sample(), shape=(self.no_units, self.input_shape))
        return self.W

    def init_b_from_prior(self):
        """initialize bias vector from bias prior"""
        self.b = self.prior_b.sample()
        return self.b

    @tf.function
    def dense(self, x, W, b):
        return self.activation(tf.linalg.matvec(W, x) + b)

    @tf.function
    def log_prob(self, W, b):
        """
        Prior log probability for entire Unit
        :param W: stacked vector of W matrix
        :param b: bias vector
        :return: joined prior log_prob value of the Hidden unit.
        """
        # TODO Hyperparam for Variance.
        return self.prior_stackedW.log_prob(W) + self.prior_b.log_prob(b)


if __name__ == '__main__':
    h = Hidden(input_shape=2, no_units=3, activation='relu')

    # check dense functon
    h.dense(x=tf.constant([1., 2.]),
            W=tf.constant([[1., 1.], [1., 2.], [3., 4.]]),  # three hidden units
            b=tf.constant([0.5, 1., 1.]))

    # check prior logprob
    h.init_W_from_prior()
    h.init_b_from_prior()
    h.log_prob(W=tf.reshape(h.W, shape=(tf.reduce_prod(h.W.shape),)), b=h.b)


print('')
