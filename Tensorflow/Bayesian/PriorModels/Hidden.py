from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
from Tensorflow.Util import setkwargs

tfd = tfp.distributions
tfb = tfp.bijectors


class Hidden:
    @setkwargs
    def __init__(self, input_shape, no_units=10, activation='relu'):
        """create independent normal priors with MVN(0,tau*I)"""

        identity = lambda x: x
        self.activation = {
            'relu': tf.nn.relu,
            'tanh': tf.math.tanh,
            'sigmoid': tf.math.sigmoid,
            'identity': identity}[activation]

        tau = tf.constant([5.])  # TODO CONSIDER TAU increased - or reference GIBBS, making it truely hierarchical
        self.joint = tfd.JointDistributionNamed(OrderedDict(
            # tau=tfd.InverseGamma(1., 1.),
            # W=tfd.Normal(tf.zeros((no_units, input_shape)), tau),
            # b=tfd.Normal(loc=tf.zeros(no_units,1), scale=1.)

            W=tfp.distributions.Sample(
                tfd.Normal(0., tau), sample_shape=(no_units, input_shape), validate_args=False, name=None),
            b=tfp.distributions.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=no_units)
        ))


        self.parameters = self.joint._parameters['model'].keys()
        self.bijectors = {'W': tfb.Identity(), 'b': tfb.Identity()}
        self.bijectors = [self.bijectors[k] for k in self.parameters]

    #@tf.function
    def dense(self, X, W, b):
        return self.activation(tf.matmul(X, W, transpose_b=True) + b)

    @tf.function
    def sample(self):
        """wrapper for unified format"""
        return self.joint.sample()

    #@tf.function
    def prior_log_prob(self, param):
        """prior_log_prob wrapper for unified format"""
        return tf.reduce_sum(self.joint.log_prob(**param))

    # self.PriorModels[0].joint.log_prob(self.PriorModels[0].joint.sample())


if __name__ == '__main__':
    h = Hidden(input_shape=2, no_units=3, activation='relu')

    # check dense functon
    h.dense(X=tf.constant([1., 2.]),
            W=tf.constant([[1., 1.], [1., 2.], [3., 4.]]),  # three hidden units
            b=tf.constant([0.5, 1., 1.]))

    # check init from prior & dense
    h.init = h.sample()
    h.prior_log_prob(h.init)
    h.dense(X=tf.constant([1., 2.]),
            W=h.init['W'],
            b=h.init['b'])

print('')
