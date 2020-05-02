from collections import OrderedDict
from Python.Bayesian.layers.Hidden import Hidden
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

class HiddenFinal(Hidden):
    def __init__(self, input_shape, no_units=10, activation='relu'):
        super().__init__(input_shape, no_units, activation)

        tau = tf.constant([1.])
        self.joint = tfd.JointDistributionNamed(OrderedDict(
            # tau=tfd.InverseGamma(1., 1.),
            W=tfd.Sample(  # lambda tau:
                distribution=tfd.Normal(0., tau),
                sample_shape=(no_units, input_shape)),
        ), name='HiddenFinal')

        self.parameters = list(self.joint._parameters['model'].keys())
        self.bijectors = {'W': tfb.Identity()}

    @tf.function
    def dense(self, X, W):
        return self.activation(tf.linalg.matvec(W, X))



if __name__ == '__main__':
    # check HiddenFinal:
    hf = HiddenFinal(input_shape=2, no_units=3, activation='relu')
    hf.dense(X=tf.constant([1., 2.]),
             W=tf.constant([[1., 1.], [1., 2.], [3., 4.]]))
    hf.init = hf.sample()
    hf.prior_log_prob(hf.init)


