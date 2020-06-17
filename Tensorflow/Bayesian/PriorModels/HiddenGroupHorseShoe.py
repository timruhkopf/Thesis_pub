from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Tensorflow.Bayesian.PriorModels.Hidden import Hidden


class HiddenGroupHorseShoe(Hidden):
    def __init__(self, input_shape, no_units=10, activation='relu'):
        pass
