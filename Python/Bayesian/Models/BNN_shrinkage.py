import tensorflow as tf
from Python.Bayesian.Models.BNN import Hidden

class Input(Hidden):
    def __init__(self, activation='relu'):
        """useing shrinkage priors for variable selection."""
        self.prior_W
        self.prior_b
        self.activation = {'relu': tf.nn.relu,
                           'tanh': tf.math.tanh}[activation]