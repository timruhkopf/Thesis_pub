import torch
import torch.nn as nn
import torch.distributions as td

import inspect
from copy import deepcopy

from src.Layer.Hidden import Hidden
from src.Util.Util_Model import Util_Model


class BNN(nn.Module, Util_Model):

    def __init__(self, hunits=(1, 10, 5, 1), activation=nn.ReLU(), final_activation=nn.Identity()):
        """
        Bayesian Neural Network, consisting of hidden layers.
        :param hunits: list of integers, specifying the input dimensions of hidden units
        :param activation: each layers activation function (except the last)
        :param final_activation: activation function of the final layer.
        :remark: see Hidden.activation doc for available activation functions
        """
        nn.Module.__init__(self)

        self.hunits = hunits
        self.no_in = hunits[0]
        self.no_out = hunits[-1]
        self.activation = activation
        self.final_activation = final_activation

        self.define_model()
        self.reset_parameters()

        self.true_model = deepcopy(self.state_dict())

    # CLASSIC METHODS ---------------------------------------------------------
    def define_model(self):
        # Defining the layers depending on the mode.
        self.layers = nn.Sequential(
            *[Hidden(no_in, no_units, True, self.activation)
              for no_in, no_units in zip(self.hunits[:-2], self.hunits[1:-1])],
            Hidden(self.hunits[-2], self.hunits[-1], bias=False, activation=self.final_activation))

    def reset_parameters(self, seperated=False):
        """samples each layer individually.
        :param seperated: bool. indicates whether the first layer (given its
        local reset_parameters function has a seperated argument) will sample
        the first layers 'shrinkage' to create an effect for a variable, that is indeed
        not relevant to the BNNs prediction."""
        inspected = inspect.getfullargspec(self.layers[0].reset_parameters).args
        if 'seperated' in inspected:
            self.layers[0].reset_parameters(seperated)
        else:
            # default case: all Hidden units, no shrinkage -- has no seperated attr.
            self.layers[0].reset_parameters()

        for h in self.layers[1:]:
            h.reset_parameters()

        self.init_model = deepcopy(self.state_dict())

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        return sum([h.prior_log_prob().sum() for h in self.layers])

    # SURROGATE (AGGREGATING) METHODS ------------------------------------------
    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)

    def update_distributions(self):
        for h in self.layers:
            h.update_distributions()

    @staticmethod
    def check_chain(chain):
        return Util_Model.check_chain_seq(chain)

    def sample_model(self, n):
        X_dist = td.Uniform(torch.ones(self.no_in) * (-10.), torch.ones(self.no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, self.no_in)
        self.reset_parameters()  # true_model
        self.true_model = deepcopy(self.state_dict())
        y = self.likelihood(X).sample()

        return X, y


if __name__ == '__main__':
    n = 1000
    bnn = BNN(hunits=(1, 2, 5, 1), activation=nn.ReLU())
    X, y = bnn.sample_model(n)
    bnn.reset_parameters()
    bnn.plot(X, y)

    # check forward path
    bnn.layers(X)
    bnn.forward(X)

    bnn.prior_log_prob()

    # check accumulation of parameters & parsing
    bnn.log_prob(X, y)
