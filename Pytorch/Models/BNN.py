import torch
from itertools import chain
from Pytorch.Layer.Hidden import Hidden


class BNN:
    activ = Hidden.activ

    def __init__(self, hunits=[1, 10, 5], activation='relu'):
        self.hunits = hunits
        self.layers = [Hidden(no_in, no_units, activation)
                       for no_in, no_units in zip(self.hunits[:-2], self.hunits[1:-1])]
        self.layers.append(Hidden(input_shape=hunits[-2], no_units=hunits[-1],
                                  bias=False, activation='identity'))

    @property
    def parameters(self):
        return list(chain(*[list(h.parameters) for h in self.layers]))

    @property
    def bijectors(self):
        return list(chain(*[list(h.bijectors) for h in self.layers]))

    def forward(self, X, param):
        pass

    def sample(self):
        """sample the Prior Model"""
        pass

    def log_prob(self):
        pass
