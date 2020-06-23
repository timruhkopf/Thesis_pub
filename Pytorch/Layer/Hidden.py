import torch
import torch.distributions as td


class Hidden:
    # TODO : see how to subclass torch layer
    activ = {'identity': lambda x: x}  # fixme: Torch.Identity?

    # 'tanh':
    # 'relu':
    # 'sigmoid':

    def __init__(self, no_in, no_units, bias=True, activation='relu'):
        self.no_in = no_in
        self.no_units = no_units
        self.bias = bias
        self.activation = self.activ[activation]

        self.parameters = ['W']
        # self.bijectors = ['Identity']  FIXME Bijectors: has Torch functionals?

        self.W_prior = td.Normal(0., 1.)  # fixme: check implementation with hyperparam tau & joint density!
        if self.bias is not None:
            self.parameters.append('b')
            # self.bijectors.append('Identity')
            self.b_prior = None

    def sample(self):
        self.W_prior.sample()  # fixme: check implementation
        if self.bias is not None:
            pass
        else:
            bias = 0.

    def log_prob(self):
        if self.bias is not None:
            pass

    def dense(self, X, W, b=None):
        if self.bias is not None:
            self.activation(X.mm(W))
        else:
            self.activation(X.mm(W) + b)
