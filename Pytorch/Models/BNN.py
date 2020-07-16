import torch
import torch.nn as nn
import torch.distributions as td

from hamiltorch.util import flatten, unflatten
from functools import partial
from itertools import accumulate
import inspect

from Pytorch.Layer import *


class BNN(nn.Module):
    def __init__(self, hunits=[1, 10, 5, 1], activation=nn.ReLU(), final_activation=nn.Identity(), heteroscedast=False):
        """
        Bayesian Neural Network, consisting of hidden layers.
        :param hunits: list of integers, specifying the input dimensions of hidden units
        :param activation: each layers activation function (except the last)
        :param final_activation: activation function of the final layer.
        :param heteroscedast: bool: indicating, whether or not y's conditional
        variance y|x~N(mu, sigma²) is to be estimated as well
        :remark: see Hidden.activation doc for available activation functions
        """

        super().__init__()
        self.heteroscedast = heteroscedast
        self.hunits = hunits
        self.layers = nn.Sequential(
            *[Hidden(no_in, no_units, True, activation)
              for no_in, no_units in zip(self.hunits[:-2], self.hunits[1:-1])],
            Hidden(hunits[-2], hunits[-1], bias=False, activation=final_activation))

        if self.heteroscedast:
            self.sigma_ = nn.Parameter(torch.Tensor(1))
            self.dist_sigma = td.TransformedDistribution(td.Gamma(0.01, 0.01), td.ExpTransform())
            self.sigma = self.dist_sigma.sample()
        else:
            self.sigma = torch.tensor(1.)

        self.true_model = None

    @property
    def vec(self):
        return torch.cat([h.vec for h in self.layers])

    @property
    def parameters_list(self):
        return [h.parameters_list for h in self.layers]

    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        p_log_prob = sum([h.prior_log_prob().sum() for h in self.layers])

        if self.heteroscedast:
            p_log_prob += self.dist_sigma.log_prob(self.sigma)

        return p_log_prob

    def likelihood(self, X):
        """:returns the conditional distribution of y | X"""
        return td.Normal(self.forward(X), scale=self.sigma)

    def log_prob(self, X, y, vec):
        """SG flavour of Log-prob: any batches of X & y can be used"""
        self.vec_to_attrs(vec)  # parsing to attributes
        return self.prior_log_prob() + \
               self.likelihood(X).log_prob(y).sum()

    def closure_log_prob(self, X=None, y=None):
        """log_prob factory, to fix X & y for samplers operating on the entire
        dataset.
        :returns None. changes inplace attribute log_prob"""
        print('Setting up "Full Dataset" mode')
        self.log_prob = partial(self.log_prob, X, y)

    def vec_to_attrs(self, vec):
        """surrogate for Hidden Layers' vec_to_attrs, but actually needs to sett a
        BNN.attribute in case of e.g. heteroscedasticity,
        i.e. where sigma param is in likelihood"""

        # delegate the vector parts to the layers
        lengths = list(accumulate([0] + [h.n_params for h in self.layers]))
        for i, j, h in zip(lengths, lengths[1:], self.layers):
            h.vec_to_attrs(vec[i:j])

        if self.heteroscedast:
            self.__setattr__('sigma', vec[-1])

    def reset_parameters(self, seperated=False):
        inspected = inspect.getfullargspec(self.layers[0].reset_parameters).args
        if 'seperated' in inspected:
            self.layers[0].reset_parameters(seperated)
        else:
            # default case: all Hidden units, no shrinkage -- has no seperated attr.
            self.layers[0].reset_parameters()

        for h in self.layers[1:]:
            h.reset_parameters()



if __name__ == '__main__':
    bnn = BNN(hunits=[1, 10, 5, 1])

    # generate data
    X_dist = td.Uniform(torch.tensor(-10.), torch.tensor(10.))
    X = X_dist.sample(torch.Size([100])).view(100, 1)
    X.detach()
    X.requires_grad_()

    y = bnn.likelihood(X).sample()

    # check forward path
    bnn.layers(X)
    bnn.forward(X)

    bnn.reset_parameters()
    bnn.true_model = bnn.vec

    # check vec_to_attrs
    bnn.vec_to_attrs(torch.cat([i * torch.ones(h.n_params) for i, h in enumerate(bnn.layers)]))
    bnn.parameters_list
    bnn.vec_to_attrs(torch.ones(80))
    bnn.forward(X)

    # check accumulation of parameters & parsing
    bnn.log_prob(X, y, flatten(bnn))

    # check log_prob "full dataset"
    bnn.closure_log_prob(X, y)
    bnn.log_prob(flatten(bnn))

    # check estimation
    import hamiltorch
    import hamiltorch.util

    N = 200
    step_size = .3
    L = 5

    bnn.reset_parameters()

    init_theta = hamiltorch.util.flatten(bnn)
    params_hmc = hamiltorch.sample(
        log_prob_func=bnn.log_prob, params_init=init_theta, num_samples=N,
        step_size=step_size, num_steps_per_sample=L)

    # check heteroscedastic case
    het_bnn = BNN(heteroscedast=True)
    het_bnn.forward(X)
    het_bnn.closure_log_prob(X, y)
    het_bnn.log_prob(flatten(het_bnn))

    N = 3000
    step_size = .3
    L = 5

    init_theta = hamiltorch.util.flatten(bnn)
    params_hmc = hamiltorch.sample(
        log_prob_func=bnn.log_prob, params_init=init_theta, num_samples=N,
        step_size=step_size, num_steps_per_sample=L)

    print()
