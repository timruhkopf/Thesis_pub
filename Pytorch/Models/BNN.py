import torch
import torch.nn as nn
import torch.distributions as td
import torch.distributions.transforms as bij

from itertools import chain
from functools import partial
import Pytorch.utils as utils
from Pytorch.Layer.Hidden import Hidden


class BNN(nn.Module):
    # coding example of BNN with MCMC
    # https://github.com/OscarJHernandez/bayesian_neural_networks/blob/master/Code/BNN.py
    activ = Hidden.activ

    def __init__(self, hunits=[1, 10, 5], activation='relu', final_activation='identity'):
        """
        Bayesian Neural Network, consisting of hidden layers.
        :param hunits: list of integers, specifying the input dimensions of hidden units
        :param activation: each layers activation function (except the last)
        :param final_activation: activation function of the final layer.
        :remark: see Hidden.activ for available activation functions
        """
        super().__init__();
        self.hunits = hunits

        # LAYER PRIOR MODEL
        # for the definition of adaptive layer sizes, consult
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/19
        # on ModuleList & Sequential to register the parameters
        self.layers = [Hidden(no_in, no_units, activation)
                       for no_in, no_units in zip(self.hunits[:-2], self.hunits[1:-1])]
        self.layers.append(Hidden(no_in=hunits[-2], no_units=hunits[-1],
                                  bias=False, activation=final_activation))
        self.module_layers = nn.ModuleList(self.layers)
        # self.Ls = nn.Sequential(*self.layers) # DEPREC

        # Variance Prior model
        self.SIGMA = td.TransformedDistribution(td.Gamma(0.1, 0.1), bij.ExpTransform())
        self.sigma = nn.Parameter(self.SIGMA.sample())

        self.log_prob = None  # placeholder for closure to be set

    def sample(self):
        """delegate the sampling of parameters to each layer"""
        for h in self.layers:
            h.sample()

    def forward(self, X):
        for h in self.module_layers:
            X = h(X)  # call to "dense method XW' "
        return X

    def prior_log_prob(self):
        return torch.stack([l.prior_log_prob() for l in self.layers], dim=0).sum(dim=0)

    def likelihood(self, X, sigmasq):
        return td.MultivariateNormal(self.forward(X), sigmasq * torch.eye(X.shape[0]))

    def closure_log_prob(self, X=None, y=None):
        """

        :param X: (Optional) Batched Tensor.
        :param y: (Optional) Batched Tensor.
        :return: if both X & y are provided, a log_prob function is added to bnn,
        which has pre set X & y, making it a full-dataset log_prob. If neither are
        supplied, the added log_prob function requires X & y's for every call on
        log_prob. However, this enables the SG behaviour of the log_prob function.
        """

        def log_prob(self, X, y, theta):
            """By default works as an SG flavour, can be fixed using self.fix_log_prob(X,y)
            to make it a full dataset sampler"""

            # INPLACE ASSIGNMENT OF THETA ON NET WEIGHTS & param
            utils.unflatten(vec=theta, model=self)

            return self.likelihood(X, self.sigma).log_prob(y).sum() + \
                   self.SIGMA.log_prob(self.sigma) + \
                   self.prior_log_prob()

        if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
            print('"Full Dataset" log prob is being set up...')
            if any([X is None, y is None]):
                raise ValueError('"Full Dataset" modus requires specification of X & y')
            self.log_prob = partial(log_prob, self, X, y)
        else:
            print('SG log prob is being set up...')
            self.log_prob = partial(log_prob, self)


if __name__ == '__main__':
    # check bnn functions on simple regression data
    # data pre-setup
    from Pytorch.Layer.Hidden import Hidden
    from Pytorch.utils import flatten, unflatten

    reg = Hidden(no_in=2, no_units=1, bias=True, activation='identity')
    X_dist = td.Uniform(torch.tensor([-10., -10]), torch.tensor([10., 10]))
    X = X_dist.sample(torch.Size([100]))
    beta = torch.tensor([[1., 2.]])
    reg.linear.weight = nn.Parameter(beta)
    mu = reg(X)

    bnn = BNN(hunits=[2, 10, 1], activation='relu', final_activation='identity')

    # generate y!
    y = td.Normal(mu, torch.tensor(1.)).sample()

    bnn.forward(X)
    l = bnn.likelihood(X, torch.tensor(1.))
    l.log_prob(y).sum()

    # test full dataset log_prob
    bnn.closure_log_prob(X, y)
    bnn.sample()
    theta = flatten(bnn)
    # theta = torch.ones(41) *2
    print(bnn.log_prob(theta))

    # TODO: test sg log_prob:
    #   check that each and every sg sampler accepts this format
    #   particularly that theta can be at the last postion!
    bnn.closure_log_prob()
    bnn.log_prob(X=X, y=y, theta=theta)

    # PARTIAL FUNCTION EXAMPLE
    # from functools import partial
    # # A normal function
    # def f(a, b, c, x):
    #     return 1000 * a + 100 * b + 10 * c + x
    # # A partial function that calls f with
    # # a as 3, b as 1 and c as 4.
    # g = partial(f, 3, 1, 4)
    #
    # # Calling g()
    # print(g(5))
