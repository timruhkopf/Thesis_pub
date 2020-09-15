import torch
import torch.nn as nn
import torch.distributions as td
import inspect

from Pytorch.Layer.Hidden import Hidden, Hidden_flat
from Pytorch.Util.ModelUtil import Model_util


class BNN(nn.Module, Model_util):
    L = {'flat': Hidden_flat, 'normal': Hidden}

    def __init__(self, hunits=[1, 10, 5, 1], activation=nn.ReLU(), final_activation=nn.Identity(),
                 heteroscedast=False, prior='normal'):
        """
        Bayesian Neural Network, consisting of hidden layers.
        :param hunits: list of integers, specifying the input dimensions of hidden units
        :param activation: each layers activation function (except the last)
        :param final_activation: activation function of the final layer.
        :param heteroscedast: bool: indicating, whether or not y's conditional
        variance y|x~N(mu, sigma²) is to be estimated as well
        :remark: see Hidden.activation doc for available activation functions
        """
        nn.Module.__init__(self)
        self.heteroscedast = heteroscedast
        self.prior = prior
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation

        self.define_model()
        self.reset_parameters()

        self.true_model = None

    # CLASSICS METHODS ---------------------------------------------------------
    def define_model(self):
        L = self.L[self.prior]
        # Defining the layers depending on the mode.
        self.layers = nn.Sequential(
            *[L(no_in, no_units, True, self.activation)
              for no_in, no_units in zip(self.hunits[:-2], self.hunits[1:-1])],
            L(self.hunits[-2], self.hunits[-1], bias=False, activation=self.final_activation))

        if self.heteroscedast:
            self.sigma_ = nn.Parameter(torch.Tensor(1))
            self.dist_sigma = td.TransformedDistribution(td.Gamma(0.01, 0.01), td.ExpTransform())
            self.sigma = self.dist_sigma.sample()
        else:
            self.sigma = torch.tensor(1.)

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

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        p_log_prob = sum([h.prior_log_prob().sum() for h in self.layers])

        if self.heteroscedast:
            p_log_prob += self.dist_sigma.log_prob(self.sigma)

        return p_log_prob

    def likelihood(self, X):
        """:returns the conditional distribution of y | X"""
        return td.Normal(self.forward(X), scale=self.sigma)

    # SURROGATE (AGGREGATING) METHODS ------------------------------------------
    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)

    def update_distributions(self):
        for h in self.layers:
            h.update_distributions()

    @staticmethod
    def check_chain(chain):
        return Model_util.check_chain_seq(chain)


if __name__ == '__main__':
    no_in = 2
    no_out = 1
    n = 1000
    bnn = BNN(hunits=[no_in, 10, 5, no_out], activation=nn.ReLU(), prior='flat')

    # generate data
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)

    from copy import deepcopy
    bnn.reset_parameters() # true_model
    bnn.true_model = deepcopy(bnn.state_dict())
    y = bnn.likelihood(X).sample()

    bnn.reset_parameters()
    bnn.plot(X, y)

    # check forward path
    bnn.layers(X)
    bnn.forward(X)

    bnn.prior_log_prob()

    # check accumulation of parameters & parsing
    bnn.log_prob(X, y)

    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']
    sampler = Sampler(bnn, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    # from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    #
    # param = dict(epsilon=0.001,
    #              num_steps=5000,  # <-------------- important
    #              pretrain=False,
    #              tune=False,
    #              burn_in=2000,
    #              # num_chains 		type=int, 	default=1
    #              num_chains=1,  # os.cpu_count() - 1
    #              L=24)
    #
    # batch_size = 50
    # val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    # val_prediction_steps = 50
    # val_converge_criterion = 20
    # val_per_epoch = 200
    #
    # bnn.reset_parameters()
    # sgnht = SGNHT(bnn, trainloader, **param)
    # sgnht.sample()
    # print(sgnht.chain)
    #
    # import random
    # sgnht.model.plot(X, y, random.sample(sgnht.chain, 30))
    #
    # sgnht.model.plot(X, y, sgnht.chain[0:10])
