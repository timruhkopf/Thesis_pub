import torch
import torch.nn as nn
import torch.distributions as td
import inspect

from Pytorch.Layer.Hidden import Hidden, Hidden_flat
from Pytorch.Util.Util_Model import Util_Model

from copy import deepcopy


class BNN(nn.Module, Util_Model):
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

        self.init_model = deepcopy(self.state_dict())

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
        return Util_Model.check_chain_seq(chain)


if __name__ == '__main__':
    no_in = 2
    no_out = 1
    n = 1000
    bnn = BNN(hunits=[no_in, 3, 2, no_out], activation=nn.ReLU(), prior='normal')

    # generate data
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)

    from copy import deepcopy

    bnn.reset_parameters()  # true_model
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

    # ------------------------------------------------------
    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib
    import random
    import os
    import traceback

    matplotlib.use('Agg')  # 'TkAgg' for explicit plotting

    sampler_name = ['SGNHT', 'SGLD', 'MALA', 'RHMC', 'SGRLD', 'SGRHMC'][3]
    model = bnn
    # Setting up the parameters  -----------------------------------------------
    sg_batch = 100
    # from pathlib import Path

    # home = str(Path.home())
    #
    # path = home + '/results_bnn/'
    path = '/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/Results/Results_BNN/'
    if not os.path.isdir(path):
        os.mkdir(path)

    for rep in range(3):
        for L in [1, 2, 3]:
            for eps in np.arange(0.001, 0.03, 0.005):
                model.reset_parameters()
                name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))

                sampler_param = dict(
                    epsilon=eps, num_steps=1000, burn_in=100,
                    pretrain=False, tune=False, num_chains=1)

                if sampler_name in ['SGNHT', 'RHMC', 'SGRHMC']:
                    sampler_param.update(dict(L=L))

                if sampler_name == 'SGRHMC':
                    sampler_param.update(dict(alpha=0.2))

                if 'SG' in sampler_name:
                    batch_size = sg_batch
                else:
                    batch_size = X.shape[0]

                trainset = TensorDataset(X, y)

                # Setting up the sampler & sampling
                if sampler_name in ['SGNHT', 'SGLD', 'MALA']:  # geoopt based models
                    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
                    Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                               'MALA': MALA,  # step_size
                               'SGLD': SGLD  # step_size
                               }[sampler_name]
                    sampler = Sampler(model, trainloader, **sampler_param)
                    try:
                        sampler.sample()
                        sampler.model.check_chain()
                        print(sampler.chain[:3])
                        print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------
                        sampler.model.plot(X[:100], y[:100], sampler.chain[:30], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)
                        matplotlib.pyplot.close('all')
                    except Exception as error:
                        print(name, 'failed')
                        sampler.model.plot(X[:100], y[:100], path=path + 'failed_' + name)
                        print(error)
                        print(traceback.format_exc())

                elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:
                    n_samples = sampler_param.pop('num_steps')
                    burn_in = sampler_param.pop('burn_in')
                    sampler_param.pop('pretrain')
                    sampler_param.pop('tune')
                    sampler_param.pop('num_chains')

                    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

                    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                               'SGRLD': myRSGLD,  # epsilon
                               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                               }['RHMC']
                    sampler = Sampler(model, **sampler_param)
                    try:
                        sampler.sample(trainloader, burn_in, n_samples)

                        sampler.model.check_chain()
                        print(sampler.chain[:3])
                        print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------

                        sampler.model.plot(X[:100], y[:100], sampler.chain[:30], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)
                        matplotlib.pyplot.close('all')
                    except Exception as error:
                        print(name, 'failed')
                        sampler.model.plot(X[:100], y[:100], path=path + 'failed_' + name)
                        print(error)
                        print(traceback.format_exc())

    print()
    # -------------------------------------------------------
    #
    # from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    # from torch.utils.data import TensorDataset, DataLoader
    #
    # burn_in, n_samples = 100, 100
    #
    # trainset = TensorDataset(X, y)
    # trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)
    #
    # Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
    #            'SGRLD': myRSGLD,  # epsilon
    #            'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
    #            }['RHMC']
    # sampler = Sampler(bnn, epsilon=0.01, L=2)
    # sampler.sample(trainloader, burn_in, n_samples)
    # sampler.check_chain()
    # import random
    #
    # sampler.model.plot(X[0:100], y[0:100], sampler.chain[:30])
    # sampler.model.plot(X[0:100], y[0:100], random.sample(sampler.chain, 30))
    # print(sampler.chain[0])
    # print(sampler.chain[-1])
    # print(sampler.model.true_model)
    # print(sampler.model.init_model)

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
