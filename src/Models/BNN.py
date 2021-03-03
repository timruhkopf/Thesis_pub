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

    # ------------------------------------------------------
    from src.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
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
    path = '/home/tim/PycharmProjects/Thesis/Experiments/Results_BNN2/'
    if not os.path.isdir(path):
        os.mkdir(path)

    for rep in range(3):
        for L in [1, 2]:
            for eps in np.arange(0.003, 0.001, -0.0005):
                model.reset_parameters()
                name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))
                print(name)

                sampler_param = dict(
                    epsilon=eps, num_steps=3000, burn_in=100,
                    pretrain=False, tune=False, num_chains=1)

                if sampler_name in ['SGNHT', 'RHMC', 'SGRHMC']:
                    sampler_param.update(dict(L=L))

                if sampler_name == 'SGRHMC':
                    sampler_param.update(dict(alpha=0.8))

                if 'SG' in sampler_name:
                    batch_size = sg_batch
                else:
                    batch_size = X.shape[0]

                eps = batch_size ** (-1) * eps
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
                        sampler.model.check_chain(sampler.chain)
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

                    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

                    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                               'SGRLD': myRSGLD,  # epsilon
                               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                               }[sampler_name]
                    sampler = Sampler(model, **sampler_param)
                    try:
                        sampler.sample(trainloader, burn_in, n_samples)

                        sampler.model.check_chain(sampler.chain)
                        # print(sampler.chain[:3])
                        # print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------

                        # sampler.model.plot(X[:100], y[:100], sampler.chain[:30], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)
                        print('last avg MSE:', nn.MSELoss()(y, model.forward(X)))
                        model.load_state_dict(model.init_model)
                        print('init avg MSE:', nn.MSELoss()(y, model.forward(X)))
                        matplotlib.pyplot.close('all')
                    except Exception as error:
                        print(name, 'failed')
                        sampler.model.plot(X[:100], y[:100], path=path + 'failed_' + name)
                        print(error)
                        print(traceback.format_exc())

    print()
