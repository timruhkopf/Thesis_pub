import torch
import torch.nn as nn
import torch.distributions as td

from copy import deepcopy
from Pytorch.Util.Util_Model import Util_Model


class Hidden(nn.Module, Util_Model):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """
        Hidden Layer, that provides its prior_log_prob model

        :param no_in:
        :param no_out:
        :param bias: bool, whether or not to use bias in the computation
        :param activation: Any "nn.activation function" instance. Examine
        https://pytorch.org/docs/stable/nn.html?highlight=nn%20relu#torch.nn.ReLU
        should work with their functional correspondance e.g. F.relu. Notably,
        A final Layer in Regression setup looks like e.g.
        Hidden(10, 1, bias=False, activation=nn.Identity())
        """

        super().__init__()

        self.no_in = no_in
        self.no_out = no_out
        self.has_bias = bias
        self.activation = activation

        # Add the prior model
        self.dist = {}
        self.define_model()

        # initialize the parameters
        self.reset_parameters()
        self.true_model = None

    def define_model(self):
        self.tau_w = torch.tensor([1.])

        # self.tau_w = nn.Parameter(torch.tensor([1.]))
        # self.tau_w.distrib = td.Gamma(torch.tensor([2.]), torch.tensor([1.]))
        # self.tau_w.data = self.tau_w.distrib.sample()

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W.distrib = td.Normal(loc=torch.zeros_like(self.W.data), scale=self.tau_w)

        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.distrib = td.Normal(torch.zeros(self.no_out), 1.)

    @torch.no_grad()
    def reset_parameters(self):
        for p in self.parameters():
            p.data = p.distrib.sample()

        self.init_model = deepcopy(self.state_dict())

    def update_distributions(self):
        # here no hierarchical distributions exists
        # self.W.distrib.scale = torch.ones_like(self.W.distrib.scale) * self.tau_w
        return None

    # (2) DEFINE THESE FUNCTIONS IN REFERENCE TO SURROGATE PARAM: --------------
    # & inherit for layers
    def forward(self, X):
        XW = X @ self.W
        if self.has_bias:
            XW += self.b
        return self.activation(XW)

    def prior_log_prob(self):
        """evaluate each parameter in respective distrib."""
        value = torch.tensor(0.)
        for p in self.parameters():
            value += p.distrib.log_prob(p).sum()
        return value


class Hidden_flat(Hidden):
    def define_model(self):
        """
        formulate the model with truncated flat priors
        :return:
        """
        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W.distrib = td.Uniform(torch.ones(self.no_in, self.no_out) * -100.,
                                    torch.ones(self.no_in, self.no_out) * 100)

        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.distrib = td.Uniform(torch.ones(self.no_out) * -100.,
                                        torch.ones(self.no_out) * 100)




if __name__ == '__main__':
    from copy import deepcopy

    no_in = 2
    no_out = 1
    n = 1000

    # flat = Hidden_flat(no_in, no_out, bias=True, activation=nn.ReLU())
    # flat.true_model = deepcopy(flat.state_dict())
    # flat.forward(X=torch.ones(100, no_in))
    #
    # X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    # X = X_dist.sample(torch.Size([n]))
    # y = flat.likelihood(X).sample()
    #
    # flat.prior_log_prob()
    # # flat.reset_parameters()
    # flat.plot(X, y)
    #
    # reg = flat

    # single Hidden Unit Example
    reg = Hidden(no_in, no_out, bias=True, activation=nn.ReLU())
    reg.true_model = deepcopy(reg.state_dict())

    # reg.W = reg.W_.data
    # reg.b = reg.b_.data
    reg.forward(X=torch.ones(100, no_in))
    reg.prior_log_prob()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n]))
    y = reg.likelihood(X).sample()

    print(reg.log_prob(X, y))

    # reg.reset_parameters()
    reg.plot(X[:100], y[:100])

    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib
    import random
    import traceback

    matplotlib.use('Agg')  # 'TkAgg' for explicit plotting

    sampler_name = ['SGNHT', 'SGLD', 'MALA', 'RHMC', 'SGRLD', 'SGRHMC'][3]
    model = reg

    # Setting up the parameters  -----------------------------------------------
    sg_batch = 100
    for rep in range(15):
        for L in [1, 2, 3]:
            for eps in np.arange(0.001, 0.008, 0.002):
                model.reset_parameters()
                name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))
                path = '/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/Results/Results_Hidden1/'
                sampler_param = dict(
                    epsilon=eps, num_steps=200, burn_in=200,
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

                    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

                    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                               'SGRLD': myRSGLD,  # epsilon
                               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                               }['RHMC']
                    sampler = Sampler(model, **sampler_param)
                    try:
                        sampler.sample(trainloader, burn_in, n_samples)

                        sampler.model.check_chain(sampler.chain)
                        print(sampler.chain[:3])
                        print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------

                        sampler.model.plot(X[:100], y[:100], sampler.chain[:30], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)

                    except Exception as error:
                        print(name, 'failed')
                        sampler.model.plot(X[:100], y[:100], path=path + 'failed_' + name)
                        print(error)
                        print(traceback.format_exc())

                    matplotlib.pyplot.close('all')
    print()
    # EXPERIMENTAL ---------------------------
    # x = x[:, 1]
    # a = acf(x, nlags=100, fft=True)
    # sgnht.acf = a
    # len(sgnht.chain) / (1 + 2 * sum(sgnht.acf))
    # print(sgnht.ess())
    # lag = np.arange(len(a))
    # # lag = np.arange(len(sgnht.acf))
    # plt.plot(lag, a)
    # plt.show()
    #
    # tsaplots.plot_acf(x, lags=100)
    #
    # # my own trials on autocorrelation plots
    # x = sgnht.chain_mat
    # x = x - np.mean(x, axis=0)
    # x = x[:, 0]
    #
    # # Display the autocorrelation plot of your time series
    # fig = tsaplots.plot_acf(x, lags=1000, fft=True)
    # plt.show()
    #
    # import tidynamics
    #
    # sgnht.acf = tidynamics.acf(x)[1: len(sgnht.chain) // 3]
    # lag = np.arange(len(sgnht.acf))
    # plt.plot(lag, sgnht.acf)
    # plt.show()
    #
    # sgnht.fast_autocorr(0)
