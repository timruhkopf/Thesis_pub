import torch
import torch.nn as nn
import torch.distributions as td
from torch import optim
from copy import deepcopy
from src.Util.Util_Model import Util_Model


class Hidden(nn.Module, Util_Model):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """
        Hidden Layer, that provides its prior_log_prob model

        :param no_in:
        :param no_out:
        :param bias: bool, whether or not to use bias in the computation
        :param activation: Any "nn.activation function" instance. Examine
        https://src.org/docs/stable/nn.html?highlight=nn%20relu#torch.nn.ReLU
        should work with their functional correspondance e.g. F.relu. Notably,
        A final Layer in Regression setup looks like e.g.
        Hidden(10, 1, bias=False, activation=nn.Identity())
        """

        super().__init__()

        self.no_in = no_in
        self.no_out = no_out
        self.has_bias = bias
        self.activation = activation

        self.define_model()

        # initialize the parameters
        self.reset_parameters()
        self.true_model = deepcopy(self.state_dict())

    def define_model(self):
        self.tau_w = torch.tensor([1.])

        # self.tau_w = nn.Parameter(torch.tensor([1.]))
        # self.tau_w.distrib = td.Gamma(torch.tensor([2.]), torch.tensor([1.]))
        # self.tau_w.data = self.tau_w.distrib.sample()

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W.dist = td.Normal(loc=torch.zeros_like(self.W.data), scale=self.tau_w)

        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.dist = td.Normal(torch.zeros(self.no_out), 1.)

    @torch.no_grad()
    def reset_parameters(self):
        for p in self.parameters():
            p.data = p.dist.sample()

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
            value += p.dist.log_prob(p).sum()
        return value

    def sample_model(self, n):
        self.true_model = deepcopy(self.state_dict())
        X_dist = td.Uniform(torch.ones(self.no_in) * (-10.), torch.ones(self.no_in) * 10.)
        X = X_dist.sample(torch.Size([n]))
        y = self.likelihood(X).sample()

        return X, y


class Hidden_flat(Hidden):
    def define_model(self):
        """
        formulate the model with truncated flat priors
        :return:
        """
        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W.dist = td.Uniform(torch.ones(self.no_in, self.no_out) * -100.,
                                 torch.ones(self.no_in, self.no_out) * 100)

        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.dist = td.Uniform(torch.ones(self.no_out) * -100.,
                                     torch.ones(self.no_out) * 100)


if __name__ == '__main__':
    from copy import deepcopy

    no_in = 1
    no_out = 1
    n = 1000

    # single Hidden Unit Example
    reg = Hidden(no_in, no_out, bias=True) #, activation=nn.Identity())
    X, y = reg.sample_model(n)
    reg.reset_parameters()

    reg.forward(X=torch.ones(100, no_in))

    # (Pretraining) ------------------------------------------------------------
    optimizer = optim.SGD(reg.parameters(), lr=0.1)
    # criterion = nn.MSELoss()
    criterion = reg.log_prob
    LOSS = []
    CHAIN = []
    n = 1
    from tqdm import tqdm

    for epoch in tqdm(range(500)):
        # x, Y = next(data._get_iterator())
        yhat = reg(X)
        # loss = criterion(yhat, y) + (reg.W.data.t() @ reg.W.data)[0][0]
        # loss = -criterion(X, y)

        # explicit posterior:
        # prior:
        value = torch.tensor([0.])
        for p in reg.parameters():
            value += p.dist.log_prob(p.data).sum()

        # posterior = like + prior
        # - due to optim!
        loss = -X.shape[0]**-1 *(reg.likelihood(X).log_prob(y).sum() + value)

        print('loss', loss)
        print('penMSE:', nn.MSELoss()(yhat, y) + (reg.W.data.t() @ reg.W.data)[0][0])
        optimizer.zero_grad()
        loss.backward()
        for name, p in reg.named_parameters():
            print(name, 'grad:', p.grad, '\n', 'current', p.data)
        optimizer.step()
        LOSS.append(loss)

        # LOSS.append(reg.log_prob(X, y))
        CHAIN.append(deepcopy(reg.state_dict()))

    print('true model', reg.true_model)
    print('current model', reg.state_dict())

    print('init model', reg.init_model)
    print(reg.log_prob(X, y))
    #

    # (PLOTTING THE TRAVERSAL) ---------------------------------------------------
    # from numpy import exp, arange
    # from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
    #
    # # the function that I'm going to plot
    # # def z_func(x, y):
    # #     return (1 - (x ** 2 + y ** 3)) * exp(-(x ** 2 + y ** 2) / 2)
    #
    # z_func = reg.log_prob
    #
    # a = arange(-10.0, 10., 0.2)  # x1
    # b = arange(-10.0, 10.0, 0.2)  # x2
    # A, B = meshgrid(a, b)  # grid of point
    #
    # params = torch.stack([torch.flatten(torch.tensor(A, dtype=torch.float)),
    #                       torch.flatten(torch.tensor(B, dtype=torch.float))]).t()
    # from collections import OrderedDict
    #
    # loss = []
    # for p in params:
    #     reg.load_state_dict(OrderedDict({'W': torch.reshape(p, [2, 1])}))
    #
    #     loss.append(z_func(X, y))  # evaluation of the function on the grid
    #     # loss.append(criterion(reg.forward(X), y) +torch.tensor([1.0]) )
    # Z = torch.reshape(torch.tensor(loss), A.shape).numpy()
    #
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter
    # import matplotlib.pyplot as plt
    #
    # import matplotlib
    #
    # matplotlib.use('TkAgg')
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(A, B, Z, rstride=1, cstride=1,
    #                        cmap=cm.RdBu, linewidth=0, antialiased=False)
    #
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax1 = fig.gca(projection='3d')
    # ax1.plot(torch.flatten(torch.stack([d['W'][0] for d in CHAIN[:4]])).numpy(),
    #          torch.flatten(torch.stack([d['W'][1] for d in CHAIN[:4]])).numpy(),
    #          torch.flatten(torch.stack(loss[:4]).detach()).numpy())
    # plt.show()
    #
    # # reg.reset_parameters()
    # import matplotlib
    #
    # matplotlib.use('TkAgg')
    # reg.plot(X[:100], y[:100], chain=CHAIN[-20:])

    from src.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib
    import random
    import traceback

    matplotlib.use('Agg')  # 'TkAgg' for explicit plotting

    sampler_name = ['SGNHT', 'SGLD', 'MALA', 'RHMC', 'SGRLD', 'SGRHMC'][0]
    model = reg


    # Setting up the parameters  -----------------------------------------------
    sg_batch = 1000
    for rep in range(1):
        for L in reversed([1, 2, 3]):
            for eps in np.arange(0.0003, 0.0001, -0.00003):
                model.reset_parameters()
                print('init avg MSE:', nn.MSELoss()(y, model.forward(X)))
                name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))
                path = '/home/tim/PycharmProjects/Thesis/Experiments/Results_Hidden4/'
                sampler_param = dict(
                    epsilon=eps, num_steps=1000, burn_in=1000,
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
                               }[sampler_name]
                    sampler = Sampler(model, **sampler_param)
                    try:
                        sampler.sample(trainloader, burn_in, n_samples)

                        sampler.model.check_chain(sampler.chain)
                        print(sampler.chain[:3])
                        print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------
                        # import matplotlib
                        # matplotlib.use('TkAgg')
                        # sampler.model.plot(X[:100], y[:100], sampler.chain[-30:])
                        # sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30))

                        print('avg MSE:', nn.MSELoss()(y, model.forward(X)))
                        sampler.model.plot(X[:100], y[:100], sampler.chain[-30:], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name +
                                                                                                    'random')

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
