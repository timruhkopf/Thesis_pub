import torch
import torch.distributions as td
import torch.nn as nn
from copy import deepcopy

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Util.Util_Distribution import LogTransform

from Pytorch.Util.Util_bspline import get_design, diff_mat1D


class GAM(Hidden):
    def __init__(self, xgrid=(0, 10, 0.5), order=1, no_basis=20, no_out=1,
                 activation=nn.Identity(), bijected=False):
        """
        RandomWalk Prior Model on Gamma (W) vector.
        Be carefull to transform the Data beforehand with some DeBoor Style algorithm.
        This Module mereley implements the Hidden behaviour + Random Walk Prior
        :param no_basis: number of basis from the de boor basis expansion to expect (=no_columns of X)
        which is in fact the no_in of Hidden.
        :param order: difference order to create the Precision (/Penalty) matrix K
        """
        self.bijected = bijected
        self.xgrid = xgrid
        self.order = order
        self.no_basis = no_basis

        Hidden.__init__(self, no_basis, no_out, bias=False, activation=activation)

    def define_proper_cov(self):
        # replace the numerical zero eigenvalue by fraction*(smallest non-zero eigenval)
        # to ensure a propper distribution (see Marra Wood or Wood JAGS)
        threshold = 1e-3
        fraction = 1e-1
        self.K = torch.tensor(diff_mat1D(self.no_basis, self.order)[1], dtype=torch.float32, requires_grad=False)
        val, vec = torch.eig(self.K, eigenvectors=True)
        eig_val_2nd = torch.sort(val, axis=0).values[1, :]
        val[val[:, 0] < threshold, :] = eig_val_2nd * fraction
        self.penK = vec @ torch.diag(val[:, 0]) @ vec.t()
        self.cov = torch.inverse(self.penK).detach()

    def define_model(self):
        # setting up a proper covariance for W's random walk prior
        self.define_proper_cov()

        self.tau = nn.Parameter(torch.Tensor(1))
        self.dist['tau'] = td.Gamma(torch.tensor([2.]), torch.tensor([2.]))
        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))

        if self.bijected:
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())
            self.tau.data = self.dist['tau'].sample()
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)
            self.dist['W'] = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                   self.tau_bij ** -1 * self.cov)

        else:
            self.tau.data = self.dist['tau'].sample()  # to ensure dist W is set up properly
            self.dist['W'] = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                   self.tau ** -1 * self.cov)

    def update_distributions(self):
        # tau_bij is the actual variance parameter of W (on R+)- whilest if self.bijected==True,
        # tau becomes the bijected i.e. unconstrained parameter on entire R
        if self.bijected:
            # here tau (is on R) ---> tau_bij (on R+)
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)

            try:
                self.dist['W'] = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                       self.tau_bij ** -1 * self.cov)
            except Exception as e:
                print(self.tau_bij)
                print(self.tau_bij ** -1 * self.cov)
                raise RuntimeError('GAM_update: cannot update distribution, as covariance is invalid:\n'
                                   'tau:{}\ntau_bij:{}\ncov:{}'.format(self.tau, self.tau_bij,
                                                                       self.tau_bij ** -1 * self.cov))
        else:
            self.dist['W'] = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                   self.tau ** -1 * self.cov)

    def reset_parameters(self, tau=None):
        """
        Sample the prior model, instantiating the data model
        :param xgrid: defining space for the Bspline expansion. Notice, that
        int((xgrid[1] - xgrid[0]) // 0.5) == self.no_basis is required!
        :param tau: if not None, it is the inverse variance (smoothness) of
        randomwalkprior. If None, the self.dist_tau is used for sampling tau
        # FIXME: carefull with sigma/tau=lambda relation
        :return: None. Inplace self.W, self.tau
        """

        if int((self.xgrid[1] - self.xgrid[0]) // 0.5) != self.no_basis:
            raise ValueError('The Specified range(*xgrid) does not imply '
                             'no_basis (len(range) must be equal to no_basis)')
        if tau is None:
            # FIXME: carefull with transformed distributions! (need to transform tau back)
            self.tau.data = self.dist['tau'].sample()

        else:
            self.tau.data = tau
        self.update_distributions()

        self.W.data = self.dist['W'].sample().view(self.no_basis, 1)

        # gamma = torch.cat(
        #     [td.Uniform(torch.tensor([-1.]), torch.tensor([1.])).sample(),
        #      td.MultivariateNormal(torch.zeros(self.no_basis - 1), (self.tau) * self.cov).sample()],
        #     dim=0).view(self.no_out, self.no_basis)
        # self.W.data = gamma.view(self.no_basis, self.no_out)
        self.init_model = deepcopy(self.state_dict())

    def prior_log_prob(self):
        """
        returns: log_probability sum of gamma & tau and is calculating the
         RandomWalkPrior
        """
        # FIXME: CHECK IF IDENTIFIABILITY CAN BE HELPED IF SELF.K were penalized

        # fixme: check if tau is correct here! (and not 1/tau)
        #  BE VERY CAREFULL IF TAU IS BIJECTED!
        # p321 fahrmeir kneib lang:
        # const = - 0.5 * (self.K.shape[0] - 1) * torch.log(self.tau_bij)
        # kernel = -(2 * self.tau_bij) ** -1 * self.W.t() @ self.K @ self.W
        # return sum(const + kernel + self.dist['tau'].log_prob(self.tau))  # notice that tau can be on R if
        # # self.bijected is true!

        return sum(self.dist['W'].log_prob(self.W)) + self.dist['tau'].log_prob(self.tau)

    def plot(self, X, y, chain=None, path=None, title='', **kwargs):
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=self.no_basis), dtype=torch.float32,
                         requires_grad=False)
        df0 = self.predict_states(chain, Z)

        df0['X'] = X.view(X.shape[0], ).numpy()
        df1 = df0.melt('X', value_name='y')
        df1 = df1.rename(columns={'variable': 'functions'})
        plt = self._plot1d(X, y, df1, **kwargs)

        if path is None:
            plt.show()
        else:
            plt.savefig('{}.png'.format(path), bbox_inches='tight')


if __name__ == '__main__':


    # dense example
    no_basis = 20
    n = 1000
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([n]))
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis),
                     dtype=torch.float32, requires_grad=False)

    gam = GAM(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=True)
    # gam = GAM(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=False)

    gam.reset_parameters(tau=torch.tensor([0.001]))
    # gam.reset_parameters(tau=torch.tensor([0.01]))
    gam.true_model = deepcopy(gam.state_dict())
    y = gam.likelihood(Z).sample()

    gam.reset_parameters(tau=torch.tensor([1.]))

    gam.plot(X, y)

    gam.forward(Z)
    gam.prior_log_prob()


    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib
    import random
    import os

    matplotlib.use('Agg')  # 'TkAgg' for explicit plotting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.get_device_name(0)
    Z.to(device)
    y.to(device)

    from pathlib import Path

    home = str(Path.home())
    if '/PycharmProjects' in __file__:
        # file is on local machine
        home += '/PycharmProjects'
    path = home + '/Thesis/Pytorch/Experiments/Results_GAM/'
    if not os.path.isdir(path):
        os.mkdir(path)

    sampler_name = ['SGNHT', 'SGLD', 'MALA', 'RHMC', 'SGRLD', 'SGRHMC'][1]
    model = gam

    # Setting up the parameters  -----------------------------------------------
    sg_batch = 100
    for rep in range(3):
        for L in [1, 2, 3]:
            for eps in np.arange(0.007, 0.04, 0.003):
                model.reset_parameters(tau=torch.tensor([0.0001]))  # initialization
                name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))

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

                trainset = TensorDataset(Z, y)

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

    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 100

    trainset = TensorDataset(Z, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']
    sampler = Sampler(gam, epsilon=0.01, L=2)
    sampler.sample(trainloader, burn_in, n_samples)
    sampler.check_chain()
    import random

    sampler.model.plot(X[0:100], y[0:100], sampler.chain[:30])
    sampler.model.plot(X[0:100], y[0:100], random.sample(sampler.chain, 100))
    print(sampler.chain[0])
    print(sampler.chain[-1])

    step_size = 0.005
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    L = 24
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    # hmc = HMC(reg, X, y, X.shape[0],
    #           step_size=step_size, num_steps=num_steps, burn_in=burn_in, pretrain=pretrain, tune=tune,
    #           traj_length=hmc_traj_length,
    #           num_chains=num_chains)
    # hmc.sample()

    from Pytorch.Samplers.LudwigWinkler import SGNHT
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(Z, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    sgnht = SGNHT(gam, trainloader, epsilon=step_size, num_steps=num_steps,
                  burn_in=burn_in, pretrain=pretrain, tune=tune,
                  L=L, num_chains=num_chains)
    sgnht.sample()

    print(sgnht.chain)
    print(sgnht.chain)
    if len(sgnht.chain) == 1:
        raise ValueError('The chain did not progress beyond first step')

    if all(any(torch.isnan(v)) for chain in (sgnht.chain[-1].values(), sgnht.chain[0]) for v in chain):
        raise ValueError('first and last entry contain nan')

    # plot via stratified chain
    import random

    gam.plot(X[1:100], y[1:100], chain=random.sample(sgnht.chain, 20))

    # Todo use penalized K in GAM for prior_log_prob to improve identifiably
