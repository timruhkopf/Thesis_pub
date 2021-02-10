import torch
import torch.distributions as td
import torch.nn as nn
from copy import deepcopy

from src.Layer.GAM import GAM


class GAM_fix_var(GAM):
    """
    MAIN PURPOSE OF THIS CLASS IS TO PROVIDE EVIDENCE THAT THE SMOOTHNESS PARAMETER
    / VARIANCE TAU IS NOT RESPONSIBLE FOR THE EXPLODING GRADIENTS - INSTEAD THE BNN
    PART OR ORTHOGONALISATION MOST LIKELY CAUSE THE PROBLEM!
    """

    def __init__(self, order=1, no_basis=20, no_out=1,
                 activation=nn.Identity(), tau=1.):
        """
        RandomWalk Prior Model with fixed variance on Gamma (W) vector.
        Be carefull to transform the Data beforehand with some DeBoor Style algorithm.
        This Module merely implements the Hidden behaviour + Random Walk Prior
        :param no_basis: number of basis from the de boor basis expansion to expect (=no_columns of X)
        which is in fact the no_in of Hidden.
        :param order: difference order to create the Precision (/Penalty) matrix K
        :param tau: the fix variance of the random walk prior
        """
        self.tau = torch.tensor([tau])
        GAM.__init__(self, order, no_basis, no_out, activation, bijected=False)

    def define_model(self):
        self.define_proper_cov()

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W.dist = td.MultivariateNormal(torch.zeros(self.no_basis),
                                            self.tau ** 2 * self.cov)

    def update_distributions(self):
        # No hierarchy
        pass

    def reset_parameters(self, tau=1.):
        self.W.data = self.W.dist.sample().view(self.no_basis, 1)
        self.init_model = deepcopy(self.state_dict())

    def prior_log_prob(self):
        return self.W.dist.log_prob(self.W).sum()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    gam = GAM_fix_var(tau=1., no_basis=10)

    n = 1000
    X, Z, y = gam.sample_model(n)
    gam.reset_parameters()
    gam.plot(X, y)

    gam.prior_log_prob()
    gam.likelihood(Z).log_prob(y)

    from src.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
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
    path = home + '/Thesis/Experiments/Results_GAM/'
    if not os.path.isdir(path):
        os.mkdir(path)

    sampler_name = ['SGNHT', 'SGLD', 'MALA', 'RHMC', 'SGRLD', 'SGRHMC'][3]
    model = gam

    # Setting up the parameters  -----------------------------------------------
    sg_batch = 100
    for rep in range(3):
        for L in [1, 2, 3]:
            for eps in np.arange(0.01, 0.04, 0.003):
                model.reset_parameters(tau=torch.tensor([0.0001]))  # initialization
                name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))

                sampler_param = dict(
                    epsilon=eps,
                    num_steps=1000, burn_in=100,
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
                        sampler.save('/home/tim/PycharmProjects/Thesis/Experiments/Results/Results_GAM/')
                        print(sampler.chain[:3])
                        print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------
                        sampler.model.plot(X[:100], y[:100], sampler.chain[:30], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)
                        sampler.traceplots(baseline=True)
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
                               }[sampler_name]
                    sampler = Sampler(model, **sampler_param)
                    try:
                        sampler.sample(trainloader, burn_in, n_samples)
                        sampler.save('/home/tim/PycharmProjects/Thesis/Experiments/Results/Results_GAM/{}'.format(name))
                        sampler.traceplots(baseline=True)
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

    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(Z, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']
    sampler = Sampler(gam, epsilon=0.004, L=2)
    sampler.sample(trainloader, burn_in, n_samples)
    sampler.model.check_chain(sampler.chain)
    import random
    import matplotlib

    matplotlib.use('TkAgg')

    sampler.model.plot(X[0:100], y[0:100], sampler.chain[-30:])


