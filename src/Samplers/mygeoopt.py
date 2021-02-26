from functools import partial

import torch
from geoopt.samplers import RHMC, RSGLD, SGRHMC
from tqdm import tqdm

from src.Util.Util_Samplers import Util_Sampler


# geoopt is the inofficial implementation of
# https://openreview.net/pdf?id=r1eiqi09K7 in colloberation with the authors.
# original code can be found @ https://github.com/geoopt/geoopt
# https://geoopt.readthedocs.io/en/latest/index.html

class Geoopt_interface(Util_Sampler):
    """geoopt samplers implementations is based on
    Hamiltonian Monte-Carlo for Orthogonal Matrices"""

    def sample(self, trainloader, burn_in, n_samples):
        """

        :param trainloader:
        :param burn_in: number of burnin_steps
        :param n_samples: number of collected samples (steps)
        :return: list of state_dicts (OrderedDicts) representing each state of the model
        """
        # FiXME: make it log_prob (and trainloader) dependent and ensure, that non-SG
        #  actually has batchsize of whole dataset!
        #  see the following code snipped from RSGLD!
        #      def step(self, closure):
        #         logp = closure()

        if 'SG' not in str(self) and trainloader.batch_size != len(trainloader.dataset):
            raise ValueError('trainloader for non-SG Sampler must use the entire dataset at each step'
                             ' set trainloader.batch_size = len(trainloader.dataset')

        # self.model.closure_log_prob(X, y)   # for non-SG

        print('Burn-in')
        self.chain = list()
        for _ in tqdm(range(burn_in)):
            data = next(trainloader.__iter__())
            self.step(partial(self.model.log_prob, *data))
            state = self.model.state_dict()
            # if not all([all(state[k] == v) for k, v in samples[-1].items()]): # this is imprecise
            self.chain.append(state)

        # self.model.check_chain(self.chain)

        print('\nSampling')
        points = []
        self.burnin = False

        self.chain = list()  # reset the chain
        for _ in tqdm(range(n_samples)):
            data = next(trainloader.__iter__())
            self.step(partial(self.model.log_prob, *data))

            state = self.model.state_dict()
            # if not all([all(state[k] == v) for k, v in samples[-1].items()]): # this is imprecise
            self.chain.append(state)

        # self.model.check_chain(self.chain)
        self.log_probs = torch.tensor(self.log_probs[burn_in:])
        self.state
        self.n_rejected
        self.rejection_rate

        return self.chain





class myRHMC(RHMC, Geoopt_interface, Util_Sampler):
    def __init__(self, model, epsilon, L):
        RHMC.__init__(self, params=model.parameters(), epsilon=epsilon, n_steps=L)
        self.model = model

    def __str__(self):
        return 'myRHMC'


class myRSGLD(RSGLD, Geoopt_interface, Util_Sampler):
    def __init__(self, model, epsilon):
        RSGLD.__init__(self, params=model.parameters(), epsilon=epsilon)
        self.model = model

    def __str__(self):
        return 'myRSGLD'


class mySGRHMC(SGRHMC, Geoopt_interface, Util_Sampler):
    def __init__(self, model, epsilon, L, alpha):
        SGRHMC.__init__(self, params=model.parameters(), epsilon=epsilon,
                        n_steps=L, alpha=alpha)
        self.model = model

    def __str__(self):
        return 'mySGRHMC'


if __name__ == '__main__':
    # code taken from https://github.com/geoopt/geoopt/blob/bd6c687862e6692a018ea5201191cc982e74efcf/tests/test_rhmc.py

    import torch.nn as nn
    import torch.distributions as td
    from torch.utils.data import TensorDataset, DataLoader
    from copy import deepcopy

    from src.Layer.Hidden import Hidden

    no_in = 2
    no_out = 1

    n = 100
    # single Hidden Unit Example
    model = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    model.forward(X=torch.ones(100, no_in))
    model.prior_log_prob()

    model.true_model = deepcopy(model.state_dict())

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n]))
    y = model.likelihood(X).sample()

    params = dict(sampler="RHMC", epsilon=0.02, L=5)
    # params = dict(sampler="RSGLD", epsilon=1e-3)
    # params = dict(sampler="SGRHMC", epsilon=1e-3, L=1, alpha=0.5  ) # FIXME: seems to diverge quickly

    Sampler = {'RHMC': myRHMC, 'RSGLD': myRSGLD, 'SGRHMC': mySGRHMC}[params.pop("sampler")]

    sampler = Sampler(model, **params)

    # if non SG sampler: batch_size = n else n can be smaller than N
    batch_size = n
    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    points = sampler.sample(trainloader, burn_in=1000, n_samples=1000)

    model.plot(X, y, chain=points[0:1000:100], path=None)

    import os

    sampler.save(path=os.getcwd() + '/sghnhtsavingtest')

    model1 = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    params1 = dict(sampler="RHMC", epsilon=0.02, L=5)
    Sampler1 = {'RHMC': myRHMC, 'RSGLD': myRSGLD, 'SGRHMC': mySGRHMC}[params1.pop("sampler")]

    sampler1 = Sampler(model1, **params1)

    a = sampler1.load(path=os.getcwd() + '/sghnhtsavingtest')

    print()
