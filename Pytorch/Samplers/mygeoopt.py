from geoopt.samplers import RHMC, RSGLD, SGRHMC
from Pytorch.Samplers.Samplers import Sampler
from functools import partial
from tqdm import tqdm

import torch


# geoopt is the inofficial implementation of
# https://openreview.net/pdf?id=r1eiqi09K7 in colloberation with the authors.
# original code can be found @ https://github.com/geoopt/geoopt
# https://geoopt.readthedocs.io/en/latest/index.html

class Geoopt_interface(Sampler):

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

        # self.model.closure_log_prob(X, y)   # for non-SG
        print('Burn-in')
        for _ in tqdm(range(burn_in)):
            X, y = next(trainloader.__iter__())
            self.step(partial(self.model.log_prob, X, y))

        points = []
        self.burnin = False

        print('Sampling')
        # for i, epoch in tqdm(enumerate(range(epochs))):
        #     print('Starting Epoch {}'.format(i))
        self.chain = list()

        for _ in tqdm(range(n_samples)):
            X, y = next(trainloader.__iter__())
            self.step(partial(self.model.log_prob, X, y))

            state = self.model.state_dict()
            # if not all([all(state[k] == v) for k, v in samples[-1].items()]): # this is imprecise
            self.chain.append(state)


        if len(self.chain) == 1:
            raise ValueError('The chain did not progress beyond first step')

        if all(any(torch.isnan(v)) for chain in
               (self.chain[-1].values(), self.chain[0]) for v in chain):
            raise ValueError('first and last entry contain nan')

        self.log_probs = torch.tensor(self.log_probs[burn_in:])
        self.state
        self.n_rejected
        self.rejection_rate

        return self.chain

    def clean_chain(self):
        # a = [1,1,2,3,4,4,5,6,7,8,9,9,8]
        # b = range(len(a))
        #
        # c = [c for i, (c, l) in enumerate(zip(b, a)) \
        #  if i == 0 or l != a[i - 1]]
        #
        # le = [c for i, c in enumerate(self.chain) \
        #       if i == 0 or all([a == b for a, b in zip(c, self.chain[i-1])])]
        # len(le)
        #
        # filtered_chain = [c for i, (c, l) in enumerate(zip(self.chain, self.log_probs)) \
        #                   if i == 0 or l != self.log_probs[i - 1]]
        # len(filtered_chain)

        return


class myRHMC(RHMC, Geoopt_interface, Sampler):
    def __init__(self, model, epsilon, L):
        RHMC.__init__(self, params=model.parameters(), epsilon=epsilon, n_steps=L)
        self.model = model


class myRSGLD(RSGLD, Geoopt_interface, Sampler):
    def __init__(self, model, epsilon):
        RSGLD.__init__(self, params=model.parameters(), epsilon=epsilon)
        self.model = model


class mySGRHMC(SGRHMC, Geoopt_interface, Sampler):
    def __init__(self, model, epsilon, L, alpha):
        SGRHMC.__init__(self, params=model.parameters(), epsilon=epsilon,
                        n_steps=L, alpha=alpha)
        self.model = model


if __name__ == '__main__':
    # code taken from https://github.com/geoopt/geoopt/blob/bd6c687862e6692a018ea5201191cc982e74efcf/tests/test_rhmc.py


    import torch
    import torch.nn as nn
    import torch.distributions as td
    from torch.utils.data import TensorDataset, DataLoader
    from copy import deepcopy

    from Pytorch.Layer.Hidden import Hidden

    no_in = 2
    no_out = 1

    # single Hidden Unit Example
    model = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    model.forward(X=torch.ones(100, no_in))
    model.prior_log_prob()

    model.true_model = deepcopy(model.state_dict())


    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = model.likelihood(X).sample()

    batch_size = 64
    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    params = dict(sampler="RHMC", epsilon=0.02, L=5)
    # params = dict(sampler="RSGLD", epsilon=1e-3)
    # params = dict(sampler="SGRHMC", epsilon=1e-3, L=1, alpha=0.5  ) # FIXME: seems to diverge quickly

    Sampler = {'RHMC': myRHMC, 'RSGLD': myRSGLD, 'SGRHMC': mySGRHMC}[params.pop("sampler")]
    sampler = Sampler(model, **params)

    points = sampler.sample(trainloader, burn_in=1000, n_samples=1000)

    model.plot(X, y, chain= points[0:1000:100],path=None )

    import os
    sampler.save(path=os.getcwd() + '/sghnhtsavingtest')

    model1 = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    params1 = dict(sampler="RHMC", epsilon=0.02, L=5)
    Sampler1 = {'RHMC': myRHMC, 'RSGLD': myRSGLD, 'SGRHMC': mySGRHMC}[params1.pop("sampler")]

    sampler1 = Sampler(model1, **params1)


    a = sampler1.load(path=os.getcwd() + '/sghnhtsavingtest')



    print()
