from geoopt.samplers import RHMC, RSGLD, SGRHMC
from Pytorch.Samplers.Samplers import Sampler
from functools import partial
from tqdm import tqdm


# geoopt is the inofficial implementation of
# https://openreview.net/pdf?id=r1eiqi09K7 in colloberation with the authors.
# original code can be found @ https://github.com/geoopt/geoopt

class Geoopt_interface:
    def sample(self, trainloader, n_burn, n_samples):
        """

        :param trainloader:
        :param n_burn: number of burnin_steps
        :param n_samples: number of collected samples (steps)
        :return:
        """
        # FiXME: make it log_prob (and trainloader) dependent and ensure, that non-SG
        #  actually has batchsize of whole dataset!
        #  see the following code snipped from RSGLD!
        #      def step(self, closure):
        #         logp = closure()

        # self.model.closure_log_prob(X, y)   # for non-SG
        print('Burn-in')
        for _ in tqdm(range(n_burn)):
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

        self.log_probs
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
    def __init__(self, model, epsilon, n_steps):
        RHMC.__init__(self, params=model.parameters(), epsilon=epsilon, n_steps=n_steps)
        self.model = model


class myRSGLD(RSGLD, Geoopt_interface, Sampler):
    def __init__(self, model, epsilon):
        RSGLD.__init__(self, params=model.parameters(), epsilon=epsilon)
        self.model = model


class mySGRHMC(SGRHMC, Geoopt_interface, Sampler):
    def __init__(self, model, epsilon, n_steps, alpha):
        SGRHMC.__init__(self, params=model.parameters(), epsilon=epsilon,
                        n_steps=n_steps, alpha=alpha)
        self.model = model


if __name__ == '__main__':
    # code taken from https://github.com/geoopt/geoopt/blob/bd6c687862e6692a018ea5201191cc982e74efcf/tests/test_rhmc.py
    import numpy as np
    import pytest

    import torch
    import torch.nn as nn
    import torch.distributions as td
    from torch.utils.data import TensorDataset, DataLoader

    from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel

    no_in = 2
    no_out = 1

    # single Hidden Unit Example
    model = Hidden_ProbModel(no_in, no_out, bias=True, activation=nn.Identity())
    model.forward(X=torch.ones(100, no_in))
    model.prior_log_prob()

    original = model.state_dict()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = model.likelihood(X).sample()

    batch_size = 64
    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    params = dict(sampler="RHMC", epsilon=0.02, n_steps=5)
    params = dict(sampler="RSGLD", epsilon=1e-3)
    params = dict(sampler="SGRHMC", epsilon=1e-3, n_steps=1, alpha=0.5  )

    Sampler = {'RHMC': myRHMC, 'RSGLD': myRSGLD, 'SGRHMC': mySGRHMC}[params.pop("sampler")]
    sampler = Sampler(model, **params)

    points = sampler.sample(trainloader, n_burn=1000, n_samples=1000)

    model.plot2d(X, y, true_model=original, path=None, param=points[0:1000:100])
    print()


    # @pytest.mark.parametrize(
    #     "params",
    #     [
    #         dict(sampler="RHMC", epsilon=0.2, n_steps=5, n_burn=1000, n_samples=5000),
    #         dict(sampler="RSGLD", epsilon=1e-3, n_burn=3000, n_samples=10000),
    #         dict(
    #             sampler="SGRHMC",
    #             epsilon=1e-3,
    #             n_steps=1,
    #             alpha=0.5,
    #             n_burn=3000,
    #             n_samples=10000,
    #         ),
    #     ],
    # )
    # def test_sampling(params):
    #     class NormalDist(torch.nn.Module):
    #         def __init__(self, mu, sigma):
    #             super().__init__()
    #             self.d = torch.distributions.Normal(mu, sigma)
    #             self.x = torch.nn.Parameter(torch.randn_like(mu))
    #
    #         def forward(self):
    #             return self.d.log_prob(self.x).sum()
    #
    #     torch.manual_seed(42)
    #     D = 2
    #     n_burn, n_samples = params.pop("n_burn"), params.pop("n_samples")
    #
    #     mu = torch.randn([D])
    #     sigma = torch.randn([D]).abs()
    #
    #     nd = NormalDist(mu, sigma)
    #     Sampler = {'RHMC': myRHMC, 'RSGLD': myRSGLD, 'SGRHMC': mySGRHMC}[params.pop("sampler")]
    #     sampler = Sampler(nd.parameters(), **params)
    #
    #     for _ in range(n_burn):
    #         sampler.step(nd)
    #
    #     points = []
    #     sampler.burnin = False
    #
    #     for _ in range(n_samples):
    #         sampler.step(nd)
    #         points.append(nd.x.detach().numpy().copy())
    #
    #     points = np.asarray(points)
    #     points = points[::20]
    #     assert nd.x.is_contiguous()
    #     np.testing.assert_allclose(mu.numpy(), points.mean(axis=0), atol=1e-1)
    #     np.testing.assert_allclose(sigma.numpy(), points.std(axis=0), atol=1e-1)
    #
    # test_sampling(dict(sampler="RHMC", epsilon=0.2, n_steps=5, n_burn=1000, n_samples=5000))
    # test_sampling(dict(sampler="RSGLD", epsilon=1e-3, n_burn=3000, n_samples=10000))
    # test_sampling(dict(sampler="SGRHMC",                       epsilon=1e-3,                       n_steps=1,                       alpha=0.5,                       n_burn=3000,
    #                    n_samples=10000,
    #                    ))

    def test_sampling2(params):
        class NormalDist(torch.nn.Module):
            def __init__(self, mu, sigma):
                super().__init__()
                self.d = torch.distributions.Normal(mu, sigma)
                self.x = torch.nn.Parameter(torch.randn_like(mu))

            def forward(self):
                return self.d.log_prob(self.x).sum()

        torch.manual_seed(42)
        D = 2
        n_burn, n_samples = params.pop("n_burn"), params.pop("n_samples")

        mu = torch.randn([D])
        sigma = torch.randn([D]).abs()

        nd = NormalDist(mu, sigma)
        Sampler = {'RHMC': myRHMC, 'RSGLD': myRSGLD, 'SGRHMC': mySGRHMC}[params.pop("sampler")]
        sampler = Sampler(nd, **params)

        points = sampler.sample(500, 1000)
        print(points)


    test_sampling2(dict(sampler="RHMC", epsilon=0.2, n_steps=5, n_burn=1000, n_samples=5000))
    test_sampling2(dict(sampler="RSGLD", epsilon=1e-3, n_burn=3000, n_samples=10000))
    test_sampling2(dict(sampler="SGRHMC", epsilon=1e-3, n_steps=1, alpha=0.5, n_burn=3000, n_samples=10000))
