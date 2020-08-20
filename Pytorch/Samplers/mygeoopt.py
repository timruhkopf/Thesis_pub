from geoopt.samplers import RHMC, RSGLD, SGRHMC
from Pytorch.Samplers.Samplers import Sampler


# geoopt is the inofficial implementation of
# https://openreview.net/pdf?id=r1eiqi09K7 in colloberation with the authors.
# original code can be found @ https://github.com/geoopt/geoopt

class Geoopt_interface:
    def sample(self, n_burn, n_samples):
        for _ in range(n_burn):
            self.step(self.model)

        points = []
        self.burnin = False

        for _ in range(n_samples):
            self.step(self.model)
            points.append(self.model.x.detach().numpy().copy())

        # points = np.asarray(points)

        return points


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
    import torch
    import numpy as np
    import pytest


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
