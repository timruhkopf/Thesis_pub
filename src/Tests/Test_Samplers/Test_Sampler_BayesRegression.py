import unittest

from torch.utils.data import TensorDataset, DataLoader

from src.Samplers import *
from src.Tests.Test_Samplers.Regression_Convergence_Setup import Regression_Convergence_Setup


class Test_Sampler(Regression_Convergence_Setup, unittest.TestCase):
    # (GEOOPT) -----------------------------------------------------------------
    def test_RSGLD(self):
        eps = 0.001
        sampler_param = dict(epsilon=eps)

        burn_in, n_samples = 100, 2000
        batch_size = 100

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        Sampler = myRSGLD
        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()

    def test_RHMC(self):
        # sampler config
        eps, L = 0.01, 2
        sampler_param = dict(epsilon=eps, L=L)

        burn_in, n_samples = 100, 1000
        batch_size = self.X.shape[0]

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        Sampler = myRHMC
        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()

    def test_SGRHMC(self):
        eps, L = 0.001, 2
        sampler_param = dict(epsilon=eps, L=L)
        sampler_param.update(dict(alpha=0.01))

        burn_in, n_samples = 100, 1000
        batch_size = 10

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        Sampler = mySGRHMC

        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()

    # # (LUDWIGWINKLER) ----------------------------------------------------------
    # # FIXME: chain never progresses beyond second step
    # def test_SGNHT(self):
    #     #     burn_in, n_samples = 100, 100
    #     #     batch_size = self.X.shape[0]
    #     #     eps, L = 0.00001, 1
    #     #
    #     #     sampler_param = {'epsilon': eps, 'L':L, 'num_steps': n_samples,
    #     #                      'burn_in': burn_in, 'pretrain': False, 'tune': False, 'num_chains': 1}
    #     #
    #     #     # dataset setup
    #     #     trainset = TensorDataset(self.X, self.y)
    #     #     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    #     #
    #     #     Sampler = SGNHT  # step_size
    #     #     self.sampler = Sampler(self.model, trainloader, **sampler_param)
    #     #     self.sampler.sample()
    #     #     # FIXME: log_prob format is shit
    #     #     import torch
    #     #     self.sampler.loss = torch.stack([state['log_prob'] for state in self.sampler.log_probs])
    #     #
    #     #     from src.Tests.Test_Samplers.util import plot_sampler_path
    #     #     if len(self.sampler.chain) > 2:
    #     #         plot_sampler_path(self.sampler, self.model, steps=self.steps, skip=50,
    #     #                           loss=self.sampler.loss)
    #     pass
    #
    # # FIXME: MALA does not work at all - log_prob & chain are single state
    # def test_MALA(self):
    #     #     burn_in, n_samples = 100, 100
    #     #     batch_size = self.X.shape[0]
    #     #
    #     #     sampler_param = {'epsilon': 0.001, 'num_steps': n_samples,
    #     #                      'burn_in': burn_in, 'pretrain': False, 'tune': False, 'num_chains': 1}
    #     #
    #     #     # dataset setup
    #     #     trainset = TensorDataset(self.X, self.y)
    #     #     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    #     #
    #     #     Sampler = MALA  # step_size
    #     #     self.sampler = Sampler(self.model, trainloader, **sampler_param)
    #     #     self.sampler.sample()
    #     #     # FIXME: sampler's log_prob record format is shit
    #     #     self.sampler.loss = self.sampler.log_probs.detach().numpy()
    #     #
    #     #     # self.assertTrue(torch.allclose(chain_mat([self.model.true_model])[0],
    #     #     #                                posterior_mean(self.sampler.chain[-200:]), atol=0.03),
    #     #     #                 msg='True parameters != posterior mean(on last 200 steps of chain)')
    #     #     #
    #     #     # self.model.init_model
    #     #     # self.model.true_model
    #     #     # self.sampler.chain[-1]
    #     pass


if __name__ == '__main__':
    unittest.main(exit=False)
