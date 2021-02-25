import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.Samplers import *
from .Regression_Convergence_Setup import Regression_Convergence_Setup
from .util import posterior_mean, chain_mat, plot_sampler_path


class Test_Sampler(Regression_Convergence_Setup, unittest.TestCase):
    # def test_RHMC(self):
    #     # sampler config
    #     eps, L = 0.001, 1
    #     sampler_param = dict(epsilon=eps, L=L)
    #
    #     burn_in, n_samples = 100, 1000
    #     batch_size = self.X.shape[0]
    #
    #     # dataset setup
    #     trainset = TensorDataset(self.X, self.y)
    #     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    #     # check that the model's log_prob is not broken
    #     # TODO RHMC.step(partial(self.model.log_prob, *data))  # maybe negative?
    #     # self.model.log_prob = lambda X, y: self.model.prior_log_prob() + self.model.likelihood(X).log_prob(y).mean()
    #
    #     # init sampler & sample
    #     Sampler = myRHMC
    #     self.sampler = Sampler(self.model, **sampler_param)
    #     self.sampler.sample(trainloader, burn_in, n_samples)

    def test_MALA(self):
        burn_in, n_samples = 100, 1000
        batch_size = self.X.shape[0]

        sampler_param = {'epsilon': 0.001, 'num_steps': n_samples,
                         'burn_in': burn_in, 'pretrain': False, 'tune': False, 'num_chains': 1}

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        Sampler = MALA  # step_size
        self.sampler = Sampler(self.model, trainloader, **sampler_param)
        self.sampler.sample()

        self.assertTrue(torch.allclose(chain_mat([self.model.true_model])[0],
                                       posterior_mean(self.sampler.chain[-200:]), atol=0.03),
                        msg='True parameters != posterior mean(on last 200 steps of chain)')

        self.model.init_model
        self.model.true_model
        self.sampler.chain[-1]

    def test_SGRHMC(self):
        # instantiate Sampler & sampler_config
        # if sampler_name in ['SGNHT', 'RHMC', 'SGRHMC']:
        #     sampler_param.update(dict(L=L))
        #
        # if sampler_name == 'SGRHMC':
        #     sampler_param.update(dict(alpha=0.2))
        #
        # if 'SG' in sampler_name:
        #     batch_size = sg_batch
        # else:
        #     batch_size = X.shape[0]
        #
        # # sample

        # check the resulting estimates are within a certain range

        pass

    # def test_SGNHT(self):
    #     pass
    #


if __name__ == '__main__':
    unittest.main(exit=False)
