import unittest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.Models.BNN import BNN
from src.Samplers.mygeoopt import myRSGLD, mySGRHMC
from src.Tests.Test_Samplers.Convergence_teardown import Convergence_teardown
from src.Tests.Test_Samplers.util import odict_pmean


class Test_BNN_samplable(Convergence_teardown, unittest.TestCase):

    def setUp(self) -> None:
        """This seeding in model__init__ and reset creates an interesting learning
        problem, -- which is successfully learned from to some degree (but not found the
        perfect approximation yet."""

        torch.manual_seed(10)
        self.model = BNN(hunits=(1, 3, 2, 1), activation=nn.ReLU(), final_activation=nn.Identity())

        # Data setup
        n = 1000
        self.X, self.y = self.model.sample_model(n=n)
        # self.model.plot(self.X, self.y)

        torch.manual_seed(6)
        self.model.reset_parameters()
        # self.model.plot(self.X, self.y)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        self.X.to(device)
        self.y.to(device)

    def tearDown(self) -> None:
        try:
            Convergence_teardown.tearDown(self)
            l_chain = len(self.sampler.chain)

            self.model.plot(self.X, self.y, self.sampler.chain[:: l_chain // 30],
                            title='subsampled chain (every 30th state, sorted)')

            pmean = odict_pmean(self.sampler.chain[int(l_chain * 3 / 4)::5])
            self.model.plot(self.X, self.y, chain=[pmean],
                            title='posterior mean on last quarter of the chain; interleave: every 5th')

        except Exception as e:
            # Debugging: allows to plot failed model
            l_chain = len(self.sampler.chain)

            self.model.plot(self.X, self.y, self.sampler.chain[:: l_chain // 30],
                            title='subsampled chain (every 30th state, sorted)')

            pmean = odict_pmean(self.sampler.chain[int(l_chain * 3 / 4)::5])
            self.model.plot(self.X, self.y, chain=[pmean],
                            title='posterior mean on last quarter of the chain; interleave: every 5th')

            raise e

    def test_samplable_RSGLD(self):
        # Sampler set up
        eps = 0.0001
        sampler_param = dict(epsilon=eps)

        burn_in, n_samples = 100, 3000
        batch_size = 100

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        Sampler = myRSGLD
        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()
        n_samples = len(self.sampler.chain)

    def test_samplable_SGRHMC(self):
        eps, L, alpha = 0.0001, 2, 0.1
        sampler_param = dict(epsilon=eps, L=L, alpha=alpha)

        burn_in, n_samples = 100, 2000
        batch_size = 50

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        Sampler = mySGRHMC

        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()


if __name__ == '__main__':
    unittest.main(exit=False)