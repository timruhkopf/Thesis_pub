import unittest

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.Samplers import myRSGLD
from src.Layer.Hierarchical_GroupHorseShoe import GroupHorseShoe
from src.Tests.Test_Samplers.Convergence_teardown import Convergence_teardown

from src.Tests.Test_Samplers.util import chain_mat


class Test_GroupHorse_samplable(Convergence_teardown, unittest.TestCase):

    def tearDown(self) -> None:
        del self.model
        del self.X
        del self.y
        del self.sampler

    def test_not_seperated(self):
        self.model = GroupHorseShoe(2, 1, bias=True, no_shrink=1)
        self.model.reset_parameters(separated=False)
        X, y = self.model.sample_model(1000)

        self.model.reset_parameters(separated=False)
        self.X = X
        self.y = y

        self.sample_RSGLD()
        self.model.plot(X[200:], y[200:], self.sampler.chain[-100::5])

    def test_multiple_variables(self):
        """not separated at all"""
        self.model = GroupHorseShoe(3, 1, bias=True, no_shrink=2)
        self.model.reset_parameters(separated=False)
        X, y = self.model.sample_model(1000)

        self.model.reset_parameters(separated=False)
        self.X = X
        self.y = y

        self.sample_RSGLD(eps=0.001, n_samples=2000)

    def test_separated_multivar(self):
        """FLAT SURFACE (since both variables are shrunken towards 0)"""
        self.model = GroupHorseShoe(2, 1, bias=True, no_shrink=2)
        self.model.reset_parameters(separated=True)
        X, y = self.model.sample_model(1000)  # overwrites True_model!!

        self.model.reset_parameters(separated=False)
        self.X = X
        self.y = y

        self.sample_RSGLD()
        # self.sampler.chain = self.model.invert_bij(self.sampler.chain)
        self.model.plot(X[200:], y[200:], self.sampler.chain[-100::5])

    def test_separated_singlevar(self):
        """only one of two variables is shrunken"""
        self.model = GroupHorseShoe(2, 1, bias=True, no_shrink=1)
        self.model.reset_parameters(separated=True)
        X, y = self.model.sample_model(1000)  # overwrites True_model!!

        self.model.reset_parameters(separated=False)
        self.X = X
        self.y = y

        self.sample_RSGLD()
        # self.sampler.chain = self.model.invert_bij(self.sampler.chain)
        self.model.plot(X[200:], y[200:], self.sampler.chain[-100::5])

    def sample_RSGLD(self, eps=0.001, burn_in=100, n_samples=1000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        self.X.to(device)
        self.y.to(device)

        # sampler setup
        sampler_param = dict(epsilon=eps)
        batch_size = 50

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        Sampler = myRSGLD
        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()


if __name__ == '__main__':
    unittest.main(exit=False)
