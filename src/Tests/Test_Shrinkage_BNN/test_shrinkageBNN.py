import unittest
from copy import deepcopy

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.Models.ShrinkageBNN import ShrinkageBNN
from src.Tests.Test_Samplers.Convergence_teardown import Convergence_teardown
from src.Samplers.mygeoopt import myRSGLD


class Test_ShrinkageBNN_samplable(Convergence_teardown, unittest.TestCase):

    # (true model is shrunken, deeper models) ------------------------------
    def test_glasso_single_shrink_separated(self):
        """true model has a shrunk variable!"""
        self.model = ShrinkageBNN(hunits=(2, 2, 2, 1), shrinkage='glasso', no_shrink=1)
        self.model.reset_parameters(separated=True)
        self.model.true_model = deepcopy(self.model.state_dict())

    def test_ghorse_single_shrink_separated(self):
        """true model has a shrunk variable!"""
        self.model = ShrinkageBNN(hunits=(2, 2, 2, 1), shrinkage='ghorse', no_shrink=1)
        self.model.reset_parameters(separated=True)
        self.model.true_model = deepcopy(self.model.state_dict())

    # (true model is shrunken) ------------------------------
    def test_glasso_single_shrink_separated(self):
        """true model has a shrunk variable!"""
        self.model = ShrinkageBNN(hunits=(2, 2, 1), shrinkage='glasso', no_shrink=1)
        self.model.reset_parameters(separated=True)
        self.model.true_model = deepcopy(self.model.state_dict())

    def test_ghorse_single_shrink_separated(self):
        """true model has a shrunk variable!"""
        self.model = ShrinkageBNN(hunits=(2, 2, 1), shrinkage='ghorse', no_shrink=1)
        self.model.reset_parameters(separated=True)
        self.model.true_model = deepcopy(self.model.state_dict())


    # (different share of shrinked vars)--------------------------------------------
    def test_glasso_single_shrink(self):
        self.model = ShrinkageBNN(hunits=(2, 2, 1), shrinkage='glasso', no_shrink=1)

    def test_glasso_all_shrink(self):
        self.model = ShrinkageBNN(hunits=(2, 2, 1), shrinkage='glasso', no_shrink=2)

    def test_glasso_mix_shrink(self):
        self.model = ShrinkageBNN(hunits=(3, 2, 1), shrinkage='glasso', no_shrink=2)

    def test_ghorse_single_shrink(self):
        self.model = ShrinkageBNN(hunits=(2, 2, 1), shrinkage='ghorse', no_shrink=1)

    def test_ghorse_all_shrink(self):
        self.model = ShrinkageBNN(hunits=(2, 2, 1), shrinkage='ghorse', no_shrink=2)

    def test_ghorse_mix_shrink(self):
        self.model = ShrinkageBNN(hunits=(3, 2, 1), shrinkage='ghorse', no_shrink=2)

    def tearDown(self) -> None:

        self.X, self.y = self.model.sample_model(1000)
        self.to_device(self.X, self.y)

        eps = 0.0001
        sampler_param = dict(epsilon=eps)

        burn_in, n_samples = 100, 3000
        batch_size = 10

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        self.model.reset_parameters(separated=False)
        Sampler = myRSGLD
        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()

        if self.X.shape[1] == 2:
            # look at the entire chain subsampled
            self.model.plot(self.X, self.y, self.sampler.chain[::n_samples // 30],
                            title='subsampled chain (every 30th state, sorted)')

            # look at the latest 300 steps, but subsampled
            self.model.plot(self.X, self.y, self.sampler.chain[-300::n_samples // 30],
                            title='subsampled chain (every 30th state, sorted)')

    def to_device(self, X, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        X.to(device)
        y.to(device)


if __name__ == '__main__':
    unittest.main(exit=False)
