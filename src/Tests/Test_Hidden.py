import unittest

import torch
import torch.distributions as td
import torch.nn as nn
from ..Layer import Hidden
from ..Samplers.mygeoopt import myRHMC


class TestHidden(unittest.TestCase):
    def tearDown(self) -> None:
        del self.model

    def test_prior_log_prob(self):
        """ensure, that each w is evaluated under standard normal correctly"""
        dist = td.Normal(torch.tensor(0.), torch.tensor(1.))

        self.model = Hidden(2, 1, bias=True, activation=nn.ReLU())
        prior = dist.log_prob(self.model.W).sum() + dist.log_prob(self.model.b).sum()
        self.assertEqual(self.model.prior_log_prob(), prior, msg='Hidden\'s prior log prob is not correct')

        self.model = Hidden(3, 2, bias=True, activation=nn.ReLU())
        prior = dist.log_prob(self.model.W).sum() + dist.log_prob(self.model.b).sum()
        self.assertEqual(self.model.prior_log_prob(), prior, msg='Hidden\'s prior log prob is not correct')

        self.model = Hidden(3, 2, bias=False, activation=nn.ReLU())
        prior = dist.log_prob(self.model.W).sum()
        self.assertEqual(self.model.prior_log_prob(), prior, msg='Hidden\'s prior log prob is not correct')

    def test_forward(self):
        X = torch.ones((1, 3))
        self.model = Hidden(3, 2, bias=False, activation=nn.ReLU())
        self.model.W.data = torch.tensor([[1., 2.],
                                          [1., 2.],
                                          [1., 2.]])
        self.assertTrue(torch.all(torch.eq(self.model.forward(X), torch.tensor([[3., 6.]]))))

        X = torch.ones((1, 3))
        self.model = Hidden(3, 2, bias=True, activation=nn.ReLU())
        self.model.W.data = torch.tensor([[1., 2.],
                                          [1., 2.],
                                          [1., 2.]])
        self.model.b.data = torch.tensor([1., 1.])
        self.assertTrue(torch.all(torch.eq(self.model.forward(X), torch.tensor([[4., 7.]]))))

    def test_resample_model(self):
        """ensure, that after sampling, the model is a new one, but the tensor shapes are preserved"""
        from copy import deepcopy
        self.model = Hidden(3, 2, bias=True, activation=nn.ReLU())

        W = deepcopy(self.model.W)
        b = deepcopy(self.model.b)

        self.model.reset_parameters()
        self.assertFalse(torch.all(torch.eq(self.model.W, W)))
        self.assertFalse(torch.all(torch.eq(self.model.b, b)))

        self.assertEqual(self.model.W.shape, W.shape)
        self.assertEqual(self.model.b.shape, b.shape)

    def test_samplable(self):
        """double check that a simple regression example works"""
        from .Test_Samplers import Test_Samplers
        self.model = Hidden(3, 1, bias=True, activation=nn.Identity())
        Test_Samplers.test_RHMC(self)


if __name__ == '__main__':
    unittest.main(exit=False)
