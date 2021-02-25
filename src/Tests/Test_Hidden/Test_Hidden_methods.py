import unittest

import torch
import torch.distributions as td
import torch.nn as nn
from src.Layer.Hidden import Hidden


class TestHidden(unittest.TestCase):

    def setUp(self) -> None:
        self.p = 3
        self.no_out = 2
        self.dist = td.Normal(torch.tensor(0.), torch.tensor(1.))
        self.model = Hidden(self.p, self.no_out, bias=True, activation=nn.ReLU())

    def tearDown(self) -> None:
        del self.model

    def test_prior_log_prob(self):
        """ensure, that each w is evaluated under standard normal correctly"""
        self.model = Hidden(2, 1, bias=True, activation=nn.ReLU())
        prior = self.dist.log_prob(self.model.W).sum() + self.dist.log_prob(self.model.b).sum()
        self.assertEqual(self.model.prior_log_prob(), prior, msg='Hidden\'s prior log prob is not correct')

        self.model = Hidden(3, 2, bias=False, activation=nn.ReLU())
        prior = self.dist.log_prob(self.model.W).sum()
        self.assertEqual(self.model.prior_log_prob(), prior, msg='Hidden\'s prior log prob is not correct')

    def test_prior_log_prob2(self):
        prior = self.dist.log_prob(self.model.W).sum() + self.dist.log_prob(self.model.b).sum()
        self.assertEqual(self.model.prior_log_prob(), prior, msg='Hidden\'s prior log prob is not correct')

    def test_forward1(self):
        X = torch.ones((1, 3))
        self.model = Hidden(3, 2, bias=False, activation=nn.ReLU())
        self.model.W.data = torch.tensor([[1., 2.],
                                          [1., 2.],
                                          [1., 2.]])
        self.assertTrue(torch.all(torch.eq(self.model.forward(X), torch.tensor([[3., 6.]]))))

    def test_forward2(self):
        X = torch.ones((1, self.p))

        self.model.W.data = torch.tensor([[1., 2.]] * self.p)
        self.model.b.data = torch.ones(self.no_out)
        self.assertTrue(torch.all(torch.eq(self.model.forward(X),
                                           torch.tensor([[1., 2.]] * self.p).sum(0) + self.model.b)))

    def test_resample_model(self):
        """ensure, that after sampling, the model is a new one, but the tensor shapes are preserved"""
        from copy import deepcopy
        W = deepcopy(self.model.W)
        b = deepcopy(self.model.b)

        self.model.reset_parameters()
        self.assertFalse(torch.all(torch.eq(self.model.W, W)))
        self.assertFalse(torch.all(torch.eq(self.model.b, b)))

        self.assertEqual(self.model.W.shape, W.shape)
        self.assertEqual(self.model.b.shape, b.shape)

    # def test_samplable(self):
    #     # TODO: once samplers are checked, move a test from TestSampler here - so it is checked everytime Hidden is
    #     #  run
    #     """double check that a simple regression example works"""
    #     from .Test_Samplers import Test_Samplers
    #     self.model = Hidden(3, 1, bias=True, activation=nn.Identity())
    #     X, y = self.model.sample_model()
    #     Test_Samplers.test_RHMC(self)


if __name__ == '__main__':
    unittest.main(exit=False)
