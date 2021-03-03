import unittest

import torch
import torch.nn as nn

from src.Layer.GAM import GAM


class Test_(unittest.TestCase):

    def setUp(self) -> None:
        self.model = GAM(no_basis=5, order=1, activation=nn.Identity(), bijected=True)

    def test_proper_cov(self):
        """from imporper K derive proper K"""
        self.assertEqual(torch.matrix_rank(self.model.cov),
                         torch.tensor(self.model.cov.shape[0]))

    def test_resetparameters_fix_tau(self):
        """tests resetting parameters with a fix value
        and implicitly update_distributions."""
        prev_tau = self.model.tau.clone().detach()
        prev_tau_bij = self.model.tau_bij.clone().detach()
        prev_W = self.model.W.clone().detach()
        prev_cov = self.model.cov.clone().detach()

        fix_tau = torch.tensor(1.)
        self.model.reset_parameters(tau=fix_tau)

        # (1) actual update happened
        self.assertNotEqual(self.model.tau, prev_tau)
        self.assertNotEqual(self.model.tau_bij, prev_tau_bij)
        self.assertFalse(torch.all(torch.eq(self.model.W, prev_W)))

        # (2) fixed value is correct
        self.assertEqual(self.model.tau, fix_tau)

        # the tau_bij_was updated
        self.assertEqual(self.model.tau_bij,
                         self.model.tau.dist.transforms[0]._inverse(self.model.tau))

        # (3) W'distribution is updated
        self.assertFalse(torch.all(torch.eq(self.model.W.dist.covariance_matrix, prev_cov)))
        self.test_proper_cov()


if __name__ == '__main__':
    unittest.main(exit=False)
