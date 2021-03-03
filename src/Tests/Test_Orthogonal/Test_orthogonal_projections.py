import unittest

import torch
import torch.distributions as td
import torch.nn as nn

from src.Layer.GAM.GAM import GAM
from src.Models.Orthogonal_GAM_REG import OrthogonalBNN
from src.Layer.Hidden import Hidden
from src.Util.Util_bspline import get_design


class Test_orthoghonal_projections(unittest.TestCase):

    def test_orth_projection(self):
        """check that Pz_orth of a 1 vector creates a vector of means.
        this ensures, the projection is actually correct!"""
        # projection on the orth complement
        X = torch.ones(10, 1)
        y = torch.distributions.Uniform(-10., 10.).sample([10, 1])

        # Notice torch.allclose(torch.ones(1, 10) @ y, y.sum())
        # 1(1'1)**(-1)1' = 1/n * 11', which multiplied with a vec gives a vector of means
        orth_projection = torch.eye(10) - OrthogonalBNN.orth_projection(X)
        self.assertTrue(torch.allclose((orth_projection @ y)[0], y.mean()))

    def test_ortho_projection_singleXGAM(self):
        """this is not exacly a unittest, but a test what happens, if
         X is projected on Z's orthogonal complenent, given Z is a function of X"""
        n = 100
        X = td.Uniform(-10., 10.).sample([n, 1])

        Z = torch.tensor(get_design(X[:, 0].numpy(), degree=2, no_basis=5),
                         dtype=torch.float32, requires_grad=False)

        orth_projection = torch.eye(n) - OrthogonalBNN.orth_projection(Z)
        self.assertFalse(torch.allclose(orth_projection @ X, torch.zeros_like(X)))

    def test_ortho_projection_models0(self):
        """test if GAM + Reg model can be split correctly"""

        # y = ß1 x1 + ß2 x2 + ß3 x3, f_gam(x4) + e
        reg = Hidden(3, 1, bias=False, activation=nn.Identity())
        gam = GAM(no_basis=5, no_out=1, activation=nn.Identity())

        n = 10
        X = td.Uniform(-10., 10.).sample([n, 4])
        y = reg.likelihood(X[:, :3]).sample()
        e = y - reg.forward(X[:, :3])

        Z = torch.tensor(get_design(X[:, 3].numpy(), degree=2, no_basis=gam.no_basis),
                         dtype=torch.float32, requires_grad=False)
        full_y = y + gam.forward(Z)

        # xb cleaned for gam part
        # (I- Z(Z'Z)**-1 Z') @ full_y = (I- Z(Z'Z)**-1 Z') (xb + Z\gamma + e)
        # = (I- Z(Z'Z)**-1 Z') @ xb + (I- Z(Z'Z)**-1 Z') @ e
        self.assertTrue(torch.allclose(OrthogonalBNN.orth_projection(Z) @ full_y,
                                       OrthogonalBNN.orth_projection(Z) @ X[:, :3] @ reg.W + \
                                       OrthogonalBNN.orth_projection(Z) @ e
                                       ))

    def test_ortho_projection_models1(self):
        """basically the test case test_ortho_projection_models0,
        but the reverse orthogonalisation: GAM without linear effect"""
        # test the reverse: clean gam from linear part
        # y = ß1 x1 + ß2 x2 + ß3 x3, f_gam(x4) + e
        reg = Hidden(3, 1, bias=False, activation=nn.Identity())
        gam = GAM(no_basis=5, no_out=1, activation=nn.Identity())

        n = 10
        X = td.Uniform(-10., 10.).sample([n, 4])
        Z = torch.tensor(get_design(X[:, 3].numpy(), degree=2, no_basis=gam.no_basis),
                         dtype=torch.float32, requires_grad=False)

        y = gam.likelihood(Z).sample()
        e = y - gam.forward(Z)
        full_y = y + reg.forward(X[:, :3])

        # gam cleaned of xb part
        # (I- X(X'X)**-1 X') @ full_y = (I- X(X'X)**-1 X') (Xb + Z\gamma + e)
        # = (I- X(X'X)**-1 X') @ Z\gamma + (I- X(X'X)**-1 X') @ e
        self.assertTrue(torch.allclose(OrthogonalBNN.orth_projection(X) @ full_y,
                                       gam.forward(OrthogonalBNN.orth_projection(X) @ Z) + \
                                       OrthogonalBNN.orth_projection(X) @ e))


if __name__ == '__main__':
    unittest.main(exit=False)
