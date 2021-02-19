import unittest

import torch
import torch.distributions as td
import numpy as np

from ..Util.Util_Model import Util_Model
from ..Util.Util_Distribution import LogTransform
from ..Util.Util_bspline import get_design, eval_basis, diff_mat1D
from ..Models.BNN import BNN
from ..Layer.Hidden import Hidden


class TestUtil(unittest.TestCase):
    def test_check_chain_seq(self):
        # check if did not progress at all : len == 1
        with self.assertRaises(RuntimeError):
            bnn = BNN()
            chain = [bnn.state_dict()]
            Util_Model.check_chain_seq(chain)

        # check if with nan  / inf in chain
        with self.assertRaises(RuntimeError):
            bnn = BNN()
            bnn.layers[1].b.data[0] = torch.tensor(float('nan'))
            chain = [bnn.true_model, bnn.state_dict()]
            Util_Model.check_chain_seq(chain)

        # check if first and last are same (rejected are also in the list)
        with self.assertRaises(RuntimeError):
            # FIXME: does not raise runtime error
            bnn = BNN()
            chain = [bnn.true_model, bnn.true_model]
            Util_Model.check_chain_seq(chain)

    def test_check_chain(self):
        # check if did not progress at all : len == 1
        with self.assertRaises(RuntimeError):
            h = Hidden(5, 2, bias=True)
            chain = [h.true_model['W']]
            Util_Model.check_chain(chain)

        # check if with nan  todo / inf in chain
        with self.assertRaises(RuntimeError):
            h = Hidden(5, 2, bias=True)
            h.W.data[0] = torch.tensor([float('nan'), 1.])
            chain = [h.true_model['W'], h.state_dict()]
            # any([True for t in h.state_dict().values()
            #      if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t))])
            Util_Model.check_chain(chain)

        # check if first and last are same (rejected are also in the list)
        with self.assertRaises(RuntimeError):
            h = Hidden(5, 2, bias=True)
            chain = [h.true_model['W'], h.true_model['W']]
            Util_Model.check_chain(chain)

    def test_bijection(self):
        dist = td.Gamma(0.1, 0.1)
        dist_bij = td.TransformedDistribution(dist, LogTransform())

        a = dist_bij.sample([1000])
        b = dist_bij.transforms[0]._inverse(a)

        dist.log_prob(b)

        # b can never be negative! (and a is very unlikely to have no negative)
        self.assertTrue(torch.any(a < 0.))
        self.assertFalse(torch.any(b < 0.))

        # the inverse is same as the original
        c = dist_bij.transforms[0](b)
        self.assertTrue(torch.allclose(a, c))

        # Not optimal testing
        # consider sample from dist_bij, then invert to gamma's space
        #  calculating ML estimators for rate and concentration on large sample.
        #  they should be very close!

    def test_get_design(self):
        """sum of Z's row must always be 1."""
        n = 10
        no_basis = 10
        X = td.Uniform(-10., 10.).sample([n])
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis),
                         dtype=torch.float32, requires_grad=False)

        # Fixme: fails: observations, that fall close to the edge are not guaranteed to
        #  sum to 1, but approximately 1 (round about - 0.04)
        self.assertTrue(torch.allclose(Z.sum(1), torch.ones(no_basis)))

    def test_util_bspline_LS(self):
        """generate linear data X = [1, x1] & do an GAM OLS fit using Z, the bspline expansion of x1"""
        # (1.1) Least squares example: ---------------------------------------------
        # Effects generation
        n = 100
        X = np.stack([np.ones(n), np.random.uniform(0, 10, n)], axis=1)
        beta = np.array([4, -2])
        mu = X.dot(beta)
        y = mu + np.random.normal(loc=0, scale=1, size=n)

        # basis expansion & OLS fit
        Z = get_design(X[:, 1], no_basis=10, degree=2)
        OLS = lambda Z, y: np.linalg.inv(Z.T.dot(Z)).dot(Z.T).dot(y)
        beta_hat = OLS(Z, y)
        y_hat = Z.dot(beta_hat)

        # Metrics
        resid = y - y_hat
        bias = (mu - y_hat) ** 2

        self.assertAlmostEqual(sum(resid), 0, places=2)

    def test_eval_basis(self):
        """evaluate an observation on bspline grid; single obs. becomes a vector that must sum to 1"""
        # carefull to get the boundary regions right! Through in extra kappas, such that the
        lower, upper = 0, 10  # support of x
        degree = 2
        l_knots = lower - degree - 1
        u_knots = upper + degree + 2

        z = eval_basis(x=upper, knots=np.arange(l_knots, u_knots, 1), degree=degree)
        self.assertAlmostEqual(z.sum(), 1.)

        z = eval_basis(x=lower, knots=np.arange(l_knots, u_knots, 1), degree=degree)
        self.assertAlmostEqual(z.sum(), 1.)

    def test_diff_mat1D(self):
        d1, k1 = diff_mat1D(dim=5, order=1)
        self.assertTrue(np.all(d1 == np.array([[-1, 1, 0, 0, 0],
                                               [0, -1, 1, 0, 0],
                                               [0, 0, -1, 1, 0],
                                               [0, 0, 0, -1, 1]])))

        self.assertTrue(np.all(k1 == np.array([[1, -1, 0, 0, 0],  # K1
                                               [-1, 2, -1, 0, 0],
                                               [0, -1, 2, -1, 0],
                                               [0, 0, -1, 2, -1],
                                               [0, 0, 0, -1, 1]])))

        # Check the diff_mat for higher orders -----------------------------
        # for order in [1, 2, 3, 4, 5]:
        #     d, K = diff_mat1D(dim=5, order=order)
        #     print('order {order}\n, D{order}: \n {d}, \n K{order} \n{K}, \n'.format(order=order, d=d, K=K))


if __name__ == '__main__':
    unittest.main(exit=False)
