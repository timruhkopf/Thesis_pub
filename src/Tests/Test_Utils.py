import unittest

import torch
import torch.distributions as td

from ..Util.Util_Model import Util_Model
from ..Util.Util_Distribution import LogTransform
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

        # check if with nan todo / inf in chain
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


if __name__ == '__main__':
    unittest.main(exit=False)
