import unittest
import torch

from src.Layer.Hierarchical_GroupLasso import GroupLasso


class Test_GroupLasso_module(unittest.TestCase):

    def test_partionedW(self):
        self.model = GroupLasso(5, 3, bias=True, no_shrink=2)
        self.assertEqual(self.model.tau.shape, torch.Size([2]))

        self.assertEqual(self.model.W.shape, (5, 3))
        self.assertEqual(self.model.W.dist_shrinked.batch_shape, (2, 3))
        self.assertEqual(self.model.W.dist.batch_shape, (3, 3))

        # check the appropriate parts of W are sampled accordingly
        self.model.W.data = torch.ones_like(self.model.W.data)
        self.model.tau.data = self.model.tau.bij(torch.tensor([0.0001, 0.0001]))
        self.model.update_distributions()

        # #  code snipped from source: reminder of how W is structured!
        self.model.W.data[
        :self.model.no_shrink] = self.model.W.dist_shrinked.sample()  # .view(self.model.no_shrink, self.model.no_out)
        # # now only values in the first two rows are updated!

    def test_reset_parameters_seperateT(self):
        self.model = GroupLasso(5, 3, bias=True, no_shrink=2)
        self.model.reset_parameters(separated=True)  # all shrinked have variance 0.05
        self.assertTrue(torch.eq(self.model.tau.inv(self.model.tau).detach(), torch.tensor([0.05, 0.05])))

        # TODO continue testing

    def test_reset_parameters_seperateF(self):
        # TODO continue testing
        pass


if __name__ == '__main__':
    unittest.main(exit=False)
