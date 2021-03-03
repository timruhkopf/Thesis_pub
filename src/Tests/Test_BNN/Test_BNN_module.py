import unittest

import torch
import torch.nn as nn

from src.Models.BNN import BNN


class Test_BNN_module(unittest.TestCase):

    def setUp(self) -> None:
        self.model = BNN(hunits=(1, 2, 5, 1), activation=nn.ReLU())

    def test_parsing_architecture(self):
        self.assertEqual(len(self.model.layers), 3)
        self.assertEqual((1, 2), (self.model.layers[0].no_in, self.model.layers[0].no_out))
        self.assertEqual((2, 5), (self.model.layers[1].no_in, self.model.layers[1].no_out))
        self.assertEqual((5, 1), (self.model.layers[2].no_in, self.model.layers[2].no_out))

    def test_reset_parameters(self):
        prev_param = {name: p.clone().detach() for name, p in self.model.named_parameters()}
        self.model.reset_parameters()
        new_param = {name: p.clone().detach() for name, p in self.model.named_parameters()}

        for name in prev_param.keys():
            prev, new = prev_param[name], new_param[name]
            self.assertEqual(prev.shape, new.shape)
            self.assertFalse(torch.all(torch.eq(prev, new)))


if __name__ == '__main__':
    unittest.main(exit=False)
