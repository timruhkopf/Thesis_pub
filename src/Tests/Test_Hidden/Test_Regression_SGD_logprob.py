import unittest

import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from tqdm import tqdm

from .Optimizer import Optimizer
from ..Test_Samplers.Regression_Convergence_Setup import Regression_Convergence_Setup


class Test_Regression_SGD_logprob(Regression_Convergence_Setup, unittest.TestCase):

    def test_OptimizerLogProb(self):
        lr = 0.001

        batch_size = self.X.shape[0]
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.sampler = Optimizer(self.model, trainloader)
        self.sampler.sample(loss_closure=lambda X, y: - self.model.log_prob(X, y), steps=1000, lr=lr)

    def test_ADAM_log_prob(self):
        loss_fn = lambda X, y: -self.model.log_prob(X, y)
        self.sampler = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        self.sampler.chain = []
        for j in tqdm(range(1000)):
            y_pred = self.model.forward(self.X)
            loss = loss_fn(y_pred, self.y)

            self.sampler.zero_grad()
            loss.backward()

            self.sampler.step()
            self.sampler.chain.append(deepcopy(self.model.state_dict()))

            if (j + 1) < 10 or (j + 1) % 500 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
                print("gradients:",
                      torch.cat([p.grad.reshape((p.grad.shape[0], 1)) for p in self.model.parameters()], 0), '\n')


if __name__ == '__main__':
    unittest.main(exit=False)
