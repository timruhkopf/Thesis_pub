import unittest

from torch.utils.data import TensorDataset, DataLoader

from src.Tests.Optimizer import Optimizer
from copy import deepcopy
from tqdm import tqdm
import torch

from .Test_Samplers import Test_Samplers


class Test_ModelSGD_logprob(unittest.TestCase):
    def setUp(self) -> None:
        Test_Samplers.setUp(self)

    def tearDown(self) -> None:
        Test_Samplers.tearDown(self)

    def test_OptimizerLogProb(self):
        # TODO test with MSE & with log_prob
        lr = 0.001

        batch_size = self.X.shape[0]
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.sampler = Optimizer(self.model, trainloader)
        self.sampler.sample(loss_closure=lambda X, y: - self.model.log_prob(X, y), steps=1000, lr=lr)

    def test_ADAM_log_prob(self):
        loss_fn = lambda X, y: -self.model.log_prob(X, y)
        self.sampler = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        num_iterations = 1000

        self.sampler.chain = []
        for j in tqdm(range(num_iterations)):
            # run the model forward on the data
            y_pred = self.model.forward(self.X)  # .squeeze(-1)
            # calculate the mse loss
            loss = loss_fn(y_pred, self.y)
            # initialize gradients to zero
            self.sampler.zero_grad()
            # backpropagate
            loss.backward()
            # take a gradient step
            self.sampler.step()

            self.sampler.chain.append(deepcopy(self.model.state_dict()))

            if (j + 1) < 10 or (j + 1) % 500 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
                print("gradients:",
                      torch.cat([p.grad.reshape((p.grad.shape[0], 1)) for p in self.model.parameters()], 0), '\n')


if __name__ == '__main__':
    unittest.main()
