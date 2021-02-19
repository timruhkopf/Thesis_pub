import unittest

from torch.utils.data import TensorDataset, DataLoader

from src.Tests.Optimizer import Optimizer
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn

from .Test_Samplers import Test_Samplers


class Test_Models_SGD_MSE(unittest.TestCase):
    def setUp(self) -> None:
        Test_Samplers.setUp(self)

    def tearDown(self) -> None:
        Test_Samplers.tearDown(self)

    def test_usingADAM(self):
        loss_fn = torch.nn.MSELoss(reduction='mean')
        self.sampler = torch.optim.Adam(self.model.parameters(), lr=0.05)

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

        # print(chain_mat([self.model.state_dict(),
        #                  self.model.true_model,
        #                  self.model.init_model]))
        # print(self.LS, '\n', self.model_in_LS_format())


    def test_OptimSGD(self):

        from torch import optim
        from tqdm import tqdm
        from copy import deepcopy
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.MSELoss()

        batch_size = 100
        trainset = TensorDataset(self.X, self.y)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer.chain = list()
        for i in tqdm(range(1000)):
            X, y = next(self.trainloader.__iter__())
            optimizer.zero_grad()
            loss = loss_fn(self.model.forward(X), y)
            loss.backward()
            optimizer.step()
            optimizer.chain.append(deepcopy(self.model.state_dict()))

        self.sampler = optimizer

    def test_OptimizerMSE(self):
        # TODO test with MSE & with log_prob

        lr = 0.02
        n_samples = 1000
        batch_size = self.X.shape[0]
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # x, y = next(trainloader.__iter__())
        # x1 = x.detach().clone()
        self.sampler = Optimizer(self.model, trainloader)
        # self.sampler.sample(loss_closure=lambda X, y: ((self.model.forward(X) - y) ** 2).mean(),
        #                     steps=n_samples, lr=lr)

        loss = nn.MSELoss()
        self.sampler.sample(loss_closure=lambda X, y: loss(self.model.forward(X), y),
                            steps=n_samples, lr=lr)


if __name__ == '__main__':
    unittest.main()
