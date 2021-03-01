import unittest
from copy import deepcopy

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .Optimizer import Optimizer
from ..Test_Samplers.Regression_Convergence_Setup import Regression_Convergence_Setup


class Test_Regression_SGD_logprob(Regression_Convergence_Setup, unittest.TestCase):

    def test_OptimizerLogProb(self):
        lr = 0.001

        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=0)

        self.sampler = Optimizer(self.model, trainloader)
        self.sampler.sample(loss_closure=lambda X, y: - self.model.log_prob(X, y), steps=self.steps, lr=lr)

    def test_ADAM_log_prob(self):
        loss_fn = lambda X, y: -self.model.log_prob(X, y)
        # loss_fn = lambda X, y: -(self.model.prior_log_prob() + self.model.likelihood(X).log_prob(y).sum())
        # loss_fn = lambda X,y: - self.model.likelihood(X).log_prob(y).sum()
        self.sampler = torch.optim.Adam(self.model.parameters(), lr=0.05)

        print(self.model.true_model)
        self.sampler.loss = []
        self.sampler.chain = []
        for j in tqdm(range(self.steps)):

            loss = loss_fn(self.X, self.y)
            self.sampler.loss.append(loss)

            self.sampler.zero_grad()
            loss.backward()

            self.sampler.step()
            self.sampler.chain.append(deepcopy(self.model.state_dict()))

            if (j + 1) < 10 or (j + 1) % 500 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
                print("gradients:",
                      torch.cat([p.grad.reshape((p.grad.shape[0], 1)) for p in self.model.parameters()], 0), '\n')

    def test_ADAM_log_prob_on_batches(self):
        """This test uses an Adam optimiser on batches of data and log_prob as criterion"""
        # loss_fn = lambda X, y: -self.model.log_prob(X, y)
        loss_fn = lambda X, y: -(self.model.prior_log_prob() + self.model.likelihood(X).log_prob(y).sum())
        # loss_fn = lambda X, y: - self.model.likelihood(X).log_prob(y).sum()
        self.sampler = torch.optim.Adam(self.model.parameters(), lr=0.05)

        print(self.model.true_model)

        train_set = torch.utils.data.TensorDataset(self.X, self.y)
        loader = DataLoader(train_set, batch_size=50, shuffle=True)

        self.sampler.loss = []
        self.sampler.chain = []
        for j in tqdm(range(self.steps)):
            # batch
            X, y = next(iter(loader))

            loss = loss_fn(X, y)
            self.sampler.loss.append(loss)

            self.sampler.zero_grad()
            loss.backward()

            self.sampler.step()
            self.sampler.chain.append(deepcopy(self.model.state_dict()))

            if (j + 1) < 10 or (j + 1) % 500 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
                print("gradients:",
                      torch.cat([p.grad.reshape((p.grad.shape[0], 1)) for p in self.model.parameters()], 0), '\n')

    def test_ADAM_likelihood_on_batches(self):
        """This test uses an Adam optimiser on batches of data and likelihood as criterion """
        # loss_fn = lambda X, y: -self.model.log_prob(X, y)
        # loss_fn = lambda X, y: -(self.model.prior_log_prob() + self.model.likelihood(X).log_prob(y).sum())
        loss_fn = lambda X, y: - self.model.likelihood(X).log_prob(y).sum()
        self.sampler = torch.optim.Adam(self.model.parameters(), lr=0.05)

        print(self.model.true_model)

        train_set = torch.utils.data.TensorDataset(self.X, self.y)
        loader = DataLoader(train_set, batch_size=50, shuffle=True)
        # X, y = next(iter(loader))
        # self.sampler.true_loss = loss_fn(self.model.forward(X), )

        self.sampler.loss = []
        self.sampler.chain = []
        for j in tqdm(range(self.steps)):
            # batch
            X, y = next(iter(loader))

            loss = loss_fn(X, y)
            self.sampler.loss.append(loss)

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
