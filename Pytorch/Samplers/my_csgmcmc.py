import torch
import torch.distributions as td
from tqdm import tqdm

from Pytorch.Samplers.Samplers import Sampler


class CSG_MCMC(Sampler):

    # TODO extend the CSG_MCMC to paralellel chains.

    def __init__(self, model, trainloader, alpha0, M, K, beta):
        """

        :param model: nn.Module model, containing forward and log_prob method
        :param trainloader: X, y Dataset trainloader
        :param size_dataset:
        :param alpha0: initial learningrate
        :param M: no. of cycles
        :param K: total no. of trainingsteps
        :param beta: proportion of exploration stage (relative to exploitation)
        """
        # model init (already initialized in model)
        self.KdivM = torch.tensor(K / M)
        self.alpha0 = alpha0
        self.M = M
        self.K = K
        self.beta = torch.tensor(beta)

        self.size_dataset = len(trainloader.dataset)
        self.trainloader = trainloader

        import math
        self.pi = torch.Tensor([math.pi])

        self.model = model

        # adjust the likelihood for stochastic noise
        self.model.log_prob = lambda X, y: \
            sum((len(X) / self.size_dataset) * self.model.likelihood(X).log_prob(y) + \
                self.model.prior_log_prob())

        self.chain = list()

    def r(self, k):
        """'share of cycle', i.e. the percentage of completion of the current cycle"""
        return torch.fmod(torch.tensor(k - 1, dtype=torch.float32), self.KdivM) / self.KdivM

    def alpha(self, k):
        """current learning rate """
        return self.alpha0 / 2 * (torch.cos(self.pi * self.r(k)) + 1)

    def explore_step(self, k):
        """deterministic updates in the exploration stage as T = 0,
        effectively reducing the posterior to a pointmass, that can be optimized to"""
        for p in self.model.parameters():
            print(p.data)
            p.data.add_(-1 * self.alpha(k) * p.grad.data)

    def exploit_step(self, k):
        """the specific sampling SG sampling scheme actually sampling values
        implementation """
        raise NotImplementedError('define the specific Sampling scheme')

    def sample(self, **kwargs):
        """kwargs: see samplers' exploit's specific documentation"""
        for k in tqdm(range(self.K)):
            X, y = next(self.trainloader.__iter__())  # ad infinum sampling shuffled data
            self.model.zero_grad()
            loss = self.model.log_prob(X, y)
            loss.backward()

            if self.r(k) < self.beta:
                self.explore_step(k)
            else:
                self.chain.append(self.exploit_step(k))

        return self.chain


class cSGHMC(CSG_MCMC):
    def __init__(self, *args, **kwargs):

        super(self, cSGHMC).__init__(*args, **kwargs)

        # initalize the momentum variables on top of the parameter
        for p in self.model.parameters():
            p.nu = torch.zeros(p.size())  # todo: send this to cuda
            p.Noise = td.Normal(torch.zeros(p.size()), 1.)

    def exploit_step(self, eta, gamma, k):
        """
        :param eta: (1-eta) is momentum
        :param gamma: estimated noise of the minibatch
        :param k: current iteration step
        """
        # FIXME!! unclear reference in paper
        #   potentially see "complete reciepe for sgmcmc"

        for p in self.model.parameters():
            # update theta parameter with nu_{k-1}
            print(p.data)
            p.data.add_(p.nu)

            # update the momentum variables nu
            alpha = self.alpha(k)
            noise = p.Noise.sample()
            print(p.nu)
            p.nu.add_(-alpha * p.grad.data - eta * p.nu + \
                      torch.sqrt(2 * (eta - gamma) * alpha) * noise)


class cSGLD(CSG_MCMC):
    def __init__(self, *args, **kwargs):
        super(cSGLD, self).__init__(*args, **kwargs)

        # initalize the Noise distributions on top of the parameter
        for p in self.model.parameters():
            p.Noise = td.Normal(torch.zeros(p.size()), 1.)

    def exploit_step(self, k):
        """
        :param k:
        :returns vec model # TODO fix this eventually to state_dicts
        """

        for p in self.model.parameters():
            alpha = self.alpha(k)
            print(p.data)
            p.data.add_(-alpha * p.grad.data + torch.sqrt(2 * alpha) * p.Noise.sample())

        return self.model.state_dict()
        # return self.model.vec


if __name__ == '__main__':
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel

    # model specification
    no_in = 10
    no_out = 1

    # single Hidden Unit Example
    model = Hidden_ProbModel(no_in, no_out, bias=True, activation=nn.Identity())

    # model.reset_parameters()
    print(model.parameters_dict)

    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = model.likelihood(X).sample()

    # setting up the sampler
    batch_size = 10
    size_dataset = len(X)
    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    # SAMPLING cSGLD
    csgld = cSGLD(model, trainloader, alpha0=0.1, M=50, K=2000, beta=0.9)
    csgld.sample()

    # SAMPLING cSGHMC
    csghmc = cSGHMC(model, trainloader, alpha0=0.1, M=50, K=2000, beta=0.9)
    csghmc.sample()
