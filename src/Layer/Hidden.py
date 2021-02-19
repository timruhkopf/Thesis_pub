import torch
import torch.nn as nn
import torch.distributions as td
from copy import deepcopy
from src.Util.Util_Model import Util_Model


class Hidden(nn.Module, Util_Model):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """
        Hidden Layer, that provides its prior_log_prob model

        :param no_in:
        :param no_out:
        :param bias: bool, whether or not to use bias in the computation
        :param activation: Any "nn.activation function" instance. Examine
        https://src.org/docs/stable/nn.html?highlight=nn%20relu#torch.nn.ReLU
        should work with their functional correspondance e.g. F.relu. Notably,
        A final Layer in Regression setup looks like e.g.
        Hidden(10, 1, bias=False, activation=nn.Identity())
        """

        super().__init__()

        self.no_in = no_in
        self.no_out = no_out
        self.has_bias = bias
        self.activation = activation

        self.define_model()

        # initialize the parameters
        self.reset_parameters()
        self.true_model = deepcopy(self.state_dict())

    def define_model(self):
        self.tau_w = torch.tensor([1.])

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W.dist = td.Normal(loc=torch.zeros_like(self.W.data), scale=self.tau_w)

        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.dist = td.Normal(torch.zeros(self.no_out), 1.)

    @torch.no_grad()
    def reset_parameters(self):
        for p in self.parameters():
            p.data = p.dist.sample()

        self.init_model = deepcopy(self.state_dict())

    def update_distributions(self):
        return None

    def forward(self, X):
        XW = X @ self.W
        if self.has_bias:
            XW += self.b
        return self.activation(XW)

    def prior_log_prob(self):
        """evaluate each parameter in respective distrib."""
        value = torch.tensor(0.)
        for p in self.parameters():
            value += p.dist.log_prob(p).sum()
        return value

    def sample_model(self, n):
        self.true_model = deepcopy(self.state_dict())
        X_dist = td.Uniform(torch.ones(self.no_in) * (-10.), torch.ones(self.no_in) * 10.)
        X = X_dist.sample(torch.Size([n]))
        y = self.likelihood(X).sample()

        return X, y

