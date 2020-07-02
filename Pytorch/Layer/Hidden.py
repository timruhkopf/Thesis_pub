import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F


class Hidden(nn.Module): # (nn.Module):
    # TODO : see how to subclass torch layer
    activ = {'identity': lambda x: x,
             'relu': lambda x: F.relu(x),
             'sigmoid': lambda x: F.sigmoid(x),
             'tanh': lambda x: F.tanh(x)}

    def __init__(self, no_in, no_units, bias=True, activation='relu'):
        # super(Hidden, self).__init__()
        super().__init__()
        self.no_in = no_in
        self.no_units = no_units
        self.bias = bias

        self.activation = self.activ[activation]
        self.linear = nn.Linear(no_in, no_units, bias)

        # self.parameters = ['W']
        # self.bijectors = ['Identity']  FIXME Bijectors: has Torch functionals?

        # fixme: check implementation with hyperparam tau & joint density!
        self.W_prior = td.Normal(torch.zeros(no_units * no_in),
                                 torch.tensor([1.]))

        self.tau_b = 1.
        if self.bias is not None:
            # self.parameters.append('b')
            # self.bijectors.append('Identity')
            self.b_prior = td.Normal(torch.zeros(no_units), self.tau_b)

        # Initialize the layer
        self.sample()

    def sample(self):
        """Sampling the weight and biases from standard Normal distributions
        sampling a model causes the history to be removed potentially."""
        self.linear.weight = nn.Parameter(
            self.W_prior.sample().view(self.no_units, self.no_in))
        if self.bias:
            self.linear.bias = nn.Parameter(self.b_prior.sample())

    def prior_log_prob(self):
        log_prob = self.W_prior.log_prob(self.linear.weight.view(self.no_in * self.no_units)).sum()
        if self.bias:
            log_prob += self.b_prior.log_prob(self.linear.bias).sum()

        return log_prob

    def __call__(self, X):
        """:returns the mean i.e. XW^T """
        return self.activation(self.linear(X))


if __name__ == '__main__':
    # replacing Parameters after Sampling values for them.
    x = nn.Linear(10, 1)
    x(torch.ones(10))
    x.bias = nn.Parameter(torch.tensor([1.]), True)
    x(torch.ones(10))
    x.weight.sum() + 1.  # (+ bias)

    # Regression example (only prior model) -------------------------------------------------
    reg = Hidden(no_in=2, no_units=1, bias=True, activation='identity')

    X_dist = td.Uniform(torch.tensor([-10., -10]), torch.tensor([10., 10]))
    X = X_dist.sample(torch.Size([100]))
    beta = torch.tensor([[1., 2.]])
    reg.linear.weight = nn.Parameter(beta)
    mu = reg(X)

    # test log prob
    reg.prior_log_prob()

    # 2 consecutive Hidden Unit example --------------------------------------------------------------
    X_dist = td.Uniform(torch.tensor([-10., -10]), torch.tensor([10., 10]))
    X = X_dist.sample(torch.Size([100]))

    first_layer = Hidden(no_in=2, no_units=10, bias=True, activation='relu')

    W = torch.ones(10,2)
    first_layer.linear.weight = nn.Parameter(W)
    first_out = first_layer(X)

    second_layer = Hidden(no_in=first_out.shape[1], no_units=1, bias=True, activation='identity')
    mu = second_layer(first_out)

    # prior model log_prob
    first_layer.prior_log_prob()
    second_layer.prior_log_prob()

    # check the sampling function once more
    first_layer.sample()
    second_layer.sample()
    first_layer.prior_log_prob()
    second_layer.prior_log_prob()

