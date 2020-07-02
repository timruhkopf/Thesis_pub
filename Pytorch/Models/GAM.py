import torch
import torch.distributions as td
import torch.nn as nn
from functools import partial
import Pytorch.utils as utils

import numpy as np
from Tensorflow.Effects.bspline import get_design, diff_mat1D


class GAM(nn.Module):
    def __init__(self, no_basis, order=1):
        """
        RandomWalk Prior Model on Gamma
        :param no_basis: number of basis from the de boor basis expansion to expect (=no_columns of X)
        :param order: difference order to create the Precision (/Penalty) matrix K
        """
        super(GAM, self).__init__()
        self.order = order
        self.no_basis = no_basis
        self.K = torch.tensor(diff_mat1D(no_basis, order)[1], dtype=torch.float32, requires_grad=False)
        self.cov = torch.inverse(self.K[1:, 1:])
        self.linear = nn.Linear(no_basis, 1, bias=False)

        # FIXME: log_transform??
        self.TAU_GAMMA = td.TransformedDistribution(td.Gamma(0.1, 0.1), td.ExpTransform())
        self.tau_gamma = nn.Parameter(self.TAU_GAMMA.sample())

        self.SIGMA = td.TransformedDistribution(td.Gamma(0.1, 0.1), td.ExpTransform())
        self.sigma = nn.Parameter(self.SIGMA.sample())

    def sample(self):
        """sample the prior model"""
        gamma = torch.cat(
            [td.Uniform(torch.tensor([-1.]), torch.tensor([1.])).sample(),
             td.MultivariateNormal(torch.zeros(self.no_basis - 1), self.cov).sample()],
            dim=0).view(1, self.no_basis)
        self.linear.weight = nn.Parameter(gamma)
        self.tau_gamma = nn.Parameter(self.TAU_GAMMA.sample())

    def prior_log_prob(self):
        """returns: log_probability sum of of gamma & tau_gamma """
        gamma = self.linear.weight[0]

        const = torch.log(2 * np.pi * self.tau_gamma)
        kernel = (2 * self.tau_gamma) ** -1 * gamma @ self.K @ gamma.t()
        return -const - kernel + \
               self.TAU_GAMMA.log_prob(self.tau_gamma)

    def forward(self, X):
        """:returns the mean i.e. XW^T """
        return self.linear(X)

    # STANDALONE GAM MODEL FOR ESTIMATION:
    def likelihood(self, X, sigmasq):
        return td.Normal(self.forward(X), sigmasq)

    def closure_log_prob(self, X=None, y=None):
        """

        :param X: (Optional) Batched Tensor.
        :param y: (Optional) Batched Tensor.
        :return: if both X & y are provided, a log_prob function is added to bnn,
        which has pre set X & y, making it a full-dataset log_prob. If neither are
        supplied, the added log_prob function requires X & y's for every call on
        log_prob. However, this enables the SG behaviour of the log_prob function.
        """

        def log_prob(self, X, y, theta):
            """By default works as an SG flavour, can be fixed using self.fix_log_prob(X,y)
            to make it a full dataset sampler"""

            # INPLACE ASSIGNMENT OF THETA ON NET WEIGHTS & param
            utils.unflatten(vec=theta, model=self)

            return self.likelihood(X, self.sigma).log_prob(y).sum() + \
                   self.SIGMA.log_prob(self.sigma) + \
                   self.prior_log_prob()

        if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
            print('"Full Dataset" log prob is being set up...')
            if any([X is None, y is None]):
                raise ValueError('"Full Dataset" modus requires specification of X & y')
            self.log_prob = partial(log_prob, self, X, y)
        else:
            print('SG log prob is being set up...')
            self.log_prob = partial(log_prob, self)


if __name__ == '__main__':
    # dense example
    no_basis = 20
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([100]))
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis), dtype=torch.float32, requires_grad=True)

    gam = GAM(no_basis, order=1)
    gam.sample()
    mu = gam(Z)

    y = gam.likelihood(Z, torch.tensor(10.)).sample()

    # sampling Example
    gam.sample()

    # prior log prob example
    gam.prior_log_prob()

    # log_prob sg example
    theta = utils.flatten(gam)
    gam.closure_log_prob()
    gam.log_prob(Z, y, theta)

    # log_prob full dataset example
    gam.closure_log_prob(Z, y)
    gam.log_prob(theta)

    # plot 1D
    print()

    import hamiltorch

    N = 200
    hamiltorch.set_random_seed(123)

    step_size = 0.15
    num_samples = 50
    num_steps_per_sample = 25
    threshold = 1e-3
    softabs_const = 10 ** 6
    L = 25

    params_irmhmc_bij = hamiltorch.sample(log_prob_func=gam.log_prob, params_init=theta, num_samples=N,
                                          step_size=step_size, num_steps_per_sample=L, sampler=hamiltorch.Sampler.RMHMC,
                                          integrator=hamiltorch.Integrator.IMPLICIT, fixed_point_max_iterations=1000,
                                          fixed_point_threshold=1e-05)

