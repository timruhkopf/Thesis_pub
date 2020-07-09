import torch
import torch.distributions as td
import torch.nn as nn
from hamiltorch.util import flatten, unflatten

from Pytorch.Layer.Hidden import Hidden

import numpy as np
from Tensorflow.Effects.bspline import get_design, diff_mat1D
from Tensorflow.Effects.Cases1D.Bspline_K import Bspline_K
from Tensorflow.Effects.Cases1D.Bspline_cum import Bspline_cum


class GAM(Hidden):
    def __init__(self, xgrid=(0, 10, 0.5), order=1, no_basis=10, no_out=1, activation=nn.Identity(), ):
        """
        RandomWalk Prior Model on Gamma (W) vector.
        Be carefull to transform the Data beforehand with some DeBoor Style algorithm.
        This Module mereley implements the Hidden behaviour + Random Walk Prior
        :param no_basis: number of basis from the de boor basis expansion to expect (=no_columns of X)
        which is in fact the no_in of Hidden.
        :param order: difference order to create the Precision (/Penalty) matrix K
        """
        super().__init__(no_basis, no_out, bias=False, activation=activation)
        self.xgrid = xgrid
        self.order = order
        self.no_basis = no_basis
        self.no_in = no_basis
        self.K = torch.tensor(diff_mat1D(no_basis, order)[1], dtype=torch.float32, requires_grad=False)
        self.cov = torch.inverse(self.K[1:, 1:])  # FIXME: Multivariate Normal cholesky decomp fails!

        self.tau_ = nn.Parameter(torch.Tensor(1))
        self.dist_tau = td.Gamma(0.1, 0.1)

        self.sample()

    def sample(self, tau=1., mode='K'):
        """
        Sample the prior model, instantiating the data model
        :param xgrid: defining space for the Bspline expansion. Notice, that
        int((xgrid[1] - xgrid[0]) // 0.5) == self.no_basis is required!
        :param tau: if not None, it is the inverse variance (smoothness) of
        randomwalkprior. If None, the self.dist_tau is used for sampling tau
        # FIXME: carefull with sigma/tau=lambda relation
        :param mode: 'K' or 'cum'. 'K' specifies, that the model is sampled based
        on the 1/tau * null-space-penalized precision matrix, which is plugged into MVN
        :return: None. Inplace self.W, self.tau
        """

        if int((self.xgrid[1] - self.xgrid[0]) // 0.5) != self.no_basis:
            raise ValueError('The Specified range(*xgrid) does not imply '
                             'no_basis (len(range) must be equal to no_basis)')
        if tau is None:
            # FIXME: carefull with transformed distributions! (need to transform tau back)
            self.tau = self.dist_tau.sample()
        else:
            self.tau = tau

        # FIXME: need to specify tau / variance)
        if mode == 'K':
            bspline_k = Bspline_K(self.xgrid, no_coef=self.no_basis, order=1, sig_Q=0.1, sig_Q0=0.01, threshold=10 **
                                                                                                                -3)
            self.W = torch.tensor(bspline_k.z, dtype=torch.float32).view(self.no_basis,self.no_out)
            self.W_.data = self.W

        elif mode == 'cum':
            bspline_cum = Bspline_cum(self.xgrid, coef_scale=0.3)
            self.W = torch.tensor(bspline_cum.z, dtype=torch.float32).view(self.no_out, self.no_basis)
            self.W_.data = self.W

        else:
            raise ValueError('Mode is incorreclty specified')

        # gamma = torch.cat(
        #     [td.Uniform(torch.tensor([-1.]), torch.tensor([1.])).sample(),
        #      td.MultivariateNormal(torch.zeros(self.no_basis - 1), self.cov).sample()],
        #     dim=0).view(self.no_out, self.no_basis)
        # self.W = gamma
        # self.W_.data = self.W
        #
        # self.tau = self.dist_tau.sample()
        # self.tau_.data = self.tau

    def prior_log_prob(self):
        """
        returns: log_probability sum of gamma & tau and is calculating the
         RandomWalkPrior
        """

        # FIXME: CHECK IF IDENTIFIABILITY CAN BE HELPED IF SELF.K where penalized
        const = - 0.5 * torch.log(torch.tensor(2 * np.pi * self.tau))  # fixme: check if tau is correct here!
        kernel = (2 * self.tau) ** -1 * self.W.t() @ self.K @ self.W
        return sum(const - kernel + self.dist_tau.log_prob(self.tau))


if __name__ == '__main__':
    # dense example
    no_basis = 20
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([100]))
    X.detach()
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis), dtype=torch.float32, requires_grad=False)
    Z.detach()

    gam = GAM(no_basis=no_basis, order=1)
    gam.sample()
    gam(Z)

    y = gam.likelihood(Z).sample()
    y.detach()
    y.requires_grad_()

    # prior log prob example
    gam.prior_log_prob()

    # log_prob sg example
    theta = flatten(gam)
    # gam.closure_log_prob()
    # gam.log_prob(Z, y, theta)

    # log_prob full dataset example
    gam.closure_log_prob(Z, y)
    gam.log_prob(theta)

    gam.n_params
    gam.p_names
    unflatten(gam, flatten(gam))

    # plot 1D

    # sample
    import hamiltorch

    N = 2000
    hamiltorch.set_random_seed(123)

    step_size = 0.15
    num_samples = 50
    num_steps_per_sample = 25
    threshold = 1e-3
    softabs_const = 10 ** 6
    L = 25

    params_irmhmc_bij = hamiltorch.sample(
        log_prob_func=gam.log_prob, params_init=theta, num_samples=N,
        step_size=step_size, num_steps_per_sample=L, sampler=hamiltorch.Sampler.RMHMC,
        integrator=hamiltorch.Integrator.IMPLICIT, fixed_point_max_iterations=1000,
        fixed_point_threshold=1e-05)

    # Todo
    #   use penalized K in GAM for prior_log_prob to improve identifiably
