import torch
import torch.nn as nn
import torch.distributions as td


from src.Util.Util_Distribution import LogTransform
from .GAM import GAM
from src.Util.Util_bspline import get_design, diff_mat1D


class GAM_Fahrmeir(GAM):

    def define_model(self):
        """splitting up the rank & null space part of the singular normal distribution
        null space part corresponds to the level - and is assigned a flat prior"""

        # construct double (null space) penalty
        self.K = torch.tensor(diff_mat1D(self.no_basis, self.order)[1], dtype=torch.float32, requires_grad=False)
        eig_val, eig_vec = torch.eig(self.K, eigenvectors=True)
        threshold = 1e-2

        Q_pos = torch.diag(eig_val[eig_val[:, 0] > threshold, 0])  # positive eigenval

        self.rangespace = eig_vec[:, eig_val[:, 0] > threshold]
        self.nullspace = eig_vec[:, eig_val[:, 0] < threshold]
        # self.rangespace @ torch.diag(eig_val[eig_val[:, 0] > threshold, 0]) @ self.rangespace.t()
        # self.range_cov = self.rangespace.t() @ self.K @ torch.inverse(self.rangespace.t())

        self.X = self.nullspace.t()
        L = self.rangespace @ torch.diag(torch.sqrt(eig_val[eig_val[:, 0] > threshold, 0]))
        self.U = L @ torch.inverse(L.t() @ L)

        X_t = Z @ self.X.t()
        U_t = Z @ self.U

        self.dist['beta'] = td.Normal(0., 1.)
        self.dist['random_gamma'] = td.Normal(torch.zeros(U_t.shape[1]), self.tau)

        # set up tau & W
        self.tau = nn.Parameter(torch.Tensor(1))
        self.dist['tau'] = td.Gamma(torch.tensor([2.]), torch.tensor([2.]))
        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.dist['W0'] = td.Uniform(-1, 1)

        if self.bijected:
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())
            self.tau.data = self.dist['tau'].sample()
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)
            self.dist['W'] = td.MultivariateNormal(
                torch.zeros(self.no_basis - 1), self.tau_bij ** -1 * self.range_cov)
        else:
            self.tau.data = self.dist['tau'].sample()  # to ensure dist W is set up properly
            self.dist['W'] = td.MultivariateNormal(
                torch.zeros(self.no_basis - 1), self.tau ** -1 * self.range_cov)

    def reset_parameters(self, tau=None):
        self.tau.data = self.dist['tau'].sample()
        self.update_distributions()

        self.W.data[0] = self.dist['W0'].sample()
        self.W.data[1:] = self.dist['W'].sample().view(self.no_in - 1, self.no_out)

        self.init_model = deepcopy(self.state_dict())

    def update_distributions(self):
        if self.bijected:
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)
            self.dist['W'] = td.MultivariateNormal(
                torch.zeros(self.no_basis - 1), self.tau_bij ** -1 * self.range_cov)
        else:
            self.dist['W'] = td.MultivariateNormal(
                torch.zeros(self.no_basis - 1), self.tau ** -1 * self.range_cov)

    def prior_log_prob(self):
        return self.dist['W'].log_prob(self.W[1:]) + \
               self.dist['W0'].log_prob(self.W[0]) + \
               self.dist['tau'].log_prob(self.tau)


if __name__ == '__main__':
    from copy import deepcopy

    # dense example
    no_basis = 20
    n = 1000
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([n]))
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis),
                     dtype=torch.float32, requires_grad=False)

    gam = GAM_Fahrmeir(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=True)
    # gam = GAM(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=False)

    gam.reset_parameters(tau=torch.tensor([0.001]))
    # gam.reset_parameters(tau=torch.tensor([0.01]))
    gam.true_model = deepcopy(gam.state_dict())
    y = gam.likelihood(Z).sample()

    gam.reset_parameters(tau=torch.tensor([1.]))

    gam.plot(X, y)

    gam.forward(Z)
    gam.prior_log_prob()
