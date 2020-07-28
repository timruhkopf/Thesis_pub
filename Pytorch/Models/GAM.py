import torch
import torch.distributions as td
import torch.nn as nn
from hamiltorch.util import flatten, unflatten

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Util.DistributionUtil import LogTransform

from Tensorflow.Effects.bspline import get_design, diff_mat1D
from Tensorflow.Effects.Cases1D.Bspline_K import Bspline_K
from Tensorflow.Effects.Cases1D.Bspline_cum import Bspline_cum


class GAM(Hidden):
    def __init__(self, xgrid=(0, 10, 0.5), order=1, no_basis=10, no_out=1,
                 activation=nn.Identity(), bijected=False, penK=True):
        """
        RandomWalk Prior Model on Gamma (W) vector.
        Be carefull to transform the Data beforehand with some DeBoor Style algorithm.
        This Module mereley implements the Hidden behaviour + Random Walk Prior
        :param no_basis: number of basis from the de boor basis expansion to expect (=no_columns of X)
        which is in fact the no_in of Hidden.
        :param order: difference order to create the Precision (/Penalty) matrix K
        """
        self.bijected = bijected
        self.xgrid = xgrid
        self.order = order
        self.no_basis = no_basis
        self.K = torch.tensor(diff_mat1D(no_basis, order)[1], dtype=torch.float32, requires_grad=False)
        self.penK = penK
        if self.penK:
            bspline_k = Bspline_K(self.xgrid, no_coef=self.no_basis, order=1,
                                  sig_Q=0.9, sig_Q0=0.1, threshold=10 ** -3)
            self.K = torch.tensor(torch.from_numpy(bspline_k.penQ), dtype=torch.float32)

        self.cov = torch.inverse(self.K[1:, 1:])  # FIXME: Multivariate Normal cholesky decomp fails!

        Hidden.__init__(self, no_basis, no_out, bias=False, activation=activation)

    def define_model(self):

        self.tau_ = nn.Parameter(torch.Tensor(1))
        self.tau = torch.tensor([1.])
        self.dist['tau'] = td.Gamma(0.1, 0.1)

        if self.bijected:
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)

        self.W_ = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W = None
        # self.dist['W'] (RANDOMWALK PRIOR) formulated explicitly in prior model!

    def update_distributions(self):
        if self.bijected:
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)
        else:
            self.tau_bij = self.tau

    def reset_parameters(self, tau=torch.tensor([1.]), mode='K'):
        """
        Sample the prior model, instantiating the data model
        :param xgrid: defining space for the Bspline expansion. Notice, that
        int((xgrid[1] - xgrid[0]) // 0.5) == self.no_basis is required!
        :param tau: if not None, it is the inverse variance (smoothness) of
        randomwalkprior. If None, the self.dist_tau is used for sampling tau
        # FIXME: carefull with sigma/tau=lambda relation
        :param mode: 'K' or 'cum' or 'U-MVN'. 'K' specifies, that the model is sampled based
        on the 1/tau * null-space-penalized precision matrix, which is plugged into MVN.
        'cum' is an actual cumulative randomwalk. 'U-MVN': gamma0 (W0) is initialized
        by a Uniform distribution. gamma without the first value is sampled by
        MVN(0, K[1:, 1:]). Thereby ensuring that MVN is propper (and the W vector
        is similar to a randomwalk prior)
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
        self.update_distributions()

        # FIXME: need to specify tau / variance)
        # Sampling W from RandomWalkPrior (two ways)
        if mode == 'K':  # with nullspace-penalized K as precision for MVN
            bspline_k = Bspline_K(self.xgrid, no_coef=self.no_basis, order=1,
                                  sig_Q=0.1, sig_Q0=0.01, threshold=10 ** -3)
            self.W = torch.tensor(bspline_k.z, dtype=torch.float32).view(self.no_basis, self.no_out)

        elif mode == 'cum':  # sample based on actual cumulative randomwalk
            bspline_cum = Bspline_cum(self.xgrid, coef_scale=0.3)
            self.W = torch.tensor(bspline_cum.z, dtype=torch.float32).view(self.no_out, self.no_basis)

        elif mode == 'U-MVN' and not self.penK:  # Uniform for gamma0 & MVN (0, K[1:, 1:]**-1) as cov
            gamma = torch.cat(
                [td.Uniform(torch.tensor([-1.]), torch.tensor([1.])).sample(),
                 td.MultivariateNormal(torch.zeros(self.no_basis - 1), self.cov).sample()],
                dim=0).view(self.no_out, self.no_basis)
            self.W = gamma

        else:
            raise ValueError('Mode is incorreclty specified')

        self.W_.data = self.W
        self.tau_.data = self.tau

    def prior_log_prob(self):
        """
        returns: log_probability sum of gamma & tau and is calculating the
         RandomWalkPrior
        """
        # FIXME: CHECK IF IDENTIFIABILITY CAN BE HELPED IF SELF.K were penalized
        if self.bijected:
            raise NotImplementedError('GAM prior_log_prob requires bijection!')

        # fixme: check if tau is correct here! (and not 1/tau)
        #  BE VERY CAREFULL IF TAU IS BIJECTED!
        # p321 fahrmeir kneib lang:
        const = - 0.5 * (self.K.shape[0] - 1) * torch.log(self.tau_bij)
        kernel = -(2 * self.tau_bij) ** -1 * self.W.t() @ self.K @ self.W
        return sum(const + kernel + self.dist['tau'].log_prob(self.tau))


if __name__ == '__main__':
    # dense example
    no_basis = 20
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([100]))
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis), dtype=torch.float32, requires_grad=False)
    Z.detach()

    gam = GAM(no_basis=no_basis, order=1)
    gam.reset_parameters()
    gam(Z)

    gam.forward(Z)
    y = gam.likelihood(Z).sample()

    # prior log prob example
    gam.prior_log_prob()

    # log_prob sg example
    theta = flatten(gam)
    # gam.closure_log_prob()
    # gam.log_prob(Z, y, theta)
    print('true:', theta)

    # log_prob full dataset example
    gam.closure_log_prob(Z, y)
    gam.log_prob(theta)

    gam.n_params
    gam.p_names
    unflatten(gam, flatten(gam))

    # plot 1D

    # sample
    import hamiltorch

    # HMC NUTS
    N = 2000
    step_size = .3
    L = 5
    burn = 500
    N_nuts = burn + N
    params_hmc_nuts = hamiltorch.sample(
        log_prob_func=gam.log_prob, params_init=theta,
        num_samples=N_nuts, step_size=step_size, num_steps_per_sample=L,
        sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
        desired_accept_rate=0.8)

    chain_mat = torch.cat(params_hmc_nuts).reshape(len(params_hmc_nuts), -1)
    print(chain_mat.mean(dim=0))

    # evaluate log_probs
    log_probs = torch.Tensor(len(params_hmc_nuts))
    for i, tensor in enumerate(params_hmc_nuts):
        log_probs[i] = gam.log_prob(tensor)

    prior_log = torch.Tensor(len(params_hmc_nuts))
    for i, tensor in enumerate(params_hmc_nuts):
        gam.vec_to_attrs(tensor)
        # print(gam.likelihood(Z).log_prob(y).sum())
        prior_log[i] = gam.prior_log_prob()

    # implicit RMHMC
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
