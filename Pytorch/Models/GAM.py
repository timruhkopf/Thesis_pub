import torch
import torch.distributions as td
import torch.nn as nn

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
            # bspline_k = Bspline_K(self.xgrid, no_coef=self.no_basis, order=1,
            #                       sig_Q=0.9, sig_Q0=0.1, threshold=10 ** -3)
            # self.K = torch.tensor(torch.from_numpy(bspline_k.penQ), dtype=torch.float32)

            # replaced Bspline_K with the actual penalize call for K
            from Tensorflow.Effects.SamplerPrecision import penalize_nullspace
            sig_Q = 0.99
            sig_Q0 = 0.01
            threshold = 10 ** -3

            Sigma, penQ = penalize_nullspace(self.K.numpy(), sig_Q, sig_Q0, threshold, plot=False)
            self.K = torch.from_numpy(penQ).clone().detach().type(torch.FloatTensor)

        self.cov = torch.inverse(self.K[1:, 1:])  # FIXME: Multivariate Normal cholesky decomp fails!

        Hidden.__init__(self, no_basis, no_out, bias=False, activation=activation)

    def define_model(self):
        self.tau = nn.Parameter(torch.Tensor(1))
        self.dist['tau'] = td.Gamma(0.1, 0.1)

        if self.bijected:
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))

        # self.dist['W'] (RANDOMWALK PRIOR) formulated explicitly in prior model!

    def update_distributions(self):
        # tau_bij is the actual variance parameter of W (on R+)- whilest if self.bijected==True,
        # tau becomes the bijected i.e. unconstrained parameter on entire R
        if self.bijected:
            # here tau (is on R) ---> tau_bij (on R+)
            self.tau_bij = self.dist['tau'].transforms[0]._inverse(self.tau)
        else:
            self.tau_bij = self.tau

        # Notice, the explicit prior formulation of W does not allow an update of W's tau here:
        # instead it must be updated in prior_log_prob explicitly

    def reset_parameters(self, tau=torch.tensor([1.]), mode='U-MVN'):
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
            self.tau.data = self.dist_tau.sample()  #

        else:
            self.tau.data = tau
        self.update_distributions()

        # FIXME: need to specify tau / variance)
        # Sampling W from RandomWalkPrior (two ways)
        if mode == 'K':  # with nullspace-penalized K as precision for MVN
            bspline_k = Bspline_K(self.xgrid, no_coef=self.no_basis, order=1,
                                  sig_Q=0.1, sig_Q0=0.01, threshold=10 ** -3)
            self.W.data = torch.tensor(bspline_k.z, dtype=torch.float32).view(self.no_basis, self.no_out)

        elif mode == 'cum':  # sample based on actual cumulative randomwalk
            bspline_cum = Bspline_cum(self.xgrid, coef_scale=0.3)
            self.W.data = torch.tensor(bspline_cum.z, dtype=torch.float32).view(self.no_out, self.no_basis)

        elif mode == 'U-MVN' and self.penK:  # Uniform for gamma0 & MVN (0, K[1:, 1:]**-1) as cov
            gamma = torch.cat(
                [td.Uniform(torch.tensor([-1.]), torch.tensor([1.])).sample(),
                 td.MultivariateNormal(torch.zeros(self.no_basis - 1), (self.tau ** -1) * self.cov).sample()],
                dim=0).view(self.no_out, self.no_basis)
            self.W.data = gamma.view(self.no_basis, self.no_out)

        else:
            raise ValueError('Mode is incorreclty specified')

    def prior_log_prob(self):
        """
        returns: log_probability sum of gamma & tau and is calculating the
         RandomWalkPrior
        """
        # FIXME: CHECK IF IDENTIFIABILITY CAN BE HELPED IF SELF.K were penalized

        # fixme: check if tau is correct here! (and not 1/tau)
        #  BE VERY CAREFULL IF TAU IS BIJECTED!
        # p321 fahrmeir kneib lang:
        const = - 0.5 * (self.K.shape[0] - 1) * torch.log(self.tau_bij)
        kernel = -(2 * self.tau_bij) ** -1 * self.W.t() @ self.K @ self.W
        return sum(const + kernel + self.dist['tau'].log_prob(self.tau))  # notice that tau can be on R if
        # self.bijected is true!

    def plot(self, X, y, chain=None, path=None, title='', **kwargs):
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=self.no_basis), dtype=torch.float32,
                         requires_grad=False)
        df0 = self.predict_states(Z, chain)

        df0['X'] = X.view(X.shape[0], ).numpy()
        df1 = df0.melt('X', value_name='y')
        df1 = df1.rename(columns={'variable': 'functions'})
        plt = self._plot1d(X, y, df1, **kwargs)

        if path is None:
            plt.show()
        else:
            plt.savefig('{}.png'.format(path), bbox_inches='tight')


if __name__ == '__main__':
    from copy import deepcopy

    # dense example
    no_basis = 20
    n = 1000
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([n]))
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis),
                     dtype=torch.float32, requires_grad=False)

    gam = GAM(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=True)
    gam = GAM(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=False)

    gam.reset_parameters(tau=torch.tensor([0.01]))
    gam.true_model = deepcopy(gam.state_dict())
    y = gam.likelihood(Z).sample()
    gam.reset_parameters()
    gam.plot(X, y)

    gam.forward(Z)
    gam.prior_log_prob()

    step_size = 0.005
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    hmc_traj_length = 24
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    # hmc = HMC(reg, X, y, X.shape[0],
    #           step_size=step_size, num_steps=num_steps, burn_in=burn_in, pretrain=pretrain, tune=tune,
    #           traj_length=hmc_traj_length,
    #           num_chains=num_chains)
    # hmc.sample()
    from Pytorch.Samplers.LudwigWinkler import SGNHT

    sgnht = SGNHT(gam, Z, y, X.shape[0],
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  hmc_traj_length=hmc_traj_length,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)
    if len(sgnht.chain) == 1:
        raise ValueError('The chain did not progress beyond first step')

    if all(any(torch.isnan(v)) for chain in (sgnht.chain[-1].values(), sgnht.chain[0]) for v in chain):
        raise ValueError('first and last entry contain nan')

    # plot via stratified chain
    import random

    gam.plot(X[1:100], y[1:100], chain=random.sample(sgnht.chain, len(sgnht.chain) // 10))

    # Todo use penalized K in GAM for prior_log_prob to improve identifiably
