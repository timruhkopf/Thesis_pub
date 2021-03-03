from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.distributions as td
import torch.nn as nn

from src.Layer.Hidden import Hidden
from src.Util.Util_Distribution import LogTransform
from src.Util.Util_bspline import get_design, diff_mat1D


# TODO refactor GAM to state pattern bijected - this way before and after sampling
#  can be unbijected (ease of analysis) -- and sampling can be bijected.
#  notice, that this entails convergence of the chain (the tau's value)


class GAM(Hidden):
    def __init__(self,  # xgrid=(0, 10, 0.5),
                 order=1, no_basis=20, no_out=1,
                 activation=nn.Identity(), bijected=True):
        """
        RandomWalk Prior Model on Gamma (W) vector.
        Be carefull to transform the Data beforehand with some DeBoor Style algorithm.
        This Module mereley implements the Hidden behaviour + Random Walk Prior
        :param no_basis: number of basis from the de boor basis expansion to expect (=no_columns of X)
        which is in fact the no_in of Hidden.
        :param order: difference order to create the Precision (/Penalty) matrix K
        """
        self.bijected = bijected
        # self.xgrid = xgrid
        self.order = order
        self.no_basis = no_basis

        Hidden.__init__(self, no_basis, no_out, bias=False, activation=activation)
        # FIXME: Experimental feature: to increase speed of sampling, always use the like instance
        #  and change the .loc attribute rather than instantiating a new td instance!
        # self.like = td.Normal(self.forward(torch.zeros(1000, 20)), scale=torch.tensor(1.))  # n, no_basis

        # if bijected:
        #     # Consider useful for unbijection of the model
        #     self.bij_parameters = {'tau': self.tau}
        #
        #     # for p in self.bij_parameters.values():
        #     #     p.dist.transforms[0]._inverse(p)

    # TODO : check if cov (without variance factor) can be made a lazy property.
    #   compare with  MultivariateNormal(Distribution).covariance_matrix
    #   @lazy_property # from torch.distributions.utils import lazy_property
    def define_proper_cov(self):
        """replace the numerical zero eigenvalue by fraction*(smallest non-zero eigenval)
        to ensure a propper distribution (see Marra Wood or Wood JAGS)"""
        threshold = 1e-3
        fraction = 1e-1
        self.K = torch.tensor(diff_mat1D(self.no_basis, self.order)[1], dtype=torch.float32, requires_grad=False)
        val, vec = torch.eig(self.K, eigenvectors=True)
        eig_val_2nd = torch.sort(val, axis=0).values[1, :]
        val[val[:, 0] < threshold, :] = eig_val_2nd * fraction
        self.penK = vec @ torch.diag(val[:, 0]) @ vec.t()
        self.cov = torch.inverse(self.penK).detach()

    def define_model(self):
        # setting up a proper covariance for W's random walk prior
        self.define_proper_cov()

        self.tau = nn.Parameter(torch.Tensor(1))
        self.tau.dist = td.Gamma(torch.tensor([2.]), torch.tensor([2.]))
        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))

        if self.bijected:
            self.tau.dist = td.TransformedDistribution(self.tau.dist, LogTransform())
            self.tau.data = self.tau.dist.sample()
            self.tau_bij = self.tau.dist.transforms[0]._inverse(self.tau)
            # decicively tau_bij is not nn.Parameter but could also made a property
            print('rank of gam\'s new covariance matrix: ', torch.matrix_rank(self.tau_bij ** 2 * self.cov), '\n',
                  'desired rank: ', self.cov.shape[0])
            self.W.dist = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                self.tau_bij ** 2 * self.cov)

        else:
            self.tau.data = self.tau.dist.sample()  # to ensure dist W is set up properly
            self.W.dist = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                self.tau ** 2 * self.cov)

    def update_distributions(self):
        # tau_bij is the actual variance parameter of W (on R+)- whilest if self.bijected==True,
        # tau becomes the bijected i.e. unconstrained parameter on entire R
        if self.bijected:
            # here tau (is on R) ---> tau_bij (on R+)
            self.tau_bij = self.tau.dist.transforms[0]._inverse(self.tau)
            # print('UPDATE causal link: tau_bij (on R+)\n',
            #       'constrained tau: ', self.tau, '\n',
            #       'unconstrained tau: ', self.tau_bij)

            try:
                self.W.dist = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                    self.tau_bij ** 2 * self.cov)
            except Exception as e:
                raise RuntimeError('GAM_update: cannot update distribution, as covariance is invalid:\n'
                                   'tau:{}\ntau_bij:{}\ncov:{}'.format(self.tau, self.tau_bij,
                                                                       self.tau_bij ** 2 * self.cov))
        else:
            self.W.dist = td.MultivariateNormal(torch.zeros(self.no_basis),
                                                self.tau ** 2 * self.cov)

    def reset_parameters(self, tau=None):
        """
        Sample the prior model, instantiating the data model
        :param xgrid: defining space for the Bspline expansion. Notice, that
        int((xgrid[1] - xgrid[0]) // 0.5) == self.no_basis is required!
        :param tau: if not None, it is the inverse variance (smoothness) of
        randomwalkprior. If None, the self.dist_tau is used for sampling tau
        # FIXME: carefull with sigma/tau=lambda relation
        :return: None. Inplace self.W, self.tau
        """

        # update tau
        if tau is None:
            # FIXME: carefull with transformed distributions! (need to transform tau back)
            self.tau.data = self.tau.dist.sample()
        else:
            self.tau.data = tau
        self.update_distributions()

        self.W.data = self.W.dist.sample().view(self.no_basis, 1)

        # gamma = torch.cat(
        #     [td.Uniform(torch.tensor([-1.]), torch.tensor([1.])).sample(),
        #      td.MultivariateNormal(torch.zeros(self.no_basis - 1), (self.tau) * self.cov).sample()],
        #     dim=0).view(self.no_out, self.no_basis)
        # self.W.data = gamma.view(self.no_basis, self.no_out)
        self.init_model = deepcopy(self.state_dict())

    def prior_log_prob(self):
        """
        returns: log_probability sum of gamma & tau and is calculating the
         RandomWalkPrior
        """
        # FIXME: CHECK IF IDENTIFIABILITY CAN BE HELPED IF SELF.K were penalized

        # fixme: check if tau is correct here! (and not 1/tau)
        #  BE VERY CAREFULL IF TAU IS BIJECTED!
        # p321 fahrmeir kneib lang:
        # const = - 0.5 * (self.K.shape[0] - 1) * torch.log(self.tau_bij)
        # kernel = -(2 * self.tau_bij) ** -1 * self.W.t() @ self.K @ self.W
        # return sum(const + kernel + self.tau.dist.log_prob(self.tau))  # notice that tau can be on R if
        # # self.bijected is true!

        # tau can be the unconstrained if bijected is true
        return self.W.dist.log_prob(self.W).sum() + self.tau.dist.log_prob(self.tau)

    def plot(self, X, y, chain=None, path=None, **kwargs):
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=self.no_basis), dtype=torch.float32,
                         requires_grad=False)

        # ols=False
        # TODO make penOLS available
        # if ols:
        #     gamma = OLS(Z, y)
        #
        #     X_base = X.clone().numpy()
        #     X_base.sort()
        #     Z_base = torch.tensor(get_design(X_base, degree=2, no_basis=self.no_basis), dtype=torch.float32,
        #                           requires_grad=False)
        #
        #     y_ols = (Z_base @ gamma).view(X.shape[0]).numpy()

        df0 = self.predict_states(chain, Z)
        df0['X'] = X.view(X.shape[0], ).numpy()
        df1 = df0.melt('X', value_name='y')
        df1 = df1.rename(columns={'variable': 'functions'})

        # fig = plt.figure()
        # ax = fig.add_subplot(111, label="1")
        # ax.plot(x=X_base, y=y_ols, linestyle='--', dashes=(5, 1), label='OLS')
        # ax.legend()

        plt = self._plot1d(X, y, df1, **kwargs)

        if path is None:
            plt.show()
        else:
            plt.savefig('{}.pdf'.format(path), bbox_inches='tight')

    def sample_model(self, n):
        X_dist = td.Uniform(-15, 10)
        X = X_dist.sample(torch.Size([n]))
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=self.no_basis),
                         dtype=torch.float32, requires_grad=False)
        self.reset_parameters()
        self.true_model = deepcopy(self.state_dict())
        y = self.likelihood(Z).sample()
        return X, Z, y

    # def log_prob(self, Z, y, vec=None):
    #     # FIXME: Experimental feature: to increase speed of sampling, always use the like instance
    #     #  and change the .loc attribute rather than instantiating a new td instance!
    #     self.update_distributions()
    #     self.like.loc = self.forward(Z)
    #     return self.prior_log_prob() + \
    #            self.like.log_prob(y).sum()

