import torch
import torch.nn as nn
import torch.distributions as td

from copy import deepcopy

from src.Layer.Hidden import Hidden
from src.Util.Util_Distribution import LogTransform


class GroupLasso(Hidden):
    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU(), no_shrink=1):
        """
        Implements the following assumptions
        (0) Hierarchical Shrinkage
        m_g = group size
        w_g |τ_g2  ∼ MVN (0, τ_g^2 I_{m_g} )
        τ g |λ  ~Ga((m_g + 1)/2,  λ^2 / 2)
        λ ∼ C+ (0, 1)

        (1) Unshrunken
        w_g ~ MVN(0, I_{m_g}

        Group Lasso Layer, which is essentially a Hidden Layer, but with a different
        prior structure: W is partitioned rowwise - such that all weights outgoing
        the first no_shrink-variables are shrunken by bayesian lasso.
        for params see Hidden
        :param no_shrink: number of shrinked variables; the shrinked variables
        are the first no_shrink[ed] variables in X
        """
        self.no_shrink = no_shrink
        Hidden.__init__(self, no_in, no_out, bias, activation)

    def define_model(self):
        # delegate bijection to state -- define tau's distrution there
        self.m = self.no_out  # single "group size" to be penalized
        self.lamb = nn.Parameter(torch.tensor([1.]))
        self.lamb.dist = td.HalfCauchy(scale=torch.tensor([1.]))
        self.tau = nn.Parameter(torch.tensor([1.] * self.no_shrink))
        self.tau.dist = td.Gamma(torch.tensor([(self.m + 1) / 2] * self.no_shrink),
                                 torch.tensor([self.lamb.data] * self.no_shrink) ** 2 / 2)

        # TODO REFACTOR THE FOLLOWING. consider BijDistParameter
        # biject lamb
        self.lamb.dist = td.TransformedDistribution(self.lamb.dist, LogTransform())
        self.lamb.inv = lambda val: self.lamb.dist.transforms[0]._inverse(val)
        self.lamb.bij = lambda val: self.lamb.dist.transforms[0]._call(val)
        self.lamb.data = self.lamb.bij(self.lamb.data)

        # biject tau
        self.tau.dist = td.TransformedDistribution(self.tau.dist, LogTransform())
        self.tau.inv = lambda val: self.tau.dist.transforms[0]._inverse(val)
        self.tau.bij = lambda val: self.tau.dist.transforms[0]._call(val)
        self.tau.data = self.tau.bij(self.tau.data)

        # FIXME: check sigma dependence in W_shrinked: \beta_g | tau²_g, sigma² ~ MVN
        # Partitioned Group Lasso Prior Set up for W (single variable - single row shrinkage)
        self.W = nn.Parameter(torch.ones(self.no_in, self.no_out))
        self.W.dist_shrinked = td.Normal(
            loc=torch.zeros(self.no_shrink, self.no_out),
            scale=self.tau.inv(self.tau.data).view(-1, 1))  # shared variance (broadcasted rowwise)
        if self.no_in - self.no_shrink > 0:  # if there are any non-shrinked variables
            self.W.dist = td.Normal(torch.zeros((self.no_in - self.no_shrink), self.no_out), torch.tensor([1.]))

        # add optional bias
        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.dist = td.Normal(torch.zeros(self.no_out), 1.)

    def prior_log_prob(self):
        """evaluate each parameter in respective distrib."""
        # all param except W
        value = sum(p.dist.log_prob(p).sum() for name, p in self.named_parameters() if name != 'W')

        # Now calculate W
        value += self.W.dist_shrinked.log_prob(self.W[:self.no_shrink]).sum()  # first row is shrunken
        if self.no_in - self.no_shrink > 0:
            value += self.W.dist.log_prob(self.W[self.no_shrink:]).sum()  # rest of matrix is unshrunken
        return value

    def reset_parameters(self, separated=False):
        """
        :param separated: bool: if True, the model is sampled with tau variance of TODO find value!
        False: tau is sampled from its distribution.
        """

        # Setting tau according to seperated
        if separated:
            self.lamb.data = self.lamb.bij(torch.tensor([0.01]))
            self.tau.data = self.tau.bij(torch.tensor([0.05] * self.no_shrink))  # < ---------------------- fix value
        else:
            self.lamb.data = self.lamb.dist.sample()
            self.update_distributions()  # communicate to tau
            self.tau.data = self.tau.dist.sample()
        self.update_distributions()  # to ensure W's shrunken dist is updated before sampling it
        self.reset_Wb()

        self.init_model = deepcopy(self.state_dict())

    def reset_Wb(self):
        """sampling method for base parameter. In particular encapsulates the
        shrinkage / unshrunken partition in W with the respective distributions.
        This logic will be the same for derivatives of shrinkage methods"""

        # partition W according to prior model
        self.W.data[:self.no_shrink] = self.W.dist_shrinked.sample()  # .view(self.no_shrink, self.no_out)
        if self.no_in - self.no_shrink > 0:
            self.W.data[self.no_shrink:] = self.W.dist.sample()  # .view(self.no_in - self.no_shrink, self.no_out)

        if self.has_bias:
            self.b.data = self.b.dist.sample()

    def update_distributions(self):
        self.tau.dist.base_dist.rate = self.lamb.inv(self.lamb.data) ** 2 / 2

        # expand the shared parameter accordingly (memory efficient)
        self.W.dist_shrinked.scale = self.tau.inv(self.tau.data).view(-1, 1).expand(self.W.dist_shrinked.batch_shape)

    def plot_tau(self, chain):
        """
        :param chain: list of state_dicts
        """
        # TODO convert bijected chain # sampling will always
        raise NotImplementedError()
        self.tau  # current
        self.init_model
        self.true_model

    # def invert_bij(self, chain=[]):
    #     """invert chain optionally
    #     #FIXME: multiple calls (e.g. change multiple chains) to this will change the value of tau!"""
    #     self.tau.data = self.tau.inv(self.tau.data)
    #     for m in (self.init_model, self.true_model, *chain):
    #         m['tau'] = self.tau.inv(m['tau'])
    #
    #     if bool(chain):
    #         return chain

    # @property
    # def alpha(self):
    #     """attribute in interval [0,1], which decides upon the degree of how much
    #     weight gam gets in likelihoods'mu= bnn() + alpha *gam()"""
    #     # FIXME alpha value for mu in likelihood depending on used shrinkage layer
    #
    #     # as update_params already changed the tau value here explicitly
    #     if self.bijected:
    #         print('tau (unbijected) ', self.tau)
    #         print('tau (bijected)', self.dist['W_shrinked'].scale)
    #         # self.dist['W_shrinked'].scale += torch.tensor(1.)
    #         # tau = self.dist['tau'].transforms[0]._inverse(self.dist['W_shrinked'].scale)
    #         # print(tau)
    #         tau = self.dist['W_shrinked'].scale
    #         print('alpha ', self.dist['alpha'].cdf(self.dist['W_shrinked'].scale))
    #     else:
    #         tau = self.dist['W_shrinked'].scale
    #
    #     # 1- : since small tau indicate high shrinkage & the possibility to
    #     # estimate using GAM, this means that alpha should be (close to) 1
    #     return 1 - self.dist['alpha'].cdf(tau)

    # @property
    # def alpha_probab(self):
    #     """
    #     this is a potential candidate for alpha, to choose as to whether or not
    #     the model should estimate the effect of x_1 with BNN or with GAM.
    #     This procedure uses the shrinkage variance tau to decide this probabilistically,
    #     ensuring that the GAM parameters will be optimized, even if the model should
    #     favour alpha = 0 i.e. estimating x_1 in the bnn without gam (considering all
    #     interactions with other variables)
    #     """
    #     if self.bijected:
    #         tau = self.tau.dist.transforms[0]._inverse(self.W.dist_shrinked.scale)
    #     else:
    #         tau = self.W.dist_shrinked.scale
    #
    #     # map tau to [0,1] interval, making it a probability
    #     # be careful as the mapping is informative prior knowledge!
    #     # Note further, that alpha is not learned!
    #     pi = torch.tensor(1.) - self.dist['alpha'].cdf(tau)
    #     if pi.item() < 0.01:
    #         pi = torch.tensor([0.01])
    #     print(pi)
    #     delta = td.Bernoulli(pi).sample()  # FIXME despite working on its own seems to cause an error in the sampler
    #     print(delta)
    #     return delta
    #
    # @property
    # def alpha_const(self):
    #     return torch.tensor(1.)


if __name__ == '__main__':
    glasso = GroupLasso(2, 1, bias=True, no_shrink=2)
    glasso.prior_log_prob()

    glasso.reset_parameters(separated=True)
    assert glasso.tau.inv(glasso.tau.data) == torch.tensor([0.01])

    X, y = glasso.sample_model(100)

    glasso.plot(X, y)

    p = nn.Parameter(torch.tensor([[1.], [10.]]))
    dist = td.Normal(
        loc=torch.zeros((2, 3)),
        scale=p)
    torch.tensor([[1.], [2.]]).expand(glasso.W.dist_shrinked.batch_shape)
