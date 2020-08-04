import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Util.DistributionUtil import LogTransform


class Group_lasso(Hidden):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True):
        """
        Group Lasso Layer, which is essentially a Hidden Layer, but with a different
        prior structure: W is partitioned columwise - such that all weights outgoing
        the first variable are shrunken by bayesian lasso.
        for params see Hidden
        :param bijected: bool. indicates whether or not the shrinkage variances
        'tau' and 'lamb' are to be bijected i.e. unconstrained on space R.
        as consequence, the self.update_method must change. the self.prior_log_prob
        is automatically adjusted by the jacobian via td.TransformedDistribution.
        """
        self.bijected = bijected
        Hidden.__init__(self, no_in, no_out, bias, activation)
        self.dist['alpha'] = td.LogNormal(0., scale=1. / 20.)
        self.alpha_chain = list()

    def define_model(self):
        self.m = self.no_out  # single "group size" to be penalized

        # hyperparam of tau
        self.lamb_ = nn.Parameter(torch.Tensor(1))
        self.lamb = torch.tensor(1.)  # just to instantiate dist
        self.dist['lamb'] = td.HalfCauchy(scale=1.)

        # hyperparam of W: single variance parameter for group
        self.tau_ = nn.Parameter(torch.Tensor(1))
        self.tau = torch.tensor(1.)  # just to instantiate dist
        self.dist['tau'] = td.Gamma((self.m + 1) / 2, (self.lamb ** 2) / 2)

        if self.bijected:
            self.dist['lamb'] = td.TransformedDistribution(self.dist['lamb'], LogTransform())
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())

        # Group lasso structure of W
        self.W_ = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W = None
        self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_in), self.tau)
        # FIXME: check sigma dependence in W_shrinked: \beta_g | tau²_g, sigma² ~ MVN
        self.dist['W'] = td.Normal(torch.zeros(self.no_in * (self.no_out - 1)), 1.)

        # add optional bias
        if self.bias:
            self.b_ = nn.Parameter(torch.Tensor(self.no_out))
            self.b = None
            self.tau_b = 1.
            self.b = None
            self.dist['b'] = td.Normal(0., 1.)

    def update_distributions(self):
        """due to the hierarchical structure, the distributions parameters must be updated
        Note, that this function is intended to be called immediately after vec_to_attr
        in order to get thr correct log prob. This function does not """
        # self.dist['tau'].__init__((self.m + 1) / 2, self.lamb ** 2)
        # self.dist['W_shrinked'].__init__(torch.zeros(self.no_in), self.tau)

        # BE WELL AWARE of the required inverse of the bijector in order to update
        # the conditional distributions accordingly
        if self.bijected:
            self.dist['tau'].base_dist.rate = \
                self.dist['lamb'].transforms[0]._inverse(self.lamb) ** 2 / 2
            self.dist['W_shrinked'].scale = \
                self.dist['tau'].transforms[0]._inverse(self.tau)
        else:
            self.dist['tau'].rate = self.lamb ** 2 / 2
            self.dist['W_shrinked'].scale = self.tau

    def reset_parameters(self, seperated=False):
        """sampling method to instantiate the parameters"""

        self.lamb = self.dist['lamb'].sample()
        self.update_distributions()  # to ensure tau's dist is updated properly

        if seperated:  # allows XOR Decision in data generating procecss
            if self.bijected:
                self.tau = self.dist['tau'].transforms[0](torch.tensor(0.001))
            else:
                self.tau = torch.tensor(0.001)
            self.update_distributions()  # to ensure W_shrinked's dist is updated properly
        else:
            self.tau = self.dist['tau'].sample()
            self.update_distributions()  # to ensure W_shrinked's dist is updated properly

        # partition W according to prior model

        self.W = torch.cat([self.dist['W_shrinked'].sample().view(self.no_in, 1),
                            self.dist['W'].sample().view(self.no_in, self.no_out - 1)],
                           dim=1)
        if self.bias:
            self.b = self.dist['b'].sample()

        # setting the nn.Parameters's starting value
        self.lamb_.data = self.lamb
        self.tau_.data = self.tau
        self.W_.data = self.W
        if self.bias:
            self.b_.data = self.b

    def prior_log_prob(self):
        """evaluate each parameter in respective distrib."""

        param_names = self.p_names
        param_names.remove('W')

        value = torch.tensor(0.)
        for name in param_names:
            value += self.dist[name].log_prob(self.get_param(name)).sum()

        # W's split & vectorized priors
        value += self.dist['W_shrinked'].log_prob(self.W[:, 0]).sum() + \
                 self.dist['W'].log_prob(self.W[:, 1:].reshape(self.no_in * (self.no_out - 1))).sum()

        return value

    @property
    def alpha(self):
        """attribute in interval [0,1], which decides upon the degree of how much
        weight gam gets in likelihoods'mu= bnn() + alpha *gam()"""
        # FIXME alpha value for mu in likelihood depending on used shrinkage layer

        # as update_params already changed the tau value here explicitly
        tau = self.dist['W_shrinked'].scale

        # 1- : since small tau indicate high shrinkage & the possibility to
        # estimate using GAM, this means that alpha should be (close to) 1
        self.alpha_chain.append(1 - self.dist['alpha'].cdf(tau))
        return self.alpha_chain[-1]

    @property
    def alpha_probab(self):
        """
        this is a potential candidate for alpha, to choose as to whether or not
        the model should estimate the effect of x_1 with BNN or with GAM.
        This procedure uses the shrinkage variance tau to decide this probabilistically,
        ensuring that the GAM parameters will be optimized, even if the model should
        favour alpha = 0 i.e. estimating x_1 in the bnn without gam (considering all
        interactions with other variables)
        """
        tau = self.dist['W_shrinked'].scale

        # map tau to [0,1] interval, making it a probability
        # be careful as the mapping is informative prior knowledge!
        # Note further, that alpha is not learned!
        pi = 1 - self.dist['alpha'].cdf(tau)
        self.alpha_chain.append(pi)
        return td.Bernoulli(pi).sample()


if __name__ == '__main__':
    glasso = Group_lasso(3, 1, bias=True, activation=nn.Identity())
    glasso.true_model = glasso.vec

    # generate data
    X_dist = td.Uniform(torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.]))
    X = X_dist.sample(torch.Size([100])).view(100, 3)
    X.detach()
    X.requires_grad_()

    y = glasso.likelihood(X).sample()

    # check reset_parameters
    glasso.parameters_dict
    glasso.vec_to_attrs(torch.cat([i * torch.ones(glasso.__getattribute__(p).nelement())
                                   for i, p in enumerate(glasso.p_names)]))
    glasso.parameters_dict
    glasso(X)

    glasso.reset_parameters()
    glasso.parameters_dict
    glasso(X)

    # check prior_log_prob
    glasso.reset_parameters()
    print(glasso.prior_log_prob())

    # check update_distributions
    glasso.reset_parameters()
    print(glasso.W[:, 0])
    print(glasso.dist['W_shrinked'].scale)
    # value of log prob changed due to change in variance
    print(glasso.dist['W_shrinked'].log_prob(0.))
    print(glasso.dist['W_shrinked'].cdf(0.))  # location still 0

    # check alpha value
    # glasso.alpha

    # check

    # check "shrinkage_regression" example on being samplable
    import hamiltorch

    N = 2000
    hamiltorch.set_random_seed(123)

    step_size = 0.15
    num_samples = 50
    num_steps_per_sample = 25
    threshold = 1e-3
    softabs_const = 10 ** 6
    L = 25

    glasso.closure_log_prob(X, y)
    glasso.reset_parameters()
    theta = hamiltorch.util.flatten(glasso)
    print(glasso.log_prob(theta))

    # FIXME: failing internally with Glasso during irmhmc sampling due to singular U in cholesky (MVN , fisher)
    # params_irmhmc_bij = hamiltorch.sample(
    #     log_prob_func=glasso.log_prob, params_init=theta, num_samples=N,
    #     step_size=step_size, num_steps_per_sample=L, sampler=hamiltorch.Sampler.RMHMC,
    #     integrator=hamiltorch.Integrator.IMPLICIT, fixed_point_max_iterations=1000,
    #     fixed_point_threshold=1e-05)

    glasso.reset_parameters()
    theta = hamiltorch.util.flatten(glasso)
    glasso.log_prob(theta)

    # HMC NUTS
    N = 2000
    step_size = .3
    L = 5
    burn = 500
    N_nuts = burn + N
    glasso.reset_parameters()
    glasso.vec
    params_hmc_nuts = hamiltorch.sample(log_prob_func=glasso.log_prob, params_init=theta,
                                        num_samples=N_nuts, step_size=step_size, num_steps_per_sample=L,
                                        sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
                                        desired_accept_rate=0.8)
    glasso.invert_bij('tau')
    glasso.invert_bij('lamb')
    print()
