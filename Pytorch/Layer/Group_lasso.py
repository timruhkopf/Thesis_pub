import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Hidden import Hidden


class Group_lasso(Hidden):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        nn.Module.__init__(self)
        self.bias = False  # to meet the Hidden default standard!
        self.no_in = no_in
        self.no_out = no_out
        self.bias = bias
        self.activation = activation

        self.dist = dict()

        self.m = no_out  # single "group size" to be penalized

        # hyperparam of tau
        self.lamb_ = nn.Parameter(torch.Tensor(1))
        self.lamb = torch.tensor(1.)  # just to instantiate dist
        self.dist['lamb'] = td.HalfCauchy(scale=1.)

        # hyperparam of W: single variance parameter for group
        self.tau_ = nn.Parameter(torch.Tensor(1))
        self.tau = torch.tensor(1.)  # just to instantiate dist
        self.dist['tau'] = td.Gamma((self.m + 1) / 2, (self.lamb ** 2) / 2)

        # Group lasso structure of W
        self.W_ = nn.Parameter(torch.Tensor(no_in, self.no_out))
        self.W = None
        self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_in), self.tau)
        self.dist['W'] = td.Normal(torch.zeros(self.no_in * (self.no_out - 1)), 1.)

        # add optional bias
        if bias:
            self.b_ = nn.Parameter(torch.Tensor(self.no_out))
            self.b = None
            self.tau_b = 1.
            self.b = None
            self.dist['b'] = td.Normal(0., 1.)

        self.reset_parameters()

    def update_distributions(self):
        """due to the hierarchical stucture, the distributions parameters must be updated"""
        # self.dist['tau'].__init__((self.m + 1) / 2, self.lamb ** 2)
        # self.dist['W_shrinked'].__init__(torch.zeros(self.no_in), self.tau)

        self.dist['tau'].rate = self.lamb ** 2 / 2
        self.dist['W_shrinked'].scale = self.tau

    def reset_parameters(self, seperated=False):
        """sampling method to instantiate the parameters"""
        self.lamb = self.dist['lamb'].sample()
        if seperated:
            raise NotImplementedError('still need to figure this out')
        else:
            self.tau = self.dist['tau'].sample()
        self.W = torch.cat([self.dist['W_shrinked'].sample().view(self.no_in, 1),
                            self.dist['W'].sample().view(self.no_in, self.no_out - 1)],
                           dim=1)
        self.b = self.dist['b'].sample()

        # setting the nn.Parameters's starting value
        self.lamb_.data = self.lamb
        self.tau_.data = self.tau
        self.W_.data = self.W
        self.b_.data = self.b

        # update the hierarical distribution structure, to get correct log_prob
        self.update_distributions()

    def vec_to_attrs(self, vec):
        Hidden.vec_to_attrs(self, vec)

        # update the hierarical distribution structure, to get correct prior log_prob
        self.update_distributions()

    def prior_log_prob(self):
        # evaluate each parameter in respective distrib.
        value = torch.tensor(0.)
        for name in ['lamb', 'tau']:
            value += self.dist[name].log_prob(self.__getattribute__(name)).sum()

        # W's split & vectorized priors
        value += self.dist['W_shrinked'].log_prob(self.W[:, 0]).sum() + \
                 self.dist['W'].log_prob(self.W[:, 1:].reshape(self.no_in * (self.no_out - 1))).sum()

        return value


if __name__ == '__main__':
    glasso = Group_lasso(3, 10)
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
    glasso.log_prob(theta)


    # FIXME: failing internally with Glasso during irmhmc sampling due to singular U in cholesky (MVN , fisher)
    # params_irmhmc_bij = hamiltorch.sample(
    #     log_prob_func=glasso.log_prob, params_init=theta, num_samples=N,
    #     step_size=step_size, num_steps_per_sample=L, sampler=hamiltorch.Sampler.RMHMC,
    #     integrator=hamiltorch.Integrator.IMPLICIT, fixed_point_max_iterations=1000,
    #     fixed_point_threshold=1e-05)


    glasso.reset_parameters()
    theta = hamiltorch.util.flatten(glasso)
    glasso.log_prob(theta)

    N = 200
    step_size = .3
    L = 5

    # samplable, but consistent log_prob nan / -inf
    params_hmc = hamiltorch.sample(
        log_prob_func=glasso.log_prob, params_init=theta, num_samples=N,
        step_size=step_size, num_steps_per_sample=L)

    print()
