import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Layer.Group_lasso import Group_lasso


class Group_HorseShoe(Hidden):
    prior_log_prob = Group_lasso.prior_log_prob

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        nn.Module.__init__(self)
        self.bias = False  # to meet the Hidden default standard!
        self.no_in = no_in
        self.no_out = no_out
        self.bias = bias
        self.activation = activation

        self.dist = dict()

        self.tau_ = nn.Parameter(torch.Tensor(1))
        self.dist['tau'] = td.HalfCauchy(1.)
        self.tau = self.dist['tau'].sample()

        # Group lasso structure of W
        self.W_ = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W = None
        # self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_in), self.tau)
        # FIXME: check sigma dependence in W_shrinked: \beta_g | tau²_g, sigma² ~ MVN
        self.dist['W'] = td.Normal(torch.zeros(self.no_in * (self.no_out - 1)), 1.)
        self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_in), self.tau)

        # add optional bias
        if bias:
            self.b_ = nn.Parameter(torch.Tensor(self.no_out))
            self.b = None
            self.tau_b = 1.
            self.b = None
            self.dist['b'] = td.Normal(0., self.tau_b)

        self.reset_parameters()
        self.true_model = None

    def update_distributions(self):
        self.dist['W_shrinked'].scale = self.tau

    def reset_parameters(self, seperated=False):

        if seperated:
            self.tau = torch.tensor(0.001)
        else:
            self.tau = self.dist['tau'].sample()

        self.dist['W_shrinked'].scale = self.tau

        # TODO notice, that depending on the application, W can be get / set as
        #  either vec or matrix - which could be handled graciously via getters / setters
        self.W = torch.cat([self.dist['W_shrinked'].sample().view(self.no_in, 1),
                            self.dist['W'].sample().view(self.no_in, self.no_out - 1)],
                           dim=1)

        if self.bias:
            self.b = self.dist['b'].sample()
            self.b_.data = self.b

        self.tau_.data = self.tau
        self.W_.data = self.W

    @property
    def alpha(self):
        """attribute in interval [0,1], which decides upon the degree of how much
        weight gam gets in likelihoods'mu= bnn() + alpha *gam()"""
        raise NotImplementedError('horse\'s alpha is not yet implemented')


if __name__ == '__main__':
    no_in = 2
    no_out = 10

    # single Hidden Unit Example
    ghorse = Group_HorseShoe(no_in, no_out, bias=True, activation=nn.Identity())
    true_model = ghorse.vec

    X_dist = td.Uniform(torch.tensor([-10., -10.]), torch.tensor([10., 10.]))
    X = X_dist.sample(torch.Size([100])).view(100, 2)
    X.detach()
    X.requires_grad_()

    y = ghorse.likelihood(X).sample()

    ghorse.parameters_dict

    ghorse.reset_parameters(seperated=True)
    print(ghorse.parameters_dict)

    ghorse.vec_to_attrs(torch.cat([i * torch.ones(ghorse.__getattribute__(p).nelement())
                                   for i, p in enumerate(ghorse.p_names)]))

    ghorse.parameters_dict
    ghorse(X)

    ghorse.reset_parameters()
    ghorse.parameters_dict
    ghorse(X)

    ghorse

