import torch
import torch.distributions as td

from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Util.DistributionUtil import LogTransform

from geoopt.manifolds import Euclidean,  \
    ProductManifold, StereographicExact, \
    EuclideanStiefel, CanonicalStiefel, \
    Scaled,  SphereProjection, EuclideanStiefelExact

from geoopt import ManifoldParameter


class Manifold_Group_lasso(Group_lasso):
    manifold = {'Euclidean': Euclidean,
                'EuclideanStiefelExact':EuclideanStiefelExact,
                'ProductManifold': ProductManifold,
                'EuclideanStiefel': EuclideanStiefel, 'CanonicalStiefel': CanonicalStiefel,
                   # 'Scaled': Scaled, seems to scale a Manifold() instance
                'SphereProjection': SphereProjection}

    def __init__(self, manifold='Euclidean', *args, **kwargs):
        self.Manifold = self.manifold[manifold]
        super(Manifold_Group_lasso, self).__init__(*args, **kwargs)

    def define_model(self):
        self.m = self.no_out  # single "group size" to be penalized

        # hyperparam of tau
        # FIXME: when lamb is a nn.Parameter, the model fails - only in
        #   a model with a glroup lasso layer AND a GAM.
        #   making it non learnable, the model runs (if W_shrinked is also not part of prior_log_prob)
        #   self.lamb = nn.Parameter(torch.tensor([1.]))
        self.lamb = torch.tensor([0.1])  # FIXME: check how this dist looks like
        self.dist['lamb'] = td.HalfCauchy(scale=torch.tensor([1.]))

        # hyperparam of W: single variance parameter for group
        self.tau = ManifoldParameter(
            torch.ones(1), manifold=self.Manifold()
        )
        self.dist['tau'] = td.Gamma((self.m + 1) / 2, (self.lamb ** 2) / 2)

        if self.bijected:
            self.dist['lamb'] = td.TransformedDistribution(self.dist['lamb'], LogTransform())
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())

        # Group lasso structure of W
        self.W_shrinked = ManifoldParameter(
            torch.Tensor(1, self.no_out), manifold=self.Manifold())

        self.W = ManifoldParameter(
            torch.Tensor(self.no_in - 1, self.no_out), manifold=self.Manifold())

        # FIXME: check sigma dependence in W_shrinked: \beta_g | tau²_g, sigma² ~ MVN
        # FIXME: make W_shrinked ready for multiple groups
        #  make W_shrinked a MVN (0, diag(tau_1, tau_2, ...) - also when updating & joining W
        self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_out), self.tau)  # .clone().detach())
        self.dist['W'] = td.Normal(torch.zeros((self.no_in - 1) * self.no_out), torch.tensor([1.]))

        # add optional bias
        if self.has_bias:
            self.b = ManifoldParameter(
            torch.Tensor(self.no_out), manifold=self.Manifold())
            self.dist['b'] = td.Normal(torch.zeros(self.no_out), 1.)


if __name__ == '__main__':
    import torch.nn as nn

    # generate data
    no_in = 2
    no_out = 1
    n = 1000
    X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)

    # glasso = Manifold_Group_lasso(no_in=no_in, no_out=no_out, bias=True, manifold='Euclidean', activation=nn.ReLU(),
    #                               bijected=False)
    glasso = Manifold_Group_lasso(no_in=no_in, no_out=no_out, bias=True, manifold='PoincareBall', activation=nn.ReLU(),
                                  bijected=False)
    glasso.reset_parameters(seperated=False)
    glasso.true_model = glasso.state_dict()
    y = glasso.likelihood(X).sample()

    # check reset_parameters &  check prior_log_prob
    glasso.reset_parameters(seperated=True)
    glasso.plot(X, y, **{'title': 'G-lasso'})
    print(glasso.prior_log_prob())

    # check update_distributions
    # glasso.reset_parameters()
    print(glasso.W[:, 0])
    print(glasso.dist['W_shrinked'].scale)
    # value of log prob changed due to change in variance
    print(glasso.dist['W_shrinked'].log_prob(0.))
    print(glasso.dist['W_shrinked'].cdf(0.))  # location still 0

    # check alpha value
    # glasso.alpha

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter()
    # writer.add_graph(glasso, input_to_model=X, verbose=True) # FAILS unexpectedly
    # writer.close()

    # check sampling ability.
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # glasso.reset_parameters(False)
    # glasso.plot(X_joint, y)

    torch.autograd.set_detect_anomaly(True)
    sampler = Sampler(glasso, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    import random

    glasso.plot(X, y, chain=random.sample(sampler.chain, 30),
                **{'title': 'G-lasso'})

    glasso.plot(X, y, chain=sampler.chain[-30:],
                **{'title': 'G-lasso'})
