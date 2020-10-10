import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Util.Util_Distribution import LogTransform
from copy import deepcopy


class Hierarchical_Group_HorseShoe(Hidden, ):
    prior_log_prob = Group_lasso.prior_log_prob

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True, seperated=True):
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
        self.dist['alpha'] = td.HalfCauchy(0.3)
        self.reset_parameters()

    def define_model(self):
        self.lamb = nn.Parameter(torch.tensor([0.9]))
        self.lamb.dist = td.HalfCauchy(scale=torch.tensor([1.]))

        # hyperparam of W: single variance parameter for group
        self.tau = nn.Parameter(torch.ones(self.no_in))
        self.tau.dist = td.HalfCauchy(torch.ones(self.no_in))
        self.tau.data = self.tau.dist.sample()

        if self.bijected:
            self.lamb.dist = td.TransformedDistribution(self.lamb.dist, LogTransform())
            self.tau.dist = td.TransformedDistribution(self.tau.dist, LogTransform())

        # Group lasso structure of W
        self.W_shrinked = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W_shrinked.dist = td.Normal(torch.zeros(self.no_out, self.no_in), self.tau ** 2 * self.lamb)

        # add optional bias
        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
        self.b.dist = td.Normal(torch.zeros(self.no_out), 1.)

    def forward(self, X):
        XW = X @ self.W_shrinked  # necessary due to  td.MultivariateNormal's covaraince expansion
        if self.has_bias:
            XW += self.b
        return self.activation(XW)

    def update_distributions(self):
        if self.bijected:
            self.W_shrinked.dist.scale = \
                self.tau.dist.transforms[0]._inverse(self.tau) ** 2 * \
                self.lamb.dist.transforms[0]._inverse(self.lamb)
        else:
            self.W_shrinked.dist.scale = self.tau ** 2 * self.lamb

    def reset_parameters(self, seperated=False):
        for name, p in self.named_parameters():

            if name.endswith('W_shrinked'):
                p.data = p.dist.sample().t()
                continue

            p.data = p.dist.sample()
            if seperated and name.endswith('tau'):
                # enforce the first column in W to be "zero"
                if self.bijected:
                    p.data[0] = -5.  # is a tau of 6.7379e-03
                else:
                    p.data[0] = 0.001

            self.update_distributions()
        self.init_model = deepcopy(self.state_dict())

    def prior_log_prob(self):

        value = torch.tensor(0.)
        for name, p in self.named_parameters():

            if name.endswith('W_shrinked'):
                value += p.dist.log_prob(p.data.t()).sum()
                continue

            value += p.dist.log_prob(p.data).sum()

        return value


if __name__ == '__main__':

    from copy import deepcopy

    no_in = 2

    no_out = 1
    n = 1000
    X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)

    horse = Hierarchical_Group_HorseShoe(no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True)
    horse.reset_parameters(seperated=False)
    horse.true_model = deepcopy(horse.state_dict())
    y = horse.likelihood(X).sample()

    # horse.reset_parameters()
    # horse.init_model = deepcopy(horse.state_dict())
    # print(horse.prior_log_prob())

    print(horse.true_model)
    horse.plot(X[:400], y[:400])

    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 1000, 2000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # horse.reset_parameters(False)
    # horse.plot(X_joint, y)

    # torch.autograd.set_detect_anomaly(False)
    from collections import OrderedDict

    while True:
        try:
            # horse.load_state_dict(horse.init_model)
            # horse.load_state_dict(OrderedDict([('lamb', torch.tensor([0.5282])), ('tau', torch.tensor([1.,1.])),
            #              ('W_shrinked', torch.tensor([1., 1.])), ('b', torch.tensor([0.]))]))
            horse.reset_parameters()
            horse.init_model = deepcopy(horse.state_dict())
            sampler = Sampler(horse, epsilon=0.003, L=2)
            sampler.sample(trainloader, burn_in, n_samples)

            import random

            print(horse.state_dict())
            print(horse.true_model)
            horse.plot(X[:100], y[:100], chain=random.sample(sampler.chain, 30),
                       **{'title': 'G-Horse'})

            horse.plot(X[:100], y[:100], chain=sampler.chain[-30:],
                       **{'title': 'G-Horse'})
        except Exception as e:

            print(sampler.chain)
            print(horse.state_dict())
            print(e)

    # sampler.model.check_chain(sampler.chain)

    print()
    while True:
        no_in = 2
        no_out = 1
        n = 1000
        X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, no_in)

        horse = Group_HorseShoe(no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True)
        horse.reset_parameters(seperated=True)
        horse.true_model = deepcopy(horse.state_dict())
        y = horse.likelihood(X).sample()

        # check reset_parameters &  check prior_log_prob
        horse.reset_parameters(seperated=True)
        # horse.plot(X, y, **{'title': 'G-lasso'})
        print(horse.prior_log_prob())

        # check update_distributions
        # horse.reset_parameters()
        print(horse.W[:, 0])
        print(horse.dist['W_shrinked'].scale)
        # value of log prob changed due to change in variance
        # print(horse.dist['W_shrinked'].log_prob(0.))
        # print(horse.dist['W_shrinked'].cdf(0.))  # location still 0

        # check alpha value
        # horse.alpha

        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter()
        # writer.add_graph(horse, input_to_model=X, verbose=True) # FAILS unexpectedly
        # writer.close()

        # check sampling ability.
        from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
        from torch.utils.data import TensorDataset, DataLoader

        burn_in, n_samples = 2000, 2000

        trainset = TensorDataset(X, y)
        trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

        Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                   'SGRLD': myRSGLD,  # epsilon
                   'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                   }['RHMC']

        # horse.reset_parameters(False)
        # horse.plot(X_joint, y)

        torch.autograd.set_detect_anomaly(False)
        sampler = Sampler(horse, epsilon=0.001, L=2)
        try:
            sampler.sample(trainloader, burn_in, n_samples)
            sampler.model.check_chain(sampler.chain)
        except:
            print()
            continue

        import random

        horse.plot(X, y, chain=random.sample(sampler.chain, 30),
                   **{'title': 'G-Horse'})

        horse.plot(X, y, chain=sampler.chain[-30:],
                   **{'title': 'G-Horse'})
