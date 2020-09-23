import torch
import torch.nn as nn
import torch.distributions as td

from copy import deepcopy

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
        self.dist['alpha'] = td.HalfCauchy(0.3)

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
        self.tau = nn.Parameter(torch.tensor(1.))
        self.dist['tau'] = td.Gamma((self.m + 1) / 2, (self.lamb ** 2) / 2)

        if self.bijected:
            self.dist['lamb'] = td.TransformedDistribution(self.dist['lamb'], LogTransform())
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())

        # Group lasso structure of W
        self.W_shrinked = nn.Parameter(torch.Tensor(1, self.no_out))
        self.W = nn.Parameter(torch.Tensor(self.no_in - 1, self.no_out))

        # FIXME: check sigma dependence in W_shrinked: \beta_g | tau²_g, sigma² ~ MVN
        # FIXME: make W_shrinked ready for multiple groups
        #  make W_shrinked a MVN (0, diag(tau_1, tau_2, ...) - also when updating & joining W
        self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_out), self.tau)  # .clone().detach())
        self.dist['W'] = td.Normal(torch.zeros((self.no_in - 1) * self.no_out), torch.tensor([1.]))

        # add optional bias
        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.dist['b'] = td.Normal(torch.zeros(self.no_out), 1.)

    def forward(self, X):
        XW = X[:, :1] @ self.W_shrinked + X[:, 1:] @ self.W   # self.W_shrinked is top row of complete W mat
        if self.has_bias:
            XW += self.b
        return self.activation(XW)

    def update_distributions(self):
        """due to the hierarchical structure, the distributions parameters must be updated
        Note, that this function is intended to be called immediately after invoking
        log_prob (see Model_util.log_prob) to take the updated parameters (via last
        sampler.step and update the conditional distributions accordingly"""

        # BE WELL AWARE of the required inverse of the bijector in order to update
        # the conditional distributions accordingly
        # print(self.tau)
        # print(self.dist['tau'].transforms[0]._inverse(self.tau.clone().detach()))

        if self.bijected:
            self.dist['tau'].base_dist.rate = \
                self.dist['lamb'].transforms[0]._inverse(self.lamb) ** 2 / 2
            self.dist['W_shrinked'].scale = \
                self.dist['tau'].transforms[0]._inverse(self.tau.clone().detach()) # CAREFULL, THIS is ESSENTIAL for
            # the model to run -and not invoke a second backward (in which already gradients have been lost)
        else:
            self.dist['tau'].rate = self.lamb ** 2 / 2
            self.dist['W_shrinked'].scale = self.tau #.clone().detach()

    def reset_parameters(self, seperated=False):
        """sampling method to instantiate the parameters"""
        self.lamb.data = self.dist['lamb'].sample()
        self.update_distributions()  # to ensure tau's dist is updated properly

        if seperated:  # allows XOR Decision in data generating procecss
            if self.bijected:
                self.tau.data = self.dist['tau'].transforms[0](torch.tensor([0.001]))
            else:
                self.tau.data = torch.tensor([0.001])
        else:
            self.tau.data = self.dist['tau'].sample()

        self.update_distributions()  # to ensure W_shrinked's dist is updated properly

        # partition W according to prior model
        self.W.data = self.dist['W'].sample().view(self.no_in - 1, self.no_out)

        # ASSUMING MERELY ONE VARIABLE TO BE SHRUNKEN
        self.W_shrinked.data = self.dist['W_shrinked'].sample().view(1, self.no_out)

        if self.has_bias:
            self.b.data = self.dist['b'].sample()

        self.init_model = deepcopy(self.state_dict())

    def prior_log_prob(self):
        """evaluate each parameter in respective distrib."""

        value = torch.tensor(0.)
        p_names = ['tau', 'W', 'W_shrinked']
        for name in p_names:  # self.p_names
            value += self.dist[name].log_prob(self.get_param(name)).sum()

        return value

    @property
    def alpha(self):
        """attribute in interval [0,1], which decides upon the degree of how much
        weight gam gets in likelihoods'mu= bnn() + alpha *gam()"""
        # FIXME alpha value for mu in likelihood depending on used shrinkage layer

        # as update_params already changed the tau value here explicitly
        if self.bijected:
            print('tau (unbijected) ', self.tau)
            print('tau (bijected)', self.dist['W_shrinked'].scale)
            # self.dist['W_shrinked'].scale += torch.tensor(1.)
            # tau = self.dist['tau'].transforms[0]._inverse(self.dist['W_shrinked'].scale)
            # print(tau)
            tau = self.dist['W_shrinked'].scale
            print('alpha ', self.dist['alpha'].cdf(self.dist['W_shrinked'].scale))
        else:
            tau = self.dist['W_shrinked'].scale

        # 1- : since small tau indicate high shrinkage & the possibility to
        # estimate using GAM, this means that alpha should be (close to) 1
        return 1 - self.dist['alpha'].cdf(tau)


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
        if self.bijected:
            tau = self.dist['tau'].transforms[0]._inverse(self.dist['W_shrinked'].scale)
        else:
            tau = self.dist['W_shrinked'].scale

        # map tau to [0,1] interval, making it a probability
        # be careful as the mapping is informative prior knowledge!
        # Note further, that alpha is not learned!
        pi = torch.tensor(1.) - self.dist['alpha'].cdf(tau)
        if pi.item() < 0.01:
            pi = torch.tensor([0.01])
        print(pi)
        delta = td.Bernoulli(pi).sample()  # FIXME despite working on its own seems to cause an error in the sampler
        print(delta)
        return delta

    @property
    def alpha_const(self):
        return torch.tensor(1.)

if __name__ == '__main__':
    # generate data
    no_in = 2
    no_out = 1
    n = 1000
    X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)

    glasso = Group_lasso(no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True)
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

    # check "shrinkage_regression" example on being samplable
    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA

    num_samples = 1000

    step_size = 0.01
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    L = 24
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    glasso.reset_parameters()
    sgnht = SGNHT(glasso, X, y, X.shape[0],
                  step_size, num_steps, burn_in,
                  L=L,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)

    import random

    glasso.plot(X, y, chain=random.sample(sgnht.chain, 100),
                **{'title': 'G-lasso'})
