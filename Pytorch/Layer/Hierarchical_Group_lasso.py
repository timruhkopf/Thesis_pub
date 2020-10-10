import torch
import torch.nn as nn
import torch.distributions as td

from copy import deepcopy

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Util.Util_Distribution import LogTransform


class Hierarchical_Group_lasso(Hidden):

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
        self.lamb = nn.Parameter(torch.tensor([0.1]))
        self.lamb.dist = td.HalfCauchy(scale=torch.tensor([1.]))

        # hyperparam of W: single variance parameter for group
        self.tau = nn.Parameter(torch.ones(1, self.no_in))
        self.tau.dist = td.Gamma(torch.ones(1, self.no_in) * (self.m + 1) / 2, (self.lamb ** 2) / 2)

        if self.bijected:
            self.lamb.dist = td.TransformedDistribution(self.lamb.dist, LogTransform())
            self.tau.dist = td.TransformedDistribution(self.tau.dist, LogTransform())

        # Group lasso structure of W
        self.W_shrinked = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W_shrinked.dist = td.Normal(torch.zeros(self.no_out, self.no_in), self.tau ** 2)

        # add optional bias
        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.dist = td.Normal(torch.zeros(self.no_out), 1.)

    def forward(self, X):
        XW = X @ self.W_shrinked
        if self.has_bias:
            XW += self.b
        return self.activation(XW)

    def update_distributions(self):
        """due to the hierarchical structure, the distributions parameters must be updated
        Note, that this function is intended to be called immediately after invoking
        log_prob (see Model_util.log_prob) to take the updated parameters (via last
        sampler.step and update the conditional distributions accordingly"""
        if self.bijected:
            self.tau.dist.base_dist.rate = \
                torch.ones_like(self.tau.dist.base_dist.rate) * \
                self.lamb.dist.transforms[0]._inverse(self.lamb) ** 2 / 2
            # self.W_shrinked.dist.covariance_matrix = \
            #     torch.diag(self.tau.dist.transforms[0]._inverse(self.tau) ** 2)
            self.W_shrinked.dist.scale = self.tau.dist.transforms[0]._inverse(self.tau)
        else:
            self.tau.dist.rate = torch.ones_like(self.tau.dist.rate) * self.lamb ** 2 / 2
            self.W_shrinked.dist.scale = self.tau  # .clone().detach()

    def reset_parameters(self, seperated=False):
        for name, p in self.named_parameters():
            if name.endswith('W_shrinked'):
                p.data = p.dist.sample().t()
                continue

            p.data = p.dist.sample()
            if seperated and name.endswith('tau'):
                # enforce the first column in W to be "zero"
                if self.bijected:
                    p.data[0][0] = -5.  # is a tau of 6.7379e-03
                else:
                    p.data[0][0] = 0.001

            self.update_distributions()

    def prior_log_prob(self):
        value = torch.tensor(0.)
        for name, p in self.named_parameters():
            if name.endswith('W_shrinked'):
                value += self.W_shrinked.dist.log_prob(self.W_shrinked.t()).sum()
                continue

            value += p.dist.log_prob(p.data).sum()

        return value

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
    #         tau = self.dist['tau'].transforms[0]._inverse(self.dist['W_shrinked'].scale)
    #     else:
    #         tau = self.dist['W_shrinked'].scale

    # map tau to [0,1] interval, making it a probability
    # be careful as the mapping is informative prior knowledge!
    # Note further, that alpha is not learned!
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

    no_in = 2

    no_out = 1
    n = 1000
    n_val = 200
    X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)
    X_val = X_dist.sample(torch.Size([n_val])).view(n_val, no_in)

    glasso = Hierarchical_Group_lasso(no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True)
    glasso.reset_parameters(seperated=True)
    glasso.true_model = deepcopy(glasso.state_dict())

    y = glasso.likelihood(X).sample()

    # glasso.reset_parameters()
    # glasso.init_model = deepcopy(glasso.state_dict())
    # print(glasso.prior_log_prob())

    print(glasso.true_model)
    glasso.plot(X[:400], y[:400])

    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 1000, 2000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # glasso.reset_parameters(False)
    # glasso.plot(X_joint, y)

    # torch.autograd.set_detect_anomaly(False)

    import pickle
    import random
    import matplotlib

    # matplotlib.use('TkAgg')
    matplotlib.use('Agg')

    path = '/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/manualGlasso/'
    counter = 0
    glasso.check_chain
    while True:
        counter += 1
        basename = path + 'Hierarchical_glasso{}'.format(str(counter))
        try:

            # GENERATE NEW DATA & INIT --------------
            # glasso.load_state_dict(glasso.init_model)
            # glasso.load_state_dict(OrderedDict([('lamb', torch.tensor([0.5282])), ('tau', torch.tensor([1.,1.])),
            #              ('W_shrinked', torch.tensor([1., 1.])), ('b', torch.tensor([0.]))]))
            glasso.reset_parameters(True)
            glasso.true_model = deepcopy(glasso.state_dict())
            y = glasso.likelihood(X).sample()
            y_val = glasso.likelihood(X_val).sample()

            # glasso.plot(X_val, y_val)

            glasso.reset_parameters()
            glasso.init_model = deepcopy(glasso.state_dict())

            # BURN SAMPLING --------------------
            burn_in, n_samples = 1000, 1000
            sampler = Sampler(glasso, epsilon=0.001, L=2)
            sampler.sample(trainloader, burn_in, n_samples)

            print(sampler.chain)
            print('state: ', glasso.state_dict())
            print('true_model: ', glasso.true_model)

            # CONTINUE SAMPLING ------------------
            glasso.plot(X_val, y_val, chain=random.sample(sampler.chain, 30),
                        **{'title': 'G-lasso'})
            glasso.plot(X_val, y_val, chain=sampler.chain[-30:])

            manual_switch = False
            if manual_switch:
                continue

            burn_in, n_samples = 10000, 10000
            sampler = Sampler(glasso, epsilon=0.001, L=2)
            sampler.sample(trainloader, burn_in, n_samples)

            # STORE RESULTS --------------
            sampler.save(basename)
            glasso.plot(X_val, y_val, chain=random.sample(sampler.chain, 30),
                        **{'title': 'G-lasso'}, path=basename)
            glasso.plot(X_val, y_val, chain=sampler.chain[-30:],
                        **{'title': 'G-lasso'}, path=basename)

            sampler.traceplots(path=basename + '_traces_baseline.pdf')
            sampler.traceplots(path=basename + '_traces.pdf', baseline=False)

            config = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU(), bijected=True, seperated=True)
            config['true_model'] = glasso.true_model
            config['init_model'] = glasso.init_model
            with open(basename + '_config.pkl', 'wb') as handle:
                pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(basename + '_chain.pkl', 'wb') as handle:
                pickle.dump(sampler.chain, handle, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:

            print(sampler.chain)
            print(glasso.state_dict())
            print(e)
            continue

    # generate data
    no_in = 2
    no_out = 1
    n = 1000
    X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)

    glasso = Hierarchical_Group_lasso(no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True)
    glasso.reset_parameters(seperated=True)  # seperated=False
    glasso.true_model = deepcopy(glasso.state_dict())
    y = glasso.likelihood(X).sample()

    # check reset_parameters &  check prior_log_prob
    glasso.reset_parameters()  # seperated=False
    glasso.init_model = deepcopy(glasso.state_dict())

    print(glasso.true_model)
    print(glasso.init_model)

    glasso.reset_parameters(False)
    print(glasso.state_dict())
    print(glasso.W_shrinked.dist.scale)

    glasso.plot(X[:300], y[:300], **{'title': 'G-lasso'})
    print(glasso.prior_log_prob())

    # check update_distributions
    # glasso.reset_parameters()
    # print(glasso.W[:, 0])
    # print(glasso.dist['W_shrinked'].scale)
    # # value of log prob changed due to change in variance
    # print(glasso.dist['W_shrinked'].log_prob(0.))
    # print(glasso.dist['W_shrinked'].cdf(0.))  # location still 0

    # check alpha value
    # glasso.alpha

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter()
    # writer.add_graph(glasso, input_to_model=X, verbose=True) # FAILS unexpectedly
    # writer.close()

    # check sampling ability.
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader
    import random

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # glasso.reset_parameters(False)
    # glasso.plot(X_joint, y)

    # torch.autograd.set_detect_anomaly(True)
    while True:
        try:
            glasso.reset_parameters()
            glasso.init_model = deepcopy(glasso.state_dict())
            sampler = Sampler(glasso, epsilon=0.001, L=2)
            sampler.sample(trainloader, burn_in, n_samples)

            glasso.plot(X[:100], y[:100], chain=random.sample(sampler.chain, 30),
                        **{'title': 'G-lasso'})

            glasso.plot(X[:100], y[:100], chain=sampler.chain[-30:],
                        **{'title': 'G-lasso'})

        except Exception as e:
            print(e)
            continue
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
