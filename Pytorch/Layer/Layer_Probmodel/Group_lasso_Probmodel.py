import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel
from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Util.ModelUtil import Optim_Model
from Pytorch.Util.DistributionUtil import LogTransform


class Group_lasso_Probmodel(Optim_Model, Group_lasso):
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
        Hidden_ProbModel.__init__(self, no_in, no_out, bias, activation)
        self.dist['alpha'] = td.LogNormal(0., scale=1. / 20.)
        self.alpha_chain = list()

    def define_model(self):
        self.m = self.no_out  # single "group size" to be penalized

        # hyperparam of tau
        self.lamb = nn.Parameter(torch.tensor(1.))
        self.dist['lamb'] = td.HalfCauchy(scale=1.)

        # hyperparam of W: single variance parameter for group
        self.tau = nn.Parameter(torch.tensor(1.))

        self.dist['tau'] = td.Gamma((self.m + 1) / 2, (self.lamb ** 2) / 2)

        if self.bijected:
            self.dist['lamb'] = td.TransformedDistribution(self.dist['lamb'], LogTransform())
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())

        # Group lasso structure of W
        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_in), self.tau)
        # FIXME: check sigma dependence in W_shrinked: \beta_g | tau²_g, sigma² ~ MVN
        self.dist['W'] = td.Normal(torch.zeros(self.no_in * (self.no_out - 1)), 1.)

        # add optional bias
        if self.bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.tau_b = 1.
            self.dist['b'] = td.Normal(0., self.tau_b)

    def reset_parameters(self, seperated=False):
        """sampling method to instantiate the parameters"""

        self.lamb.data = self.dist['lamb'].sample()
        self.update_distributions()  # to ensure tau's dist is updated properly

        if seperated:  # allows XOR Decision in data generating procecss
            if self.bijected:
                self.tau.data = self.dist['tau'].transforms[0](torch.tensor(0.001))
            else:
                self.tau.data = torch.tensor(0.001)
            self.update_distributions()  # to ensure W_shrinked's dist is updated properly
        else:
            self.tau.data = self.dist['tau'].sample()
            self.update_distributions()  # to ensure W_shrinked's dist is updated properly

        # partition W according to prior model
        self.W.data = torch.cat(
            [self.dist['W_shrinked'].sample().view(self.no_in, 1),
             self.dist['W'].sample().view(self.no_in, self.no_out - 1)],
            dim=1)

        self.b.data = self.dist['b'].sample()


if __name__ == '__main__':
    glasso_P = Group_lasso_Probmodel(3, 1)
    glasso_P.true_model = glasso_P.vec

    # generate data
    X_dist = td.Uniform(torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.]))
    X = X_dist.sample(torch.Size([100])).view(100, 3)
    y = glasso_P.likelihood(X).sample()

    from Pytorch.Samplers.LudwigWinkler import LudwigWinkler
    ludi = LudwigWinkler(glasso_P, X, y, batch_size=X.shape[0])

    num_samples = 200
    sampler = 'sgnht'
    step_size = 0.1
    num_steps = 100
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    hmc_traj_length = 20
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    ludi.sample_SGNHT(step_size, num_steps, burn_in, pretrain=False, tune=tune, hmc_traj_length=hmc_traj_length,
                      num_chains=num_chains)

    type(ludi.sampler.chain)
    ludi.sampler.chain.__dict__


