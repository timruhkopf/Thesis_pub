import torch
import torch.nn as nn
import torch.distributions as td

from hamiltorch.util import flatten
from itertools import accumulate
import inspect

from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel
from Pytorch.Layer.Hidden import Hidden
from Pytorch.Util.ModelUtil import Vec_Model, Model_util, Optim_Model
from thirdparty_repo.ludwigwinkler.src.MCMC_ProbModel import ProbModel


class BNN(nn.Module, Model_util):
    def __init__(self, hunits=[1, 10, 5, 1], activation=nn.ReLU(), final_activation=nn.Identity(), heteroscedast=False):
        """
        Bayesian Neural Network, consisting of hidden layers.
        :param hunits: list of integers, specifying the input dimensions of hidden units
        :param activation: each layers activation function (except the last)
        :param final_activation: activation function of the final layer.
        :param heteroscedast: bool: indicating, whether or not y's conditional
        variance y|x~N(mu, sigma²) is to be estimated as well
        :remark: see Hidden.activation doc for available activation functions
        """
        nn.Module.__init__(self)
        self.heteroscedast = heteroscedast
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation

        self.define_model()
        self.reset_parameters()

        self.true_model = None

    # CLASSICS METHODS ---------------------------------------------------------
    def define_model(self):
        # Defining the layers depending on the mode.
        if isinstance(self, Vec_Model):
            L = Hidden
        elif isinstance(self, Optim_Model):
            L = Hidden_ProbModel
        else:
            L = Hidden

        self.layers = nn.Sequential(
            *[L(no_in, no_units, True, self.activation)
              for no_in, no_units in zip(self.hunits[:-2], self.hunits[1:-1])],
            L(self.hunits[-2], self.hunits[-1], bias=False, activation=self.final_activation))

        if self.heteroscedast:
            self.sigma_ = nn.Parameter(torch.Tensor(1))
            self.dist_sigma = td.TransformedDistribution(td.Gamma(0.01, 0.01), td.ExpTransform())
            self.sigma = self.dist_sigma.sample()
        else:
            self.sigma = torch.tensor(1.)

    def reset_parameters(self, seperated=False):
        """samples each layer individually.
        :param seperated: bool. indicates whether the first layer (given its
        local reset_parameters function has a seperated argument) will sample
        the first layers 'shrinkage' to create an effect for a variable, that is indeed
        not relevant to the BNNs prediction."""
        inspected = inspect.getfullargspec(self.layers[0].reset_parameters).args
        if 'seperated' in inspected:
            self.layers[0].reset_parameters(seperated)
        else:
            # default case: all Hidden units, no shrinkage -- has no seperated attr.
            self.layers[0].reset_parameters()

        for h in self.layers[1:]:
            h.reset_parameters()

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        p_log_prob = sum([h.prior_log_prob().sum() for h in self.layers])

        if self.heteroscedast:
            p_log_prob += self.dist_sigma.log_prob(self.sigma)

        return p_log_prob

    def likelihood(self, X):
        """:returns the conditional distribution of y | X"""
        return td.Normal(self.forward(X), scale=self.sigma)

    # SURROGATE (AGGREGATING) METHODS ------------------------------------------
    @property
    def vec(self):
        return torch.cat([h.vec for h in self.layers])

    @property
    def parameters_list(self):
        return [h.parameters_list for h in self.layers]

    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)

    def update_distributions(self):
        for h in self.layers:
            h.update_distributions()

    def vec_to_attrs(self, vec):
        """surrogate for Hidden Layers' vec_to_attrs, but actually needs to sett a
        BNN.attribute in case of e.g. heteroscedasticity,
        i.e. where sigma param is in likelihood"""

        # delegate the vector parts to the layers
        lengths = list(accumulate([0] + [h.n_params for h in self.layers]))
        for i, j, h in zip(lengths, lengths[1:], self.layers):
            h.vec_to_attrs(vec[i:j])

        if self.heteroscedast:
            self.__setattr__('sigma', vec[-1])



class BNN_VEC(BNN, Vec_Model):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        BNN.__init__(self, *args, **kwargs)


class BNN_OPTIM(BNN, ProbModel, Optim_Model):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        BNN.__init__(self, *args, **kwargs)


if __name__ == '__main__':
    # bnn = BNN(hunits=[1, 10, 5, 1])
    #
    # # generate data
    # X_dist = td.Uniform(torch.tensor(-10.), torch.tensor(10.))
    # X = X_dist.sample(torch.Size([100])).view(100, 1)
    # y = bnn.likelihood(X).sample()
    #
    # # check forward path
    # bnn.layers(X)
    # bnn.forward(X)
    #
    # bnn.reset_parameters()
    # bnn.true_model = bnn.vec
    #
    # # check vec_to_attrs
    # bnn.vec_to_attrs(torch.cat([i * torch.ones(h.n_params) for i, h in enumerate(bnn.layers)]))
    # bnn.parameters_list
    # bnn.vec_to_attrs(torch.ones(80))
    # bnn.forward(X)
    #
    # # check accumulation of parameters & parsing
    # bnn.log_prob(X, y, flatten(bnn))

    # (VEC MODEL & Data generation) --------------------------------------------
    vec = BNN_VEC(hunits=[1, 10, 5, 1], activation=nn.ReLU(), final_activation=nn.Identity(), heteroscedast=False)
    X_dist = td.Uniform(torch.tensor(-10.), torch.tensor(10.))
    X = X_dist.sample(torch.Size([100])).view(100, 1)
    y = vec.likelihood(X).sample()


    vec.closure_log_prob(X, y)
    vec.log_prob(flatten(vec))

    import hamiltorch
    import hamiltorch.util

    init_theta = flatten(vec)

    # HMC NUTS
    N = 200
    step_size = .3
    L = 5
    burn = 500
    N_nuts = burn + N
    params_hmc_nuts = hamiltorch.sample(
        log_prob_func=vec.log_prob, params_init=init_theta,
        num_samples=N_nuts, step_size=step_size, num_steps_per_sample=L,
        sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
        desired_accept_rate=0.8)

    # (Optim model) ------------------------------------------------------------
    from Pytorch.Samplers.LudwigWinkler import LudwigWinkler

    optim = BNN_OPTIM()
    ludi = LudwigWinkler(optim, X, y, batch_size=X.shape[0])

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

    print()
