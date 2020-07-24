import torch
import torch.distributions as td
import torch.nn as nn

from Pytorch.Models.BNN import BNN
from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Layer.Group_HorseShoe import Group_HorseShoe
from Pytorch.Layer.Hidden_Probmodel import Hidden_ProbModel
from Pytorch.Layer.Hidden import Hidden
from Pytorch.Layer.Group_lasso_Probmodel import Group_lasso_Probmodel


class ShrinkageBNN(BNN):
    # available shrinkage layers
    shrinkage_type = {
        'glasso': Group_lasso,
        # 'gspike': layer.Group_SpikeNSlab,
        'ghorse': Group_HorseShoe}

    shrinkage_type_ProbM = {
        'glasso': Group_lasso_Probmodel}

    def __init__(self, hunits=[2, 10, 1], activation=nn.ReLU(), final_activation=nn.Identity(),
                 shrinkage='glasso', seperated=False, **gam_param):
        """
        Shrinkage BNN is a regular BNN, but uses a shrinkage layer, which in the
        current implementation shrinks the first variable in the X vector,
        depending on whether or not it helps predictive performance.

        :param hunits: hidden units per layer
        :param activation: nn.ActivationFunction instance
        :param final_activation: nn.ActivationFunction instance
        :param shrinkage: 'glasso', 'gsplike', 'ghorse' each of which specifies the
        grouped shrinkage version of lasso, spike & slab & horseshoe respectively.
        See detailed doc in the respective layer. All of which assume the first variable
        to be shrunken; i.e. provide a prior log prob model on the first column of W
        :param gam_param:
        """

        self.shrinkage = shrinkage
        self.seperated = seperated
        BNN.__init__(self, hunits=[2, 10, 1], activation=nn.ReLU(),
                     final_activation=nn.Identity(), **gam_param)

    def define_model(self):
        # Defining the layers depending on the mode.
        if isinstance(self, Vec_Model):
            L = Hidden
            S = self.shrinkage_type[self.shrinkage]
        elif isinstance(self, Optim_Model):
            L = Hidden_ProbModel
            S = self.shrinkage_type_ProbM[self.shrinkage]

        self.layers = nn.Sequential(
            S(self.hunits[0], self.hunits[1], True, self.activation),
            *[L(no_in, no_units, True, self.activation)
              for no_in, no_units in zip(self.hunits[1:-2], self.hunits[2:-1])],
            L(self.hunits[-2], self.hunits[-1], bias=False, activation=self.final_activation))

        if self.heteroscedast:
            self.sigma_ = nn.Parameter(torch.Tensor(1))
            self.dist_sigma = td.TransformedDistribution(td.Gamma(0.01, 0.01), td.ExpTransform())
            self.sigma = self.dist_sigma.sample()
        else:
            self.sigma = torch.tensor(1.)


from Pytorch.Models.ModelUtil import Vec_Model, Model_util, Optim_Model
from thirdparty_repo.ludwigwinkler.src.MCMC_ProbModel import ProbModel


class ShrinkageBNN_VEC(ShrinkageBNN, nn.Module, Vec_Model, Model_util):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        ShrinkageBNN.__init__(self, *args, **kwargs)


class ShrinkageBNN_OPTIM(ShrinkageBNN, ProbModel, Optim_Model, nn.Module, Model_util):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        ShrinkageBNN.__init__(self, *args, **kwargs)


if __name__ == '__main__':
    from hamiltorch.util import flatten, unflatten

    sbnn = ShrinkageBNN_VEC()

    # sampling with effect
    sbnn.reset_parameters(seperated=True)
    sbnn.reset_parameters(seperated=False)

    X_dist = td.Uniform(torch.tensor([-10., -10]), torch.tensor([10., 10]))
    X = X_dist.sample(torch.Size([100]))
    y = sbnn.likelihood(X).sample()

    sbnn.closure_log_prob(X, y)
    sbnn.log_prob(flatten(sbnn))

    import hamiltorch
    import hamiltorch.util

    init_theta = flatten(sbnn)

    # HMC NUTS
    N = 200
    step_size = .3
    L = 5
    burn = 500
    N_nuts = burn + N
    params_hmc_nuts = hamiltorch.sample(
        log_prob_func=sbnn.log_prob, params_init=init_theta,
        num_samples=N_nuts, step_size=step_size, num_steps_per_sample=L,
        sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
        desired_accept_rate=0.8)

    # (Optim model) ------------------------------------------------------------
    from Pytorch.Samplers.LudwigWinkler import LudwigWinkler

    optim = ShrinkageBNN_OPTIM()
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
