import torch
import torch.nn as nn
import torch.distributions as td

from thirdparty_repo.ludwigwinkler.src.MCMC_ProbModel import ProbModel
from Pytorch.Layer.Hidden import Hidden
from Pytorch.Util.ModelUtil import Optim_Model


class Hidden_ProbModel(ProbModel, Optim_Model, Hidden):
    # Inheritance Order is relevant!

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """Instantiate a Hidden model, that is ready to be handled by optimbased
        samplers. all that needs to be done is copy the define_model & reset parameters function"""
        Hidden.__init__(self, no_in, no_out, bias, activation)

    def define_model(self):
        self.tau_w = 1.
        self.dist = {'W': td.MultivariateNormal(
            torch.zeros(self.no_in * self.no_out),
            self.tau_w * torch.eye(self.no_in * self.no_out))}  # todo refactor this to td.Normal()

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))

        if self.bias:
            self.tau_b = 1.
            self.dist['b'] = td.Normal(0., self.tau_b)
            self.b = nn.Parameter(torch.Tensor(self.no_out))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)
        if self.bias:
            nn.init.normal_(self.b)




if __name__ == '__main__':
    no_in = 10
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden_ProbModel(no_in, no_out, bias=True, activation=nn.Identity())
    reg.true_model = reg.vec

    # reg.W = reg.W_.data
    # reg.b = reg.b_.data
    reg.forward(X=torch.ones(100, no_in))
    reg.prior_log_prob()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()


    # log_prob sg mode:
    print(reg.log_prob(X, y))

    # log_prob full mode:
    # reg.closure_log_prob(X, y)
    # print(reg.log_prob())

    # (LUDWIGWINKLER) ----------------------------------------------------------
    from Pytorch.Samplers.LudwigWinkler import LudwigWinkler

    ludi = LudwigWinkler(reg, X, y, batch_size=X.shape[0])

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
