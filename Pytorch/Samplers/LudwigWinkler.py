import torch
from Pytorch.Samplers.Samplers import Sampler
from thirdparty_repo.ludwigwinkler.src.MCMC_Sampler import SGLD_Sampler, MALA_Sampler, SGNHT_Sampler
from torch.utils.data import TensorDataset, DataLoader
from copy import copy, deepcopy


class LudwigWinkler(Sampler):
    def __init__(self, model, X, y, batch_size=None):
        """interface to https://github.com/ludwigwinkler/
        BE AWARE, that the entire repo is build on Pytorch.Optim, which require
        that each parameter used and optimized is a nn.Parameter!"""
        self.model = copy(model)
        self.data = X
        self.target = y

        if batch_size is not None:
            batch_size = self.data.shape[0]

        self.model.dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=batch_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.log_prob_ = copy(self.model.log_prob)
        self.model.log_prob = lambda X, y: {'log_prob': self.model.log_prob_(X, y)}

        if torch.cuda.is_available():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            FloatTensor = torch.cuda.FloatTensor
            Tensor = torch.cuda.FloatTensorf
        else:
            device = torch.device('cpu')
            FloatTensor = torch.FloatTensor
            Tensor = torch.FloatTensor

    def convert_chain(self):
        vectorize = lambda D: torch.cat([a.view(a.nelement()) for a in D.values()])
        return [vectorize(c) for c in self.sampler.chain.samples]

    def sample(self):
        self.sampler.sample_chains()
        self.chain = self.convert_chain()


class MALA(LudwigWinkler):
    def __init__(self, model, X, y, batch_size, step_size, num_steps, burn_in, pretrain, tune, num_chains):
        LudwigWinkler.__init__(self, model, X, y, batch_size)
        self.sampler = MALA_Sampler(
            probmodel=self.model,
            step_size=step_size,
            num_steps=num_steps,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune,
            num_chains=num_chains)


class SGNHT(LudwigWinkler):
    def __init__(self, model, X, y, batch_size, step_size, num_steps, burn_in,
                 pretrain, tune, hmc_traj_length, num_chains):
        LudwigWinkler.__init__(self, model, X, y, batch_size)
        self.sampler = SGNHT_Sampler(
            probmodel=self.model,
            step_size=step_size,
            num_steps=num_steps,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune,
            traj_length=hmc_traj_length,
            num_chains=num_chains)


class SGLD(LudwigWinkler):
    def __init__(self, model, X, y, batch_size, step_size, num_steps, burn_in, pretrain, tune, num_chains=7):
        LudwigWinkler.__init__(self, model, X, y, batch_size)
        self.sampler = SGLD_Sampler(
            probmodel=self.model,
            step_size=step_size,
            num_steps=num_steps,
            num_chains=num_chains,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune)


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.distributions as td
    from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel

    no_in = 1
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden_ProbModel(no_in, no_out, bias=False, activation=nn.ReLU())

    # reg.W = reg.W_.data
    # reg.b = reg.b_.data
    reg.forward(X=torch.ones(100, no_in))
    reg.prior_log_prob()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()

    reg.true_model = deepcopy(reg.state_dict())
    reg.reset_parameters()

    print(reg.true_model)
    print(reg.state_dict())

    # log_prob sg mode:
    print(reg.log_prob(X, y))

    num_samples = 1000

    step_size = 0.01
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    hmc_traj_length = 5
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    reg.reset_parameters()
    sgnht = SGNHT(reg, X, y, X.shape[0],
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  hmc_traj_length=hmc_traj_length,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)

    sgnht.model.plot(X, y)  # function still error prone: multiple executions seem to change the plot

    from _collections import OrderedDict

    sgnht.model.plot1d(X, y, true_model=sgnht.model.true_model, param=
    [OrderedDict({'W': c.view((1,1))}) for i, c in enumerate(sgnht.chain) if     i % 100 == 0])

    mala = MALA(reg, X, y, X.shape[0],
                step_size, num_steps, burn_in, pretrain=True, tune=tune,
                num_chains=num_chains)
    mala.sample()
    print(mala.chain)
    mala.model.plot(X, y)

    sgld = SGLD(reg, X, y, X.shape[0],
                step_size=0.0005, num_steps=5000, burn_in=0, pretrain=False, tune=tune,
                num_chains=num_chains)
    sgld.sample()
    print(sgld.chain)
    sgld.model.plot(X, y)
