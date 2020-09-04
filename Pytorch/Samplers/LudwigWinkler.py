from Pytorch.Samplers.Samplers import Sampler
from thirdparty_repo.ludwigwinkler.src.MCMC_Sampler import HMC_Sampler, SGLD_Sampler, MALA_Sampler, SGNHT_Sampler
from torch.utils.data import TensorDataset, DataLoader
from copy import copy, deepcopy
import torch


class LudwigWinkler(Sampler):
    def __init__(self, model,trainloader):
        """interface to https://github.com/ludwigwinkler/
        BE AWARE, that the entire repo is build on Pytorch.Optim, which require
        that each parameter used and optimized is a nn.Parameter!"""
        self.model = copy(model)

        if 'SG' not in str(self) and trainloader.batch_size != len(trainloader.dataset):
            raise ValueError('trainloader for non-SG Sampler must use the entire dataset at each step'
                             ' set trainloader.batch_size = len(trainloader.dataset')

        self.model.dataloader = trainloader
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.log_prob_ = copy(self.model.log_prob)
        self.model.log_prob = lambda *data: {'log_prob': self.model.log_prob_(*data)}

        if torch.cuda.is_available():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            FloatTensor = torch.cuda.FloatTensor
            Tensor = torch.cuda.FloatTensor
        else:
            device = torch.device('cpu')
            FloatTensor = torch.FloatTensor
            Tensor = torch.FloatTensor

    def sample(self):
        """

        :return: list of state_dicts (OrderedDicts) representing each state of the model
        """

        self.sampler.sample_chains()
        self.chain = self.sampler.chain.samples
        self.log_probs = self.sampler.chain.log_probs

        # check sampler did something meaningfull.
        if len(self.chain) == 1:
            print(self.chain)
            raise ValueError('The chain did not progress beyond first step')

        if any([torch.any(torch.isnan(v)) if v.nelement() != 1 else torch.isnan(v)
                for chain in (self.chain[-1].values(), self.chain[0].values())
                for v in chain]):
            print(self.chain)
            raise ValueError('first and last entry contain nan')


class MALA(LudwigWinkler):
    def __init__(self, model, trainloader, epsilon, num_steps,
                 burn_in, pretrain, tune, num_chains):
        LudwigWinkler.__init__(self, model, trainloader)
        self.sampler = MALA_Sampler(
            probmodel=self.model,
            step_size=epsilon,
            num_steps=num_steps,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune,
            num_chains=num_chains)

    def __repr__(self):
        return 'MALA'


class SGNHT(LudwigWinkler):
    def __init__(self, model, trainloader, epsilon, num_steps, burn_in,
                 pretrain, tune, L, num_chains):
        LudwigWinkler.__init__(self, model, trainloader)
        self.sampler = SGNHT_Sampler(
            probmodel=self.model,
            step_size=epsilon,
            num_steps=num_steps,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune,
            traj_length=L,
            num_chains=num_chains)

    def __repr__(self):
        return 'SGNHT'


class SGLD(LudwigWinkler):
    def __init__(self, model, trainloader,  epsilon, num_steps,
                 burn_in, pretrain, tune, num_chains=7):
        LudwigWinkler.__init__(self, model, trainloader)
        self.sampler = SGLD_Sampler(
            probmodel=self.model,
            step_size=epsilon,
            num_steps=num_steps,
            num_chains=num_chains,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune)

    def __repr__(self):
        return 'SGLD'

# # FAILING CONSISTENTLY
# class HMC(LudwigWinkler):
#     def __init__(self, model, X, y, batch_size, step_size, num_steps,
#                  num_chains, burn_in, pretrain=False, tune=False,
#                  traj_length=5):
#         LudwigWinkler.__init__(self, model, X, y, batch_size)
#         self.sampler = HMC_Sampler(
#             self.model, step_size, num_steps,
#             num_chains, burn_in, pretrain=pretrain,
#             tune=False, traj_length=traj_length)


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.distributions as td
    from Pytorch.Layer.Hidden import Hidden

    no_in = 2
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden(no_in, no_out, bias=False, activation=nn.ReLU())

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

    step_size = 0.01
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    hmc_traj_length = 24
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    # hmc = HMC(reg, X, y, X.shape[0],
    #           step_size=step_size, num_steps=num_steps, burn_in=burn_in, pretrain=pretrain, tune=tune,
    #           traj_length=hmc_traj_length,
    #           num_chains=num_chains)
    # hmc.sample()

    reg.reset_parameters()
    sgnht = SGNHT(reg, X, y, X.shape[0],
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  hmc_traj_length=hmc_traj_length,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)

    sgnht.model.plot(X, y)  # function still error prone: multiple executions seem to change the plot

    from _collections import OrderedDict

    # sgnht.model.plot1d(X, y, true_model=sgnht.model.true_model, param=
    # [OrderedDict({'W': c.view((1, 1))}) for i, c in enumerate(sgnht.chain) if i % 100 == 0])

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

    if no_in == 1:
        kwargs = {}
    elif no_in == 2:
        kwargs = {'multi_subplots': False, 'title': 'SOMETHING'}
    sgld.model.plot(X, y, **kwargs)
