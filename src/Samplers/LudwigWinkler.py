from copy import copy

import torch

from src.Util.Util_Samplers import Util_Sampler
from thirdparty_repo.ludwigwinkler.src.MCMC_Sampler import \
    SGLD_Sampler, SGNHT_Sampler


class LudwigWinkler(Util_Sampler):
    def __init__(self, model, trainloader):
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

        self.model.check_chain(self.chain)


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
    def __init__(self, model, trainloader, epsilon, num_steps,
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

# # FIXME: FAILING CONSISTENTLY
# class MALA(LudwigWinkler):
#     def __init__(self, model, trainloader, epsilon, num_steps,
#                  burn_in, pretrain, tune, num_chains):
#         LudwigWinkler.__init__(self, model, trainloader)
#         self.sampler = MALA_Sampler(
#             probmodel=self.model,
#             step_size=epsilon,
#             num_steps=num_steps,
#             burn_in=burn_in,
#             pretrain=pretrain,
#             tune=tune,
#             num_chains=num_chains)
#
#     def __repr__(self):
#         return 'MALA'


# # FIXME: FAILING CONSISTENTLY
# class HMC(LudwigWinkler):
#     def __init__(self, model, X, y, batch_size, step_size, num_steps,
#                  num_chains, burn_in, pretrain=False, tune=False,
#                  traj_length=5):
#         LudwigWinkler.__init__(self, model, X, y, batch_size)
#         self.sampler = HMC_Sampler(
#             self.model, step_size, num_steps,
#             num_chains, burn_in, pretrain=pretrain,
#             tune=False, traj_length=traj_length)
