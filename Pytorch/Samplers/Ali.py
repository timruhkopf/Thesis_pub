import torch
import numpy as np
from tqdm import tqdm

from copy import deepcopy

from Pytorch.Samplers.Samplers import Sampler
from alisiahkoohi.Langevin_dynamics_master.langevin_sampling.samplers import MetropolisAdjustedLangevin, pSGLD, \
    LangevinDynamics
from alisiahkoohi.Langevin_dynamics_master.langevin_sampling.precondSGLD import pSGLD
from alisiahkoohi.Langevin_dynamics_master.langevin_sampling.SGLD import SGLD


class Ali(Sampler):
    def __init__(self, model, X, Y, sg=True, seed=19):
        # https://github.com/alisiahkoohi/Langevin-dynamics

        self.model = model
        if sg:
            self.model.closure_log_prob(X, Y)

        np.random.seed(seed)
        torch.manual_seed(seed)

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.chain = list()
        self.sampler = None
        self.logs = None
        self.chain = None

    def sample_MALA(self, param_init, max_itr, lr=1e-1, lr_final=4e-2):
        self.model.init_model = deepcopy(param_init)
        p_init = deepcopy(param_init)
        p_init.requires_grad_()
        self.sampler = MetropolisAdjustedLangevin(
            p_init, self.model.log_prob,
            lr=lr, lr_final=lr_final,
            max_itr=max_itr, device=self.device)
        self._sample(max_itr)

    def sample_LD(self, param_init, max_itr, lr=1e-1, lr_final=4e-2):
        self.model.init_model = deepcopy(param_init)
        p_init = deepcopy(param_init)
        p_init.requires_grad_()
        self.sampler = LangevinDynamics(
            p_init, self.model.log_prob,
            lr=lr, lr_final=lr_final,
            max_itr=max_itr, device=self.device)

        self._sample(max_itr)

    def _sample(self, max_itr):
        # surrogatge to gather log_prob value & chain + progressbar
        hist_samples = []
        loss_log = []
        for j in tqdm(range(max_itr)):

            est, loss = self.sampler.sample()
            loss_log.append(loss)
            if j % 3 == 0:
                hist_samples.append(est.cpu())

        self.logs = loss_log
        self.chain = hist_samples


if __name__ == '__main__':
    from Pytorch.Layer.Hidden import Hidden
    import torch.distributions as td
    import torch.nn as nn

    no_in = 10
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    reg.true_model = reg.vec

    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()

    reg.reset_parameters()
    init_theta = reg.vec
    reg.true_model = deepcopy(init_theta)

    ali = Ali(reg, X, y)
    ali.sample_MALA(init_theta, max_itr=1000)
    print(ali.chain)
    print(ali.posterior_mean())

    ali.sample_LD(init_theta, max_itr=1000)
    print(ali.chain)
    print(init_theta)