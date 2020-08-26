import torch
import torch.distributions as td
import os

from Pytorch.Util.GridUtil import Grid


class GAMs(Grid):
    def main(self, n, n_val, model_param, sampler_param, sampler='sgnht'):
        from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
        from Tensorflow.Effects.bspline import get_design
        from Pytorch.Models.GAM import GAM
        import random
        from copy import deepcopy

        # set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        # set up data & model
        X_dist = td.Uniform(-10., 10)
        X = X_dist.sample(torch.Size([n]))
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=model_param['no_basis']),
                         dtype=torch.float32, requires_grad=False)

        # validation data
        X_val = X_dist.sample(torch.Size([n_val]))
        Z_val = torch.tensor(get_design(X_val.numpy(), degree=2, no_basis=model_param['no_basis']),
                             dtype=torch.float32, requires_grad=False)

        gam = GAM(order=1, **model_param)
        gam.to(device)
        gam.reset_parameters()

        y = gam.likelihood(Z).sample()
        y_val = gam.likelihood(Z_val).sample()

        # save the true state & true model's performance on validation
        gam.true_model = deepcopy(gam.state_dict())
        gam.val_logprob = gam.log_prob(Z_val, y_val)
        with torch.no_grad():
            gam.val_MSE = torch.nn.MSELoss()(gam.forward(Z_val), y_val)

        # send data to device
        Z.to(device)
        y.to(device)

        # random init state
        gam.reset_parameters()
        if sampler == 'SGNHT':
            sampler = SGNHT(gam, X, y, X.shape[0], **sampler_param)

        elif sampler == 'RMHMC':

            # TODO : check whether the interface already returns
            #   chain = list(chain_dicts). otherwise ammend sample function
            pass

        sampler.sample()


        # (EVALUATION PROCEDURE) -----------------------------------------------
        # check the sampler actually did something meaningfull
        print(sampler.chain)
        if len(sampler.chain) == 1:
            raise ValueError('The chain did not progress beyond first step')

        if all(any(torch.isnan(v)) for chain in
               (sampler.chain[-1].values(), sampler.chain[0]) for v in chain):
            raise ValueError('first and last entry contain nan')

        # plot a subset of the chain's predictions
        sampler.model.plot(X_val, y_val, random.sample(sampler.chain, len(sampler.chain) // 10),
                           path=None, **{'title': 'Hidden'})  # FIXME: PATH

        sampler.traceplots(path=None)  # FIXME path
        sampler.acf_plots(path=None)  # FIXME path
        sampler.ess  # FIXME
        sampler.ess_min

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        sampler.save(self.pathresults + 'GAM_SGNHT' + self.hash + '.model')

    def grid_exec(self, n, steps):
        import numpy as np
        for step_size in np.arange(0.001, 0.05, 0.003):
            for hmc_traj_length in [1, 2, 3, 5, 10, 15, 20, 25]:
                for bijected in [True, False]:
                    yield {'n': n, 'model_param': dict(no_basis=20, bijected=bijected),
                           'sampler_param': dict(step_size=step_size, num_steps=steps, pretrain=False,
                                                 tune=False, burn_in=int(steps * 0.10), num_chains=1,
                                                 hmc_traj_length=hmc_traj_length)}


if __name__ == '__main__':
    gam_unittest = GAMs(root= \
                                 os.getcwd() if os.path.isdir(os.getcwd()) else \
                                     os.getcwd() + '/Pytorch/Experiments')

    # batch_size = 50
    # preliminary configs to check which step sizes will work
    prelim_config = gam_unittest.grid_exec(n=1000, steps=1000)
    gam_unittest.main(sampler='sgnht', **prelim_config)

    step_size = None
    hmc_traj_length = None
    steps = 1e5
    n = 1e4

    gam_unittest.main(**{'n': n, 'model_param': dict(no_basis=20, bijected=True),
                         'sampler_param': dict(step_size=step_size, num_steps=steps, pretrain=False,
                                               tune=False, burn_in=int(steps * 0.10), num_chains=1,
                                               hmc_traj_length=hmc_traj_length)})

    # # (1) unbijected
    # gam_unittest.main(n=1000, steps=1000, bijected=True, model_param={'no_basis': 20})
    #
    # # (2) bijected # FIXME: fails, since prior model is not implemented!
    # gam_unittest.main(n=1000, steps=1000, bijected=True, model_param={'no_basis': 20})

    # (3) penK

    # RECONSTRUCTION FROM MODEL FILES: ----------------------------------------------
    # from Pytorch.Models.GAM import GAM
    # from Pytorch.Samplers.Hamil import Hamil
    #
    # # filter a dir for .model files
    # models = [m for m in os.listdir('results/') if m.endswith('.model')]
    #
    # loaded_hamil = torch.load('results/' +models[0])
    # loaded_hamil.chain

print()
