import torch
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt

import os

from Pytorch.Util.GridUtil import Grid

# TODO : PATHS for images must changed
# TODO : rearange plots of acf and traces


class GAM_Grid(Grid):
    def main(self, n, n_val, model_param, sampler_param, sampler_name='sgnht'):
        """

        :param n:
        :param n_val:
        :param model_param:
        :param sampler_param:
        :param sampler_name:
        :return: dict, containing scalar values of metrics that are to be tracked by Grid
        and written to Grid's result table
        """
        import torch
        # set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        self.set_up_datamodel(n, n_val, model_param)
        X,Z, y = self.data

        self.set_up_sampler(self.model, Z, y, sampler_name, sampler_param)
        metrics = self.evaluate_model(*self.data_val)

        return metrics # to be written on Grid's result table

    def set_up_datamodel(self, n, n_val, model_param):
        """
        setting up a GAM model, and both training and validation data
        i.e. self.model, self.data = (X, Z,y), self.data_val = (X_val, Z_val, y_val),
        self.model.true_model = deepcopy(state_dict())
        :param n: number of observations for training
        :param n_val: number of observations for validation
        :param model_param: dict passed to model
        :param device:

        """
        from Tensorflow.Effects.bspline import get_design
        from Pytorch.Models.GAM import GAM
        from copy import deepcopy

        # (MODEL & DATA) set up data & model -----------------------------------
        X_dist = td.Uniform(-10., 10)
        X = X_dist.sample(torch.Size([n]))
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=model_param['no_basis']),
                         dtype=torch.float32, requires_grad=False)

        # validation data
        X_val = X_dist.sample(torch.Size([n_val]))
        Z_val = torch.tensor(get_design(X_val.numpy(), degree=2, no_basis=model_param['no_basis']),
                             dtype=torch.float32, requires_grad=False)

        self.model = GAM(order=1, **model_param)
        self.model.to(self.device)
        self.model.reset_parameters()

        y = self.model.likelihood(Z).sample()
        y_val = self.model.likelihood(Z_val).sample()

        self.data = (X, Z, y)
        self.data_val = (X_val, Z_val, y_val)

        # save the true state & true model's performance on validation
        self.model.true_model = deepcopy(self.model.state_dict())
        self.model.val_logprob = self.model.log_prob(Z_val, y_val)
        with torch.no_grad():
            self.model.val_MSE = torch.nn.MSELoss()(self.model.forward(Z_val), y_val)


    def set_up_sampler(self, model, X, y, sampler_name, sampler_param):
        from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
        from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
        from torch.utils.data import TensorDataset, DataLoader

        # (SAMPLER) Select sampler, set up  and sample -------------------------
        # random init state
        model.reset_parameters()
        batch_size = sampler_param.pop('batch_size')

        # send data to device
        X.to(self.device)
        y.to(self.device)

        if sampler_name in ['SGNHT', 'SGLD', 'MALA']: # geoopt based models
            Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                       'MALA': MALA,  # step_size
                       'SGLD': SGLD  # step_size
                       }[sampler_name]
            self.sampler = Sampler(model, X, y, batch_size, **sampler_param)
            self.sampler.sample()

        elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:

            n_burn, n_samples = sampler_param.pop('n_burn'), sampler_param.pop('n_samples')

            trainset = TensorDataset(X, y)
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

            Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                       'SGRLD': myRSGLD,  # epsilon
                       'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                       }[sampler_name]
            self.sampler = Sampler(model, **sampler_param)
            self.sampler.sample(trainloader, n_burn, n_samples)

        else:
            raise ValueError('sampler_name was not correctly specified')

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        self.sampler.save(self.pathresults + 'GAM_{}'.format(sampler_name) + self.hash)

    def evaluate_model(self, X_val, Z_val, y_val):
        import random
        # TODO : ensure, that all main calls for one GRID instance will end up in a
        #       designated folder with a result_table
        # TODO : move chain_mat to Sampler.sample call, after checkup! ensure that all
        #       references to chain_mat are on the object not the property thereafter!

        # plot a subset of the chain's predictions
        sampler = self.sampler

        sampler.model.plot(X_val, y_val, random.sample(sampler.chain, len(sampler.chain) // 10),
                           path=None, title ='GAM')  # FIXME: PATH

        # Efficiency of the sampler (Effective Sample Size)
        sampler.traceplots(path=None)  # FIXME path
        sampler.acf_plots(nlags=500, path=None)  # FIXME path

        sampler.ess(nlags=400)
        print(sampler.ess_min)

        # average performance of a sampler (MSE & log_prob) on validation
        with torch.no_grad():
            Z_val.to(self.device)
            y_val.to(self.device)

            # FIXME: subsample the chain! according to ESS
            sampler.val_MSE_chain = torch.Tensor(len(sampler.chain),)
            sampler.val_logprobs = torch.Tensor(len(sampler.chain),)

            from Pytorch.Samplers.LudwigWinkler import LudwigWinkler
            if issubclass(type(self.sampler), LudwigWinkler):
                # since Ludwig required special log_prob to begin with
                sampler.model.log_prob1 = lambda X, y: sampler.model.log_prob(X, y)['log_prob']
            else:
                sampler.model.log_prob1 =  sampler.model.log_prob

            for i, c in enumerate(sampler.chain):
                sampler.model.load_state_dict(c)
                sampler.val_logprobs[i] = sampler.model.log_prob1(Z_val, y_val)

                pred = sampler.model.forward(Z_val)

                sampler.val_MSE_chain[i] = torch.mean((pred -  y_val)**2)

            mse_diff = torch.mean(sampler.val_MSE_chain) - sampler.model.val_MSE
            log_diff = torch.mean(sampler.val_logprobs) - sampler.model.val_logprob


        fig = plt.hist(x=sampler.val_MSE_chain.detach().numpy())
        plt.axvline(x=sampler.model.val_MSE.detach().numpy(), label='True MSE', color='red')
        plt.title('MSE distribution on validation')
        plt.xlabel("value")
        plt.ylabel("Frequency")
        # fixme: path

        fig = plt.hist(sampler.val_logprobs.detach().numpy())
        plt.axvline(x=sampler.model.val_logprob.detach().numpy(), color='red')
        plt.title('Log_prob distribution on validation')
        plt.xlabel("value")
        plt.ylabel("Frequency")
        # fixme: path

        return {'ess_min':sampler.ess_min,
                'avg_MSE_diff':mse_diff.detach().numpy(),
                'avg_log_prob_diff':log_diff.detach().numpy()}

    # TODO: Check each Grid to be runnable configs & the defaults of param included in grid defaults
    # ludwig based
    def grid_exec_MALA(self, steps, epsilons):
        for epsilon in epsilons:
            yield dict(no_basis=20, bijected=False), \
                  dict(epsilon=epsilon, num_steps=steps, pretrain=False,
                       tune=False, burn_in=int(steps * 0.10), num_chains=1)

    def grid_exec_SGLD(self, steps, epsilons):
        return self.grid_exec_MALA(steps, epsilons)

    def grid_exec_sgnht(self, steps,
                        epsilons=np.arange(0.001, 0.05, 0.003),
                        hmc_traj_lengths=[1, 2, 3, 5, 10, 15, 20, 25]):
        """

        :param steps:
        :param step_sizes:
        :param hmc_traj_lengths:
        :return: generator of tuples: sampler_name, model_param, sampler_param
        """
        for epsilon in epsilons:
            for hmc_traj_length in hmc_traj_lengths:
                for bijected in [True, False]:  # FIXME: Remove me!
                    yield dict(no_basis=20, bijected=bijected), \
                          dict(epsilon=epsilon, num_steps=steps, pretrain=False,
                               tune=False, burn_in=int(steps * 0.10), num_chains=1,
                               hmc_traj_length=hmc_traj_length)

    # geoopt based
    def grid_exec_RHMC(self, steps,
                       epsilons=np.arange(0.001, 0.05, 0.003),
                       Ls=[1, 2, 3, 5, 10, 15, 20, 25]):
        """

        :param steps: number of sampler steps
        :param epsilons: step_size
        :param n_stepss: leapfrog steps
        :return: generator of tuples:  model_param, sampler_param
        """

        for epsilon in epsilons:
            for L in Ls:  # TODO Check doch if this actually hmc_traj_length
                yield dict(no_basis=20, bijected=True), \
                      dict(epsilon=epsilon, num_steps=steps,
                           burn_in=int(steps * 0.10), num_chains=1,
                           L=L)

    def grid_exec_SGRHMC(self, steps, epsilons, Ls, alphas):
        for epsilon in epsilons:
            for L in Ls:
                for alpha in alphas:
                    yield dict(no_basis=20, bijected=True), \
                          dict(epsilon=epsilon, num_steps=steps,
                               burn_in=int(steps * 0.10), num_chains=1,
                               L=L, alpha=alpha)

    def grid_exec_RSGLD(self, steps, epsilons):
        for epsilon in epsilons:
            yield dict(no_basis=20, bijected=True), \
                  dict(epsilon=epsilon, num_steps=steps,
                       burn_in=int(steps * 0.10), num_chains=1)


if __name__ == '__main__':

    # (MAIN RUN) ---------------------------------------------------------------
    gam_unittest = GAM_Grid(root=os.getcwd() if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Pytorch/Experiments')
    epsilon = 0.01 # to be determined
    hmc_traj_length = 1
    steps = int(1e5)
    n = int(1e4)

    steps = 1000
    n = 1000

    gam_unittest.main(n, n_val=100, sampler_name='SGNHT',
                      model_param=dict(no_basis=20, bijected=True),
                      sampler_param=dict(epsilon=epsilon, num_steps=steps, batch_size=n,
                                         pretrain=False,
                                         tune=False,
                                         burn_in=int(steps * 0.10),
                                         num_chains=1,
                                         hmc_traj_length=hmc_traj_length))

    print()

    # (PRELIMINARY RUN) --------------------------------------------------------
    # to figure out the appropriate parameters
    gam_unittest = GAM_Grid(root=os.getcwd() if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Pytorch/Experiments')

    # batch_size = 50
    # preliminary configs to check which step sizes will work
    prelim_configs = gam_unittest.grid_exec_sgnht(steps=1000)
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name='SGNHT', *prelim_config)



    # (RECONSTRUCTION FROM MODEL FILES) ----------------------------------------
    # from Pytorch.Models.GAM import GAM
    # from Pytorch.Samplers.Hamil import Hamil
    #
    # # filter a dir for .model files
    # models = [m for m in os.listdir('results/') if m.endswith('.model')]
    #
    # loaded_hamil = torch.load('results/' +models[0])
    # loaded_hamil.chain

print()
