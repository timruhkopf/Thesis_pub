import torch
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt

import os

from Pytorch.Util.GridUtil import Grid


# TODO : PATHS for images must changed
# TODO : rearange plots of acf and traces
# TODO : ensure, that all main calls for one GRID instance will end up in a
#       designated folder with a result_table
# TODO : move chain_mat to Sampler.sample call, after checkup! ensure that all
#       references to chain_mat are on the object not the property thereafter!


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
        self.basename = self.pathresults + 'GAM_{}'.format(sampler_name) + self.hash
        # set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        self.set_up_datamodel(n, n_val, model_param)
        X, Z, y = self.data

        self.set_up_sampler(self.model, Z, y, sampler_name, sampler_param)
        metrics = self.evaluate_model(*self.data_val)

        return metrics  # to be written on Grid's result table

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
        self.model.reset_parameters(mode='U-MVN')
        print('state: ', self.model.state_dict())

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
        model.reset_parameters(mode='U-MVN')

        try:
            batch_size = sampler_param.pop('batch_size')
        except:
            batch_size = X.shape[0]

        # send data to device
        X.to(self.device)
        y.to(self.device)

        if sampler_name in ['SGNHT', 'SGLD', 'MALA']:  # geoopt based models
            Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                       'MALA': MALA,  # step_size
                       'SGLD': SGLD  # step_size
                       }[sampler_name]
            self.sampler = Sampler(model, X, y, batch_size, **sampler_param)
            self.sampler.sample()

        elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:

            burn_in, n_samples = sampler_param.pop('burn_in'), sampler_param.pop('n_samples')

            trainset = TensorDataset(X, y)
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

            Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                       'SGRLD': myRSGLD,  # epsilon
                       'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                       }[sampler_name]
            self.sampler = Sampler(model, **sampler_param)
            self.sampler.sample(trainloader, burn_in, n_samples)

        else:
            raise ValueError('sampler_name was not correctly specified')

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        self.sampler.save(self.basename)

    def evaluate_model(self, X_val, Z_val, y_val):
        import random
        subsample = 100

        # plot a subset of the chain's predictions
        sampler = self.sampler
        sampler.model.plot(X_val, y_val, random.sample(sampler.chain, 30),
                           path=self.basename + '_datamodel.png', title='GAM')  # FIXME: PATH

        # Efficiency of the sampler (Effective Sample Size)
        sampler.traceplots(path=self.basename + '_traces.png')  # FIXME path
        sampler.acf_plots(nlags=500, path=self.basename + '_acf.png')  # FIXME path

        sampler.ess(nlags=200)
        print(sampler.ess_min)

        # average performance of a sampler (MSE & log_prob) on validation
        with torch.no_grad():
            Z_val.to(self.device)
            y_val.to(self.device)

            # Ludwig's extrawurst
            from Pytorch.Samplers.LudwigWinkler import LudwigWinkler
            if issubclass(type(self.sampler), LudwigWinkler):
                # since Ludwig required special log_prob to begin with
                sampler.model.log_prob1 = lambda X, y: sampler.model.log_prob(X, y)['log_prob']
            else:
                sampler.model.log_prob1 = sampler.model.log_prob

            subset_chain = random.sample(sampler.chain, subsample)
            sampler.val_MSE_chain = torch.Tensor(subsample, )
            sampler.val_logprobs = torch.Tensor(subsample, )

            for i, c in enumerate(subset_chain):
                sampler.model.load_state_dict(c)
                sampler.val_logprobs[i] = sampler.model.log_prob1(Z_val, y_val)

                pred = sampler.model.forward(Z_val)

                sampler.val_MSE_chain[i] = torch.mean((pred - y_val) ** 2)

            mse_diff = torch.mean(sampler.val_MSE_chain) - sampler.model.val_MSE
            log_diff = torch.mean(sampler.val_logprobs) - sampler.model.val_logprob

        fig1, ax1 = plt.subplots()
        ax1.hist(x=sampler.val_MSE_chain.detach().numpy())
        ax1.axvline(x=sampler.model.val_MSE.detach().numpy(), label='True MSE', color='red')
        ax1.set_title('MSE distribution on validation')
        ax1.set_xlabel("value")
        ax1.set_ylabel("Frequency")
        fig1.savefig(self.basename + '_MSE.png', bbox_inches='tight')

        fig1, ax1 = plt.subplots()
        ax1.hist(sampler.val_logprobs.detach().numpy())
        ax1.axvline(x=sampler.model.val_logprob.detach().numpy(), color='red')
        ax1.set_title('Log_prob distribution on validation')
        ax1.set_xlabel("value")
        ax1.set_ylabel("Frequency")
        fig1.savefig(self.basename + '_Log_probs.png', bbox_inches='tight')


        plt.close()
        return {'ess_min': sampler.ess_min,
                'avg_MSE_diff': mse_diff.detach().numpy(),
                'avg_log_prob_diff': log_diff.detach().numpy()}

    # TODO: Check each Grid to be runnable configs & the defaults of param included in grid defaults
    # ludwig based
    def grid_exec_MALA(self, steps, batch_size, epsilons=np.arange(0.0001, 0.05, 0.002)):
        for epsilon in epsilons:
            yield dict(epsilon=epsilon, num_steps=steps, batch_size=batch_size, pretrain=False,
                       tune=False, burn_in=int(steps * 0.10), num_chains=1)

    def grid_exec_SGLD(self, steps, epsilons, batch_size):
        return self.grid_exec_MALA(steps, epsilons, batch_size)

    def grid_exec_sgnht(self, steps, batch_size,
                        epsilons=np.arange(0.001, 0.01, 0.002),
                        Ls=[1, 2, 3, 5, 10]):
        """

        :param steps:
        :param step_sizes:
        :param Ls :hmc_traj_lengths:
        :return: generator of tuples: sampler_name, model_param, sampler_param
        """
        for epsilon in epsilons:
            for L in Ls:
                yield dict(epsilon=epsilon, L=L, num_steps=steps, pretrain=False,
                           tune=False, burn_in=int(steps * 0.10), num_chains=1,
                           batch_size=batch_size)

    # geoopt based
    def grid_exec_RHMC(self, steps,
                       epsilons=np.arange(0.001, 0.05, 0.005),
                       Ls=[1, 2, 3, 5, 10]):
        """

        :param steps: number of sampler steps
        :param epsilons: step_size
        :param n_stepss: leapfrog steps
        :return: generator of tuples:  model_param, sampler_param
        """

        for epsilon in epsilons:
            for L in Ls:
                yield dict(epsilon=epsilon, L=L, n_samples=steps,
                           burn_in=int(steps * 0.10)
                           )

    def grid_exec_SGRHMC(self, steps, epsilons=np.arange(0.001, 0.01, 0.002),
                         Ls=[1, 2, 3, 5, 10], alphas=np.arange(0., 0.99, 0.20)):
        for epsilon in epsilons:
            for L in Ls:
                for alpha in alphas:
                    yield dict(epsilon=epsilon, L=L, alpha=alpha,
                               n_samples=steps,
                               burn_in=int(steps * 0.10))

    def grid_exec_SGRLD(self, steps, epsilons=np.arange(0.001, 0.01, 0.002)):
        for epsilon in epsilons:
            yield dict(epsilon=epsilon, n_samples=steps, burn_in=int(steps * 0.10))


if __name__ == '__main__':
    run = 'GAM_test_grid_exec_SGRLD'

    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    # (PRELIMINARY RUN) --------------------------------------------------------
    # to figure out the appropriate parameters
    gam_unittest = GAM_Grid(root)

    # batch_size = 50
    # preliminary configs to check which step sizes will work

    # prelim_configs = gam_unittest.grid_exec_sgnht(steps=1000, batch_size=100)
    # sampler_name = 'SGNHT'
    # for prelim_config in prelim_configs:
    #     gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name,
    #                       model_param=dict(no_basis=20, bijected=True),
    #                       sampler_param=prelim_config)
    # n = 1000
    # prelim_configs = gam_unittest.grid_exec_MALA(steps=1000, batch_size=n)
    # sampler_name = 'MALA'
    # for prelim_config in prelim_configs:
    #     gam_unittest.main(n=n, n_val=100, sampler_name=sampler_name,
    #                       model_param=dict(no_basis=20, bijected=True),
    #                       sampler_param=prelim_config)



    prelim_configs = gam_unittest.grid_exec_SGRLD(steps=1000)  # TODO: EXTEND THE GRID
    next(prelim_configs)

    sampler_name = 'SGRLD'
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name,
                          model_param=dict(no_basis=20, bijected=True),
                          sampler_param=prelim_config)

    prelim_configs = gam_unittest.grid_exec_SGRHMC(steps=1000)
    sampler_name = 'SGRHMC'
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name,
                          model_param=dict(no_basis=20, bijected=True),
                          sampler_param=prelim_config)

    prelim_configs = gam_unittest.grid_exec_RHMC(steps=1000)
    sampler_name = 'RHMC'
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name,
                          model_param=dict(no_basis=20, bijected=True),
                          sampler_param=prelim_config)


    print()

    # (MAIN RUN) ---------------------------------------------------------------
    # a single execution of one config
    gam_unittest = GAM_Grid(root)
    epsilon = 0.01  # to be determined
    L = 1
    steps = int(1e5)
    n = int(1e4)

    steps = 500
    n = 1000

    gam_unittest.main(n, n_val=100, sampler_name='SGNHT',
                      model_param=dict(no_basis=20, bijected=True),
                      sampler_param=dict(epsilon=epsilon, num_steps=steps, batch_size=n,
                                         pretrain=False,
                                         tune=False,
                                         burn_in=int(steps * 0.10),
                                         num_chains=1,
                                         L=L))

    print()

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
