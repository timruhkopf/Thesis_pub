import torch
import torch.distributions as td

import matplotlib.pyplot as plt

import os

from Pytorch.Util.GridUtil import Grid
from Pytorch.Experiments.SAMPLER_GRID import SAMPLER_GRID


# TODO : rearange plots of acf and traces
# TODO : move chain_mat to Sampler.sample call, after checkup! ensure that all
#       references to chain_mat are on the object not the property thereafter!


class GAM_Grid(Grid, SAMPLER_GRID):
    def main(self, n, n_val, model_class, model_param, sampler_name, sampler_param, seperated=False):
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
        self.model_class = model_class
        self.basename = self.pathresults + 'GAM_{}'.format(sampler_name) + self.hash
        # set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        self.set_up_datamodel(n, n_val, model_param)

        self.set_up_sampler(self.model, sampler_name, sampler_param)
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
        self.Z_val = torch.tensor(get_design(X_val.numpy(), degree=2, no_basis=model_param['no_basis']),
                                  dtype=torch.float32, requires_grad=False)

        self.model = self.model_class(**model_param)
        self.model.to(self.device)
        self.model.reset_parameters()
        print('state: ', self.model.state_dict())

        y = self.model.likelihood(Z).sample()
        y_val = self.model.likelihood(self.Z_val).sample()

        self.data = (Z, y)
        self.data_val = (X_val, y_val)

        # save the true state & true model's performance on validation
        self.model.true_model = deepcopy(self.model.state_dict())
        self.model.val_logprob = self.model.log_prob(self.Z_val, y_val)
        with torch.no_grad():
            self.model.val_MSE = torch.nn.MSELoss()(self.model.forward(self.Z_val), y_val)

    def evaluate_model(self, X_val, y_val):
        import random
        subsample = 100

        # plot a subset of the chain's predictions
        sampler = self.sampler
        sampler.model.plot(X_val, y_val, random.sample(sampler.chain, 30),
                           path=self.basename + '_datamodel', title='')

        # Efficiency of the sampler (Effective Sample Size)
        sampler.traceplots(path=self.basename + '_traces.png')
        sampler.acf_plots(nlags=500, path=self.basename + '_acf.png')

        sampler.ess(nlags=200)
        print(sampler.ess_min)

        # average performance of a sampler (MSE & log_prob) on validation
        with torch.no_grad():
            self.Z_val.to(self.device)
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
                sampler.val_logprobs[i] = sampler.model.log_prob1(self.Z_val, y_val)

                pred = sampler.model.forward(self.Z_val)

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


        plt.close('all')
        return {'ess_min': sampler.ess_min,
                'avg_MSE_diff': mse_diff.detach().numpy(), 'true_MSE': sampler.model.val_MSE.detach().numpy(),
                'avg_log_prob_diff': log_diff.detach().numpy(),
                'true_log_prob': sampler.model.val_logprob.detach().numpy()}



if __name__ == '__main__':
    run = 'GAM_test_grid_exec_SGRLD1'

    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    # (PRELIMINARY RUN) --------------------------------------------------------
    # to figure out the appropriate parameters
    gam_unittest = GAM_Grid(root)

    # batch_size = 50
    # preliminary configs to check which step sizes will work

    # prelim_configs = gam_unittest.grid_exec_SGNHT(steps=1000, batch_size=100)
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

    from Pytorch.Models.GAM import GAM

    prelim_configs = gam_unittest.grid_exec_SGRLD(steps=100, batch_size=100)  # TODO: EXTEND THE GRID
    next(prelim_configs)

    sampler_name = 'SGRLD'
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name, model_class=GAM,
                          model_param=dict(no_basis=20, bijected=True),
                          sampler_param=prelim_config)

    prelim_configs = gam_unittest.grid_exec_SGRHMC(steps=100, batch_size=100)
    sampler_name = 'SGRHMC'
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name, model_class=GAM,
                          model_param=dict(no_basis=20, bijected=True),
                          sampler_param=prelim_config)

    prelim_configs = gam_unittest.grid_exec_RHMC(steps=100)
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
