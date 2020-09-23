import torch
import torch.nn as nn
import torch.distributions as td

import numpy as np
from inspect import getfullargspec
from copy import deepcopy

import os
import matplotlib.pyplot as plt
from Pytorch.Util.GridUtil import Grid


class GRID_Layout(Grid):

    def main(self, n, n_val, model_class, model_param, sampler_name, sampler_param, seperated):
        self.basename = self.pathresults + '{}_{}_'.format(model_class.__name__, sampler_name) + self.hash
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        self.model_param = model_param

        if 'batch_size' in sampler_param.keys():
            batch_size = sampler_param.pop('batch_size')
        else:
            batch_size = n

        self.set_up_model(model_class, model_param, seperated)
        self.set_up_data(n, n_val, model_param, batch_size)

        # send data to device
        for tensor in [*self.data, *self.data_val]:
            tensor.to(self.device)

        self.set_up_sampler(sampler_name, sampler_param)
        metrics = self.evaluate_model()  # *self.data_val FIXME

        return metrics

    def set_up_model(self, model_class, model_param, seperated):
        self.model = model_class(**model_param)
        self.model.to(self.device)
        if 'seperated' in getfullargspec(self.model.reset_parameters).args:
            self.model.reset_parameters(seperated)
        else:
            self.model.reset_parameters()
        self.model.true_model = deepcopy(self.model.state_dict())

    def set_up_data(self, n, n_val, model_param, batch_size):
        from torch.utils.data import TensorDataset, DataLoader

        if 'no_in' in self.model_param.keys():
            no_in = self.model_param['no_in']
        elif 'no_basis' in self.model_param.keys():
            no_in = self.model_param['no_basis']

        X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, no_in)
        X_val = X_dist.sample(torch.Size([n_val])).view(n_val, no_in)

        self.data = X,
        self.data_val = X_val,

        y = self.model.likelihood(*self.data).sample()
        y_val = self.model.likelihood(*self.data_val).sample()

        self.data = X, y
        self.data_val = X_val, y_val
        self.data_plot = X_val, y_val

        trainset = TensorDataset(*self.data)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.val_logprob = self.model.log_prob(*self.data_val)
        with torch.no_grad():
            self.val_MSE = torch.nn.MSELoss()(
                self.model.forward(*self.data_val[:-1]), self.data_val[-1])

    def set_up_sampler(self, sampler_name, sampler_param):
        from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
        from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD

        # (SAMPLER) Select sampler, set up  and sample -------------------------
        # random init state
        self.model.reset_parameters()

        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('Tkagg')
        # plt.ion()
        from copy import deepcopy
        init = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.model.true_model)
        self.model.plot(*self.data_plot, chain=[init], path=self.basename + '_initmodel')
        self.model.load_state_dict(init)

        # searching for a stable init for sgnht and all RM sampler
        # if sampler_name in ['SGNHT', 'RHMC', 'SGRLD', 'SGRHMC']:
        #     self.sampler_init_run(trainloader)

        if sampler_name in ['SGNHT', 'SGLD', 'MALA']:  # geoopt based models
            Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                       'MALA': MALA,  # step_size
                       'SGLD': SGLD  # step_size
                       }[sampler_name]
            self.sampler = Sampler(self.model, self.trainloader, **sampler_param)
            self.sampler.sample()


        elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:
            n_samples = sampler_param.pop('n_samples')
            burn_in = sampler_param.pop('burn_in')

            Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                       'SGRLD': myRSGLD,  # epsilon
                       'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                       }[sampler_name]
            self.sampler = Sampler(self.model, **sampler_param)
            self.sampler.sample(self.trainloader, burn_in, n_samples)

        else:
            raise ValueError('sampler_name was not correctly specified')

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        self.sampler.save(self.basename)

        self.sampler.traceplots(path=self.basename + '_traces.png')
        self.sampler.acf_plots(nlags=500, path=self.basename + '_acf.png')

        self.sampler.ess(nlags=200)
        print(self.sampler.ess_min)

    def evaluate_model(self):
        import random
        subsample = min(100, len(self.sampler.chain))  # for chain prediction MSE & log prob
        subsample = 100

        plot_subsample = min(30, len(self.sampler.chain))

        # plot a subset of the chain's predictions
        self.sampler.model.plot(*self.data_plot, random.sample(self.sampler.chain, plot_subsample),
                                path=self.basename + '_datamodel_random', title='')
        self.sampler.model.plot(*self.data_plot, self.sampler.chain[-plot_subsample:],
                                path=self.basename + '_datamodel_last', title='')

        with torch.no_grad():
            # Ludwig's extrawurst
            from Pytorch.Samplers.LudwigWinkler import LudwigWinkler
            if issubclass(type(self.sampler), LudwigWinkler):
                # since Ludwig required special log_prob to begin with
                if len(self.data) == 3:
                    self.sampler.model.log_prob1 = lambda X, Z, y: self.sampler.model.log_prob(X, Z, y)['log_prob']
                else:
                    self.sampler.model.log_prob1 = lambda X, y: self.sampler.model.log_prob(X, y)['log_prob']

            else:
                self.sampler.model.log_prob1 = self.sampler.model.log_prob

            val_MSE_chain = torch.Tensor(subsample, )
            val_logprobs = torch.Tensor(subsample, )
            for i, c in enumerate(random.sample(self.sampler.chain, subsample)):
                self.sampler.model.load_state_dict(c)
                val_logprobs[i] = self.sampler.model.log_prob1(*self.data_val)
                pred = self.sampler.model.forward(*self.data_val[:-1])
                val_MSE_chain[i] = torch.mean((pred - self.data_val[-1]) ** 2)

            mse_diff = torch.mean(val_MSE_chain) - self.val_MSE
            log_diff = torch.mean(val_logprobs) - self.val_logprob

        fig1, ax1 = plt.subplots()
        ax1.hist(x=val_MSE_chain.detach().numpy())
        ax1.axvline(x=self.val_MSE.detach().numpy(), label='True MSE', color='red')
        ax1.set_title('MSE distribution on validation')
        ax1.set_xlabel("value")
        ax1.set_ylabel("Frequency")
        fig1.savefig(self.basename + '_MSE.png', bbox_inches='tight')

        fig1, ax1 = plt.subplots()
        ax1.hist(val_logprobs.detach().numpy())
        ax1.axvline(x=self.val_logprob.detach().numpy(), color='red')
        ax1.set_title('Log_prob distribution on validation')
        ax1.set_xlabel("value")
        ax1.set_ylabel("Frequency")
        fig1.savefig(self.basename + '_Log_probs.png', bbox_inches='tight')

        plt.close('all')
        return {'ess_min': self.sampler.ess_min,
                'avg_MSE_diff': mse_diff.detach().numpy(),
                'true_MSE': self.val_MSE.detach().numpy(),
                'avg_log_prob_diff': log_diff.detach().numpy(),
                'true_log_prob': self.val_logprob.detach().numpy()}

    def grid_exec_MALA(self, steps, epsilons=np.arange(0.0001, 0.03, 0.003)):
        for epsilon in epsilons:
            yield dict(epsilon=epsilon, num_steps=steps, pretrain=False,
                       tune=False, burn_in=int(steps * 0.10), num_chains=1)

    def grid_exec_SGLD(self, steps, epsilons=np.arange(0.0001, 0.03, 0.003)):
        return self.grid_exec_MALA(steps, epsilons)

    def grid_exec_SGNHT(self, steps, batch_size,
                        epsilons=np.arange(0.0001, 0.03, 0.003),
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
                       epsilons=np.arange(0.0001, 0.03, 0.003),
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

    def grid_exec_SGRHMC(self, steps, batch_size, epsilons=np.arange(0.0001, 0.03, 0.003),
                         Ls=[1, 2, 3, 5, 7, 10], alphas=np.arange(0., 0.99, 0.20)):
        for epsilon in epsilons:
            for L in Ls:
                for alpha in alphas:
                    yield dict(epsilon=epsilon, L=L, alpha=alpha,
                               n_samples=steps, batch_size=batch_size,
                               burn_in=int(steps * 0.10))

    def grid_exec_SGRLD(self, steps, batch_size, epsilons=np.arange(0.0001, 0.03, 0.003)):
        for epsilon in epsilons:
            yield dict(epsilon=epsilon, n_samples=steps, burn_in=int(steps * 0.10), batch_size=batch_size)


class GRID_Layout_GAM(GRID_Layout):
    def set_up_data(self, n, n_val, model_param, batch_size):
        from torch.utils.data import TensorDataset, DataLoader
        from Tensorflow.Effects.bspline import get_design

        if 'no_in' in self.model_param.keys():
            no_in = self.model_param['no_in']
        elif 'no_basis' in self.model_param.keys():
            no_in = 1

        X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, )
        X_val = X_dist.sample(torch.Size([n_val])).view(n_val, )

        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=model_param['no_basis']),
                         dtype=torch.float32, requires_grad=False)
        Z_val = torch.tensor(get_design(X_val.numpy(), degree=2, no_basis=model_param['no_basis']),
                             dtype=torch.float32, requires_grad=False)

        self.data = Z,
        self.data_val = Z_val,

        y = self.model.likelihood(*self.data).sample()
        y_val = self.model.likelihood(*self.data_val).sample()

        self.data = Z, y
        self.data_val = Z_val, y_val
        self.data_plot = X_val, y_val

        trainset = TensorDataset(*self.data)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.val_logprob = self.model.log_prob(*self.data_val)
        with torch.no_grad():
            self.val_MSE = torch.nn.MSELoss()(
                self.model.forward(*self.data_val[:-1]), self.data_val[-1])


class GRID_Layout_STRUCTURED(GRID_Layout):
    def set_up_data(self, n, n_val, model_param, batch_size):
        from torch.utils.data import TensorDataset, DataLoader
        from Tensorflow.Effects.bspline import get_design

        no_in = self.model_param['hunits'][0]

        X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, no_in)
        X_val = X_dist.sample(torch.Size([n_val])).view(n_val, no_in)

        # explicit assumption that only the first variable is shrunken
        Z = torch.tensor(get_design(X[:, 0].numpy(), degree=2, no_basis=model_param['no_basis']),
                         dtype=torch.float32, requires_grad=False)
        Z_val = torch.tensor(get_design(X_val[:, 0].numpy(), degree=2, no_basis=model_param['no_basis']),
                             dtype=torch.float32, requires_grad=False)

        self.data = X, Z
        self.data_val = X_val, Z_val,

        y = self.model.likelihood(*self.data).sample()
        y_val = self.model.likelihood(*self.data_val).sample()

        self.data = X, Z, y
        self.data_val = X_val, Z_val, y_val
        self.data_plot = X_val, Z_val, y_val

        trainset = TensorDataset(*self.data)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.val_logprob = self.model.log_prob(*self.data_val)
        with torch.no_grad():
            self.val_MSE = torch.nn.MSELoss()(
                self.model.forward(*self.data_val[:-1]), self.data_val[-1])


if __name__ == '__main__':

    # (Hidden) -----------------------------------------------------------------
    from Pytorch.Layer.Hidden import Hidden

    run = 'Structured_HIDDEN_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    grid = GRID_Layout(root)
    prelim_configs = grid.grid_exec_SGNHT(steps=1000, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU())

    for config in prelim_configs:
        grid.main(n=n, n_val=n_val, seperated=True,
                  model_class=Hidden, model_param=model_param,
                  sampler_name='SGNHT', sampler_param=config)

    # (Glasso) -----------------------------------------------------------------
    from Pytorch.Layer.Group_lasso import Group_lasso

    run = 'Glasso_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    glasso_unittest = GRID_Layout(root)
    prelim_configs = glasso_unittest.grid_exec_SGNHT(steps=100, batch_size=100)

    n = 100
    n_val = 100
    model_param = dict(no_in=2, no_out=1, bias=False,
                       activation=nn.ReLU(), bijected=True)  # TODO CHECK not bijected

    for config in prelim_configs:
        glasso_unittest.main(n=n, n_val=n_val, model_class=Group_lasso, model_param=model_param,
                             sampler_name='SGNHT', sampler_param=config, seperated=True)

    # (Ghorse) -----------------------------------------------------------------
    from Pytorch.Layer.Group_HorseShoe import Group_HorseShoe

    run = 'Ghorse_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    glasso_unittest = GRID_Layout(root)
    prelim_configs = glasso_unittest.grid_exec_SGNHT(steps=100, batch_size=100)

    n = 100
    n_val = 100
    model_param = dict(no_in=2, no_out=1, bias=False,
                       activation=nn.ReLU(), bijected=True)  # TODO CHECK not bijected

    for config in prelim_configs:
        glasso_unittest.main(n=n, n_val=n_val, model_class=Group_HorseShoe, model_param=model_param,
                             sampler_name='SGNHT', sampler_param=config, seperated=True)

    # (GAM) -----------------------------------------------------------------

    run = 'GAM_SGRLD1'

    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)
    gam_unittest = GRID_Layout_GAM(root)

    from Pytorch.Models.GAM import GAM

    prelim_configs = gam_unittest.grid_exec_SGRLD(steps=100, batch_size=100)  # TODO: EXTEND THE GRID
    next(prelim_configs)

    sampler_name = 'SGRLD'
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name, model_class=GAM,
                          model_param=dict(no_basis=20, bijected=True),
                          sampler_param=prelim_config, seperated=True)

    # (BNN) --------------------------------------------------------------------
    import torch.nn as nn
    from Pytorch.Models.BNN import BNN

    run = 'BNN_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    grid = GRID_Layout(root)
    prelim_configs = grid.grid_exec_SGNHT(steps=1000, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                       final_activation=nn.Identity(),
                       seperated=True)

    for config in prelim_configs:
        grid.main(n=n, n_val=n_val, seperated=True,
                  model_class=BNN, model_param=model_param,
                  sampler_name='SGNHT', sampler_param=config)

    # (SHRINK BNN) -------------------------------------------------------------
    import torch.nn as nn
    from Pytorch.Models.ShrinkageBNN import ShrinkageBNN

    run = 'Shrinkage_BNN_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    grid = GRID_Layout(root)
    prelim_configs = grid.grid_exec_SGNHT(steps=1000, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                       final_activation=nn.Identity(), shrinkage='ghorse',
                       seperated=True, bijected=True)

    for config in prelim_configs:
        grid.main(n=n, n_val=n_val, seperated=True,
                  model_class=ShrinkageBNN, model_param=model_param,
                  sampler_name='SGNHT', sampler_param=config)

    # (STRUCT BNN) -------------------------------------------------------------
    import torch.nn as nn
    from Pytorch.Models.StructuredBNN import StructuredBNN

    run = 'Structured_BNN_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    grid = GRID_Layout_STRUCTURED(root)
    prelim_configs = grid.grid_exec_SGNHT(steps=1000, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                       final_activation=nn.Identity(), shrinkage='ghorse', no_basis=20,
                       seperated=True, alpha_type='constant', bijected=True)

    for config in prelim_configs:
        grid.main(n=n, n_val=n_val, seperated=True,
                  model_class=StructuredBNN, model_param=model_param,
                  sampler_name='SGNHT', sampler_param=config)
