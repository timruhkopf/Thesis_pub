import torch
import torch.nn as nn
import torch.distributions as td

from inspect import getfullargspec
from copy import deepcopy

import os
import matplotlib.pyplot as plt
from src.Grid.Util.Grid_Tracker import Grid_Tracker
from src.Grid.Util.Continuation import Continuation
from src.Grid.Util.Sampler_set_up import Sampler_set_up
import pickle


class GRID_Layout(Grid_Tracker, Continuation, Sampler_set_up):
    # FIXME: evaluate: current MSE is the most interesting, if the model has not yet converged,
    # otherwise the avgMSE is distorted!
    def main(self, n, n_val, model_class, model_param, sampler_name, sampler_param, seperated, name=''):
        self.basename = self.pathresults + '{}_{}_{}'.format(model_class.__name__, sampler_name, name) + self.hash
        self.config = {'n': n, 'n_val': n_val, 'model_class': model_class, 'model_param': model_param,
                       'sampler_name': sampler_name, 'sampler_param': sampler_param, 'seperated': seperated}
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

        # for restoration of the model, it needs to be reinstantiated
        # and loaded the state dict upon
        self.config['true_model'] = self.model.true_model
        self.config['init_model'] = self.model.init_model
        with open(self.basename + '_config.pkl', 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.basename + '_chain.pkl', 'wb') as handle:
            pickle.dump(self.sampler.chain, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return metrics

    def set_up_model(self, model_class, model_param, seperated):
        self.model = model_class(**model_param)
        self.model.to(self.device)
        if 'seperated' in getfullargspec(self.model.reset_parameters).args:
            self.model.reset_parameters(seperated)
        else:
            self.model.reset_parameters()
        self.model.true_model = deepcopy(self.model.state_dict())
        self.model.true_vec = self.model.vec

    def set_up_data(self, n, n_val, model_param, batch_size):
        from torch.utils.data import TensorDataset, DataLoader

        if 'no_in' in self.model_param.keys():
            no_in = self.model_param['no_in']
        elif 'no_basis' in self.model_param.keys():
            no_in = self.model_param['no_basis']
        elif 'hunits' in self.model_param.keys():
            no_in = self.model_param['hunits'][0]

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

    def evaluate_model(self):
        import random
        subsample = min(100, len(self.sampler.chain))  # for chain prediction MSE & log prob

        plot_subsample = min(30, len(self.sampler.chain))

        # plot a subset of the chain's predictions
        self.sampler.model.plot(*self.data_plot, random.sample(self.sampler.chain, plot_subsample),
                                path=self.basename + '_datamodel_random', title='')
        self.sampler.model.plot(*self.data_plot, self.sampler.chain[-plot_subsample:],
                                path=self.basename + '_datamodel_last', title='')

        with torch.no_grad():
            # Ludwig's extrawurst
            from src.Samplers.LudwigWinkler import LudwigWinkler
            if issubclass(type(self.sampler), LudwigWinkler):
                # since Ludwig required special log_prob to begin with
                if len(self.data) == 3:
                    self.sampler.model.log_prob1 = lambda X, Z, y: self.sampler.model.log_prob(X, Z, y)['log_prob']
                else:
                    self.sampler.model.log_prob1 = lambda X, y: self.sampler.model.log_prob(X, y)['log_prob']

            else:
                self.sampler.model.log_prob1 = self.sampler.model.log_prob

            val_MSE_chain = torch.Tensor(subsample, )
            # val_logprobs = torch.Tensor(subsample, )
            for i, c in enumerate(random.sample(self.sampler.chain, subsample)):
                self.sampler.model.load_state_dict(c)
                # val_logprobs[i] = self.sampler.model.log_prob1(*self.data_val)
                pred = self.sampler.model.forward(*self.data_val[:-1])
                val_MSE_chain[i] = torch.mean((pred - self.data_val[-1]) ** 2)

            mse_diff = torch.mean(val_MSE_chain) - self.val_MSE
            # log_diff = torch.mean(val_logprobs) - self.val_logprob

        fig1, ax1 = plt.subplots()
        ax1.hist(x=val_MSE_chain.detach().numpy())
        ax1.axvline(x=self.val_MSE.detach().numpy(), label='True MSE', color='red')
        ax1.set_title('MSE distribution on validation')
        ax1.set_xlabel("value")
        ax1.set_ylabel("Frequency")
        fig1.savefig(self.basename + '_MSE.pdf', bbox_inches='tight')

        # fig1, ax1 = plt.subplots()
        # ax1.hist(val_logprobs.detach().numpy())
        # ax1.axvline(x=self.val_logprob.detach().numpy(), color='red')
        # ax1.set_title('Log_prob distribution on validation')
        # ax1.set_xlabel("value")
        # ax1.set_ylabel("Frequency")
        # fig1.savefig(self.basename + '_Log_probs.pdf', bbox_inches='tight')

        plt.close('all')
        return {  # 'ess_min': self.sampler.ess_min,
            'avg_MSE_diff': mse_diff.detach().numpy(),
            'true_MSE': self.val_MSE.detach().numpy(),
            # 'avg_log_prob_diff': log_diff.detach().numpy(),
            # 'true_log_prob': self.val_logprob.detach().numpy()
        }


if __name__ == '__main__':

    # (Hidden) -----------------------------------------------------------------
    from src.Layer.Hidden import Hidden

    # run = 'HIDDEN_SGNHT'
    # root = '/home/tim/PycharmProjects/Thesis/src/Experiments/Results/{}/'.format(run)
    # #root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    # #    os.getcwd() + '/Results/{}/'.format(run)
    #
    # grid = GRID_Layout(root)
    # prelim_configs = grid.grid_exec_SGNHT(steps=1000, batch_size=100)
    #
    # n = 1000
    # n_val = 100
    # model_param = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU())
    #
    # for config in prelim_configs:
    #     grid.main(n=n, n_val=n_val, seperated=True,
    #               model_class=Hidden, model_param=model_param,
    #               sampler_name='SGNHT', sampler_param=config)

    # (HiddenRHMC) -----------------------------------------------------------------
    run = 'HIDDEN_RHMC2'
    root = '/home/tim/PycharmProjects/Thesis/src/Experiments/Results/{}/'.format(run)
    # root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    #     os.getcwd() + '/Results/{}/'.format(run)

    grid = GRID_Layout(root)
    prelim_configs = grid.grid_exec_RHMC(steps=1000)

    n = 1000
    n_val = 100
    model_param = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU())
    import matplotlib

    # matplotlib.use('TkAgg')

    # VERY IMPORTANT FEATURE: continue sampling from successfull models!
    grid = GRID_Layout(root)
    grid.continue_sampling_successfull(n=1000, n_val=100, n_samples=1000, burn_in=100, path=root)

    for config in prelim_configs:
        grid.main(n=n, n_val=n_val, seperated=True,
                  model_class=Hidden, model_param=model_param,
                  sampler_name='RHMC', sampler_param=config)

    # (Glasso) -----------------------------------------------------------------
    from src.Layer.Group_lasso import Group_lasso

    run = 'Glasso_SGNHT'
    root = '/home/tim/PycharmProjects/Thesis/src/Experiments/Results/{}/'.format(run)
    # #root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    # #    os.getcwd() + '/Results/{}/'.format(run)

    glasso_unittest = GRID_Layout(root)
    prelim_configs = glasso_unittest.grid_exec_SGNHT(steps=1000, batch_size=100)

    n = 100
    n_val = 100
    model_param = dict(no_in=2, no_out=1, bias=True,
                       activation=nn.ReLU(), bijected=True)  # TODO CHECK not bijected

    for config in prelim_configs:
        glasso_unittest.main(n=n, n_val=n_val, model_class=Group_lasso, model_param=model_param,
                             sampler_name='SGNHT', sampler_param=config, seperated=True)

    # (Ghorse) -----------------------------------------------------------------
    from src.Layer.Group_HorseShoe import Group_HorseShoe

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

    # (BNN) --------------------------------------------------------------------
    import torch.nn as nn
    from src.Models.BNN import BNN

    run = 'BNN_SGNHT'
    root = '/home/tim/PycharmProjects/Thesis/src/Experiments/Results/{}/'.format(run)
    # #root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    # #    os.getcwd() + '/Results/{}/'.format(run)

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
    from src.Models.ShrinkageBNN import ShrinkageBNN

    run = 'Shrinkage_BNN_SGNHT'
    root = '/home/tim/PycharmProjects/Thesis/src/Experiments/Results/{}/'.format(run)
    # #root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    # #    os.getcwd() + '/Results/{}/'.format(run)
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
