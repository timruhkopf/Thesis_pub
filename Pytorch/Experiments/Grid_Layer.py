import torch
import torch.nn as nn
import torch.distributions as td
import matplotlib.pyplot as plt
from inspect import getfullargspec
import os
from copy import deepcopy

from Pytorch.Util.GridUtil import Grid
from Pytorch.Experiments.SAMPLER_GRID import SAMPLER_GRID
from Pytorch.Experiments.Grid_GAM import GAM_Grid

class Layer_Grid(Grid, SAMPLER_GRID):
    def main(self, n, n_val, seperated, model_class, model_param, sampler_param, sampler_name):
        self.basename = self.pathresults + \
                        '{}_{}'.format(str(model_class.__class__), sampler_name) + \
                        self.hash
        self.model_class = model_class

        # set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        no_in = model_param['no_in']
        self.set_up_datamodel(n, n_val, no_in, model_class, seperated, model_param)


        self.set_up_sampler(self.model, sampler_name, sampler_param)
        metrics = self.evaluate_model(*self.data_val)

        return metrics  # to be written on Grid's result table


    def set_up_datamodel(self, n, n_val, no_in, model_class, seperated, model_param):

        X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, no_in)
        X_val = X_dist.sample(torch.Size([n_val]))

        self.model = model_class(**model_param)


        if 'seperated' in getfullargspec(self.model.reset_parameters).args:
            self.model.reset_parameters(seperated)
        else:
            self.model.reset_parameters()

        self.model.true_model = deepcopy(self.model.state_dict())
        y = self.model.likelihood(X).sample()
        y_val = self.model.likelihood(X_val).sample()

        # check reset_parameters &  check prior_log_prob
        self.data = X, y
        self.data_val = X_val, y_val

        # save the true state & true model's performance on validation
        self.model.true_model = deepcopy(self.model.state_dict())
        self.model.val_logprob = self.model.log_prob(X_val, y_val)
        with torch.no_grad():
            self.model.val_MSE = torch.nn.MSELoss()(self.model.forward(X_val), y_val)

    def evaluate_model(self, X_val, y_val):
        metrics = GAM_Grid.evaluate_model(self, X_val=X_val, Z_val=X_val, y_val=y_val)

        if hasattr(self.model, 'tau'):
            # analyse the distribution of tau
            tau = torch.stack([v for odict in self.sampler.chain for k, v in odict.items() if k == 'tau'], axis=0)
            true_tau = self.model.true_model['tau']
            if self.model.bijected:
                tau = self.model.dist['tau'].transforms[0]._inverse(tau)
                true_tau = self.model.dist['tau'].transforms[0]._inverse(true_tau)

            fig1, ax1 = plt.subplots()
            ax1.hist(x=tau.numpy(), bins=100)
            ax1.axvline(x=true_tau.numpy(), label='True Tau', color='red')
            ax1.set_title('Tau distribution across the entire chain')
            ax1.set_xlabel("value")
            ax1.set_ylabel("Frequency")
            fig1.savefig(self.basename + '_Tau.png', bbox_inches='tight')

        return metrics


if __name__ == '__main__':

    # As it turns out, this interface can be used to sample from any layer! ---------
    # from Pytorch.Layer.Hidden import Hidden
    #
    # run = 'Hidden_test_grid_exec_SGRLD1'
    # root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    #     os.getcwd() + '/Results/{}/'.format(run)
    #
    # hidden_unittest = Layer_Grid(root)
    # prelim_configs = hidden_unittest.grid_exec_SGRLD(steps=1000, batch_size=100)
    # n = 1000
    # n_val = 100
    # model_param = dict(no_in=2, no_out=1, bias=False, activation=nn.Identity())
    #
    # for config in prelim_configs:
    #     hidden_unittest.main(n=n, n_val=n_val, seperated=None, model_class=Hidden, model_param=model_param,
    #                          sampler_name='SGRLD', sampler_param=config)

    from Pytorch.Layer.Group_lasso import Group_lasso

    run = 'Glasso_test_grid_exec_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    glasso_unittest = Layer_Grid(root)
    prelim_configs = glasso_unittest.grid_exec_SGNHT(steps=10000, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(no_in=2, no_out=1, bias=False,
                       activation=nn.ReLU(), bijected=True)  # TODO CHECK not bijected

    for config in prelim_configs:
        glasso_unittest.main(n=n, n_val=n_val, seperated=True, model_class=Group_lasso, model_param=model_param,
                             sampler_name='SGNHT', sampler_param=config)

print()
