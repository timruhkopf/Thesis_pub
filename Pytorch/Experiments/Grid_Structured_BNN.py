import os
import torch
import torch.nn as nn
import torch.distributions as td
import pandas as pd
from inspect import getfullargspec
from copy import deepcopy
import matplotlib.pyplot as plt

from Tensorflow.Effects.bspline import get_design

from Pytorch.Util.GridUtil import Grid
from Pytorch.Experiments.SAMPLER_GRID import SAMPLER_GRID
from Pytorch.Experiments.Grid_Layer import Layer_Grid


class Structured_BNN_Grid(Grid, SAMPLER_GRID):

    def main(self, n, n_val, model_class, model_param, sampler_name, sampler_param,
             seperated=False):
        self.basename = self.pathresults + \
                        '{}_{}'.format(str(model_class.__class__), sampler_name) + \
                        self.hash
        self.model_class = model_class

        # set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        self.set_up_datamodel(n, n_val, no_in=model_param['hunits'][0],
                              model_class=model_class, seperated=seperated, model_param=model_param)


        self.set_up_sampler(self.model, sampler_name, sampler_param)
        X_val, Z_val, y_val = self.data_val
        metrics = self.evaluate_model(X_val, Z_val, y_val)

        return metrics  # to be written on Grid's result table

    def set_up_datamodel(self, n, n_val, no_in, model_class, seperated, model_param):

        no_basis = 20
        no_in = 2
        no_out = 1
        n = 1000

        # generate data
        X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, no_in)
        X_val = X_dist.sample(torch.Size([n_val]))
        Z = torch.tensor(get_design(X[:, 0].numpy(), degree=2, no_basis=no_basis),
                         dtype=torch.float32)
        Z_val = torch.tensor(get_design(X_val[:, 0].numpy(), degree=2, no_basis=no_basis),
                             dtype=torch.float32)

        self.model = model_class(**model_param)

        if 'seperated' in getfullargspec(self.model.reset_parameters).args:
            self.model.reset_parameters(seperated)
        else:
            self.model.reset_parameters()

        self.model.true_model = deepcopy(self.model.state_dict())
        y = self.model.likelihood(X, Z).sample()
        y_val = self.model.likelihood(X_val, Z_val).sample()

        # check reset_parameters &  check prior_log_prob
        self.data = X, Z, y
        self.data_val = X_val, Z_val, y_val

        # save the true state & true model's performance on validation
        self.model.true_model = deepcopy(self.model.state_dict())
        self.model.val_logprob = self.model.log_prob(X_val, Z_val, y_val)
        with torch.no_grad():
            self.model.val_MSE = torch.nn.MSELoss()(self.model.forward(X_val, Z_val), y_val)

    def evaluate_model(self, X_val, Z_val, y_val):

        # ONLY REAL CHANGE HERE in comparison to GAM Grid's evaluate_model
        # is that there are two design matrices - since no de boor like streaming of the data
        # is available yet.
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
            X_val.to(self.device)
            Z_val.to(self.device)
            y_val.to(self.device)

            # Ludwig's extrawurst
            from Pytorch.Samplers.LudwigWinkler import LudwigWinkler
            if issubclass(type(self.sampler), LudwigWinkler):
                # since Ludwig required special log_prob to begin with
                sampler.model.log_prob1 = lambda X, Z, y: sampler.model.log_prob(X, Z, y)['log_prob']
            else:
                sampler.model.log_prob1 = sampler.model.log_prob

            subset_chain = random.sample(sampler.chain, subsample)
            sampler.val_MSE_chain = torch.Tensor(subsample, )
            sampler.val_logprobs = torch.Tensor(subsample, )

            for i, c in enumerate(subset_chain):
                sampler.model.load_state_dict(c)
                sampler.val_logprobs[i] = sampler.model.log_prob1(X_val, Z_val, y_val)

                pred = sampler.model.forward(X_val, Z_val)

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
                'avg_MSE_diff': mse_diff.detach().numpy(),
                'avg_log_prob_diff': log_diff.detach().numpy()}


if __name__ == '__main__':
    import torch.nn as nn
    from Pytorch.Models.StructuredBNN import StructuredBNN

    run = 'Structured_BNN_SGNHT'  # ---------------------------------------------
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    grid = Structured_BNN_Grid(root)
    prelim_configs = grid.grid_exec_SGNHT(steps=1000, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                       final_activation=nn.Identity(), shrinkage='glasso', no_basis=20,
                       seperated=True, bijected=True)

    for config in prelim_configs:
        grid.main(n=n, n_val=n_val, seperated=True,
                  model_class=StructuredBNN, model_param=model_param,
                  sampler_name='SGNHT', sampler_param=config)
