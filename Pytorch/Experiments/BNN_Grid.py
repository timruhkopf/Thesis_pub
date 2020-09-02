import os
import torch
import matplotlib.pyplot as plt

from Pytorch.Util.GridUtil import Grid
from Pytorch.Experiments.SAMPLER_GRID import SAMPLER_GRID
from Pytorch.Experiments.Layer_Grid import Layer_Grid
from Pytorch.Experiments.GAM_Grid import GAM_Grid


class BNN_Grid(Grid, SAMPLER_GRID):
    set_up_datamodel = Layer_Grid.set_up_datamodel

    def main(self, n, n_val, model_class, model_param, sampler_name, sampler_param, seperated=False):
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
        X, y = self.data

        self.set_up_sampler(self.model, X, y, sampler_name, sampler_param)
        X_val, y_val = self.data_val
        metrics = self.evaluate_model(X_val, y_val)

        return metrics  # to be written on Grid's result table

    def evaluate_model(self, X_val, y_val):

        metrics = GAM_Grid.evaluate_model(self, X_val=X_val, Z_val=X_val, y_val=y_val)

        if 'tau' in list(self.model.layers[0].parameters_dict):
            # analyse the distribution of tau
            # be aware: if all layers are aggregated in model.layers, the parameters naming convention
            # is 'layers.0.tau' for the zero layer parameter tau. if model is a single layer,
            # its parameters are explicitly named! e.g. 'tau'
            tau = torch.stack([v for odict in self.sampler.chain for k, v in odict.items() if  'tau' == k[-3:]], axis=0)
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

    import torch.nn as nn

    # (0) BNN ------------------------------------------------------------------
    # SUCCESSFULLY RUNNING
    from Pytorch.Models.BNN import BNN

    run = 'BNN_test_grid_exec_SGNHT'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    bnn_unittest = BNN_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGNHT(steps=1000, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(hunits=[1, 10, 5, 1], activation=nn.ReLU(),
                       final_activation=nn.Identity(), heteroscedast=False)

    for config in prelim_configs:
        bnn_unittest.main(n=n, n_val=n_val, seperated=True, model_class=BNN, model_param=model_param,
                          sampler_name='SGNHT', sampler_param=config)

    # (1.0) shrinkageBNN RHMC -------------------------------------------------
    from Pytorch.Models.ShrinkageBNN import ShrinkageBNN

    # FIXME: consistently fails.
    run = 'shrinkageBNN_test_grid_exec_RHMC'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    bnn_shrinkage_unittest = BNN_Grid(root)
    prelim_configs = bnn_shrinkage_unittest.grid_exec_RHMC(steps=1000)

    n = 1000
    n_val = 100
    model_param = dict(hunits=[2, 3, 1], activation=nn.ReLU(),
                       final_activation=nn.Identity(),
                       shrinkage='glasso', seperated=True, bijected=True,
                       heteroscedast=False)

    for config in prelim_configs:
        bnn_shrinkage_unittest.main(
            n=n, n_val=n_val, seperated=True,
            model_class=ShrinkageBNN, model_param=model_param,
            sampler_name='RHMC', sampler_param=config)

    # (1.1) shrinkageBNN ---------------------------------------------------------
    from Pytorch.Models.ShrinkageBNN import ShrinkageBNN

    # FIXME: consistently fails.

    run = 'shrinkageBNN_test_grid_exec_SGRLD'
    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    bnn_shrinkage_unittest = BNN_Grid(root)
    prelim_configs = bnn_shrinkage_unittest.grid_exec_SGRLD(steps=100, batch_size=100)

    n = 1000
    n_val = 100
    model_param = dict(hunits=[2, 3, 1], activation=nn.ReLU(),
                       final_activation=nn.Identity(),
                       shrinkage='glasso', seperated=True, bijected=True,
                       heteroscedast=False)

    for config in prelim_configs:
        bnn_shrinkage_unittest.main(
            n=n, n_val=n_val, seperated=True,
            model_class=ShrinkageBNN, model_param=model_param,
            sampler_name='SGRLD', sampler_param=config)

    print()

    # (2) Structured BNN --------------------------------------------------------
    ## TODO Move this to a seperate Grid class
    # from Pytorch.Models.structured_BNN import Structured_BNN
    #
    # run = 'structuredBNN_test_grid_exec_SGNHT'
    # root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    #     os.getcwd() + '/Results/{}/'.format(run)
    #
    # bnn_structured_unittest = BNN_Grid(root)
    # prelim_configs = bnn_structured_unittest.grid_exec_SGNHT(steps=1000, batch_size=100)
    #
    # n = 10000
    # n_val = 100
    # model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
    #                    final_activation=nn.Identity(),
    #                    shrinkage='glasso', seperated=False, heteroscedast=False)
    #
    # for config in prelim_configs:
    #     bnn_structured_unittest.main(
    #         n=n, n_val=n_val, seperated=True,
    #         model_class=Structured_BNN, model_param=model_param,
    #         sampler_name='SGNHT', sampler_param=config)
    # model_param = dict(
    #     gam_param={'xgrid': (0, 10, 0.5), 'order': 1, 'no_basis': 20,
    #                'no_out': 1, 'activation': nn.Identity()},
    #     hunits=[2, 10, 1], shrinkage='glasso', activation=nn.ReLU(), final_activation=nn.ReLU())
    #

    print()
