import torch
import torch.distributions as td
from Pytorch.Grid.Grid_Layout import GRID_Layout


class GRID_Layout_GAM(GRID_Layout):
    def set_up_data(self, n, n_val, model_param, batch_size):
        from torch.utils.data import TensorDataset, DataLoader
        from Pytorch.Util.Util_bspline import get_design

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
        from Pytorch.Util.Util_bspline import get_design

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

    import os

    # (GAM) -----------------------------------------------------------------

    run = 'GAM_SGRLD1'

    root = os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)
    gam_unittest = GRID_Layout_GAM(root)

    from Pytorch.Layer.GAM import GAM

    prelim_configs = gam_unittest.grid_exec_SGRLD(steps=100, batch_size=100)  # TODO: EXTEND THE GRID
    next(prelim_configs)

    sampler_name = 'SGRLD'
    for prelim_config in prelim_configs:
        gam_unittest.main(n=1000, n_val=100, sampler_name=sampler_name, model_class=GAM,
                          model_param=dict(no_basis=20, bijected=True),
                          sampler_param=prelim_config, seperated=True)

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
