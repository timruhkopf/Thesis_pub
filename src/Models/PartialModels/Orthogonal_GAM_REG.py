import torch.nn as nn

from src.Layer.Hidden import Hidden
from src.Layer.GAM.GAM import GAM
from src.Layer.GAM.GAM_fix_var import GAM_fix_var

from .OrthogonalBNN import OrthogonalBNN
from copy import deepcopy


class Orth_GAM_Reg(OrthogonalBNN):
    gam_layer = {
        'fix_nullspace': GAM,
        'fix_variance': GAM_fix_var
    }

    def __init__(self, hunits=[2, 1],
                 activation=nn.ReLU(), final_activation=nn.ReLU(),
                 bijected=True,
                 no_basis=20, gam_type='fix_nullspace'):
        """
        See OrthogonalBNN & Structured for base doc. The purpose of this class
        is to test the orthogonalisation & trainings behaviour. does the Pz_ortho
        work in the training procedure?
        Here, GAM is supposed to estimate the non linear additive part after the
        linear part is estimated by a plain regression.

        :param hunits:
        :param activation:
        :param final_activation:
        :param bijected: whether or not the Gam's variance parameter (only hierarchical parameter in OrthogonalBNN
        should be bijected: see GAM doc source code for details)
        :param no_basis: number J of spline basis functions B_J(x) gathered in Z
        """
        nn.Module.__init__(self)
        self.no_basis = no_basis
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation
        self.bijected = bijected

        # define the model components
        self.bnn = Hidden(bias=True, no_in=hunits[0], no_out=hunits[1], activation=activation)

        if gam_type == 'fix_variance':
            self.gam = self.gam_layer[gam_type](no_basis=no_basis, tau=1.)
        elif gam_type == 'fix_nullspace':
            self.gam = self.gam_layer[gam_type](no_basis=no_basis, bijected=bijected)

        self.reset_parameters()
        self.true_model = deepcopy(self.state_dict())

    def forward(self, X, Z):
        self.Px_orth = self.orth_projection(X[:, :1])
        self.Px_orth.detach()
        return self.bnn(X) + self.gam(self.Px_orth @ Z)

    def reset_parameters(self, **kwargs):
        """NOTICE, that since Ortho_GAM_REG uses OrthoBNN & StructBNN interface
        with self.bnn = Hidden and Hidden.reset_parameter does not have 'seperated'
        argument, this method has to be overwritten"""
        # Resample the BNN part
        self.bnn.reset_parameters()
        self.gam.reset_parameters(**kwargs)
        self.init_model = deepcopy(self.state_dict())

    # DEPREC: below when moving to BNN instead of xb ---------------------------
    def predict_states(self, X, Z, chain=None, *args):
        import pandas as pd
        df = pd.DataFrame()
        df_gam = pd.DataFrame()
        df_xb = pd.DataFrame()

        current = deepcopy(self.state_dict())

        def predict_single_state(self, name, state):
            self.load_state_dict(state)
            df[name] = self.forward(X, Z).detach().view(X.shape[0], ).numpy()
            # df_xb[name] = self.bnn.forward(X).detach().view(X.shape[0], ).numpy()
            df_xb[name] = (X[:, :1] @ self.bnn.W[0] + self.bnn.b).detach().numpy()  # fixme remove b if bias=False
            df_gam[name] = self.gam.forward(Z).detach().view(X.shape[0], ).numpy()

        topredict = [('current', current), ('true', self.true_model), ('init', self.init_model)]
        for name, state in topredict:
            predict_single_state(self, name, state)

        # predict chain
        if chain is not None:
            for i, c in enumerate(chain):
                predict_single_state(self, str(i), c)

            # return to current state
            self.load_state_dict(current)

        return df, df_gam, df_xb

    def plot(self, X, Z, y, path=None, **kwargs):
        import seaborn as sns
        import matplotlib.pyplot as plt
        df, df_gam, df_xb = self.predict_states(X, Z)

        plt1 = self.plot2d(X, y, df, **kwargs)

        if path is None:
            plt1.show()
        else:
            plt1.savefig('{}.pdf'.format(path), bbox_inches='tight')

        # predict merely the GAM part to ensure it does not deteriorate
        df_gam['X'] = X[:, 0].view(X.shape[0], ).numpy()  # explicitly the first var is modeled by gam
        df_gam = df_gam.melt('X', value_name='y')
        df_gam = df_gam.rename(columns={'variable': 'functions'})
        plt2 = self._plot1d(X, y=None, df=df_gam, **kwargs)

        if path is None:
            plt2.show()
        else:
            plt2.savefig('{}_GAM.pdf'.format(path), bbox_inches='tight')

        # predict merely the "bnn" part to ensure it does not deteriorate
        # To Deprec: just to avoid partial projection, with two indep. effects - as proxy:
        # def LS(X, y):
        #     return X @ torch.inverse(X.t() @ X) @ X.t() @ y
        #
        # df_xb.apply(lambda y: LS(X[:, 0].view(X[:, 0].shape[0], 1), torch.tensor(y)).view(X[:, 0].shape[0],).numpy())

        df_xb['X'] = X[:, 0].view(X.shape[0], ).numpy()  # explicitly the first var is modeled by xb
        df_xb = df_xb.melt('X', value_name='y')
        df_xb = df_xb.rename(columns={'variable': 'functions'})

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(hspace=0.5)
        # FIXME: since only X[:, 0] is used, (and X[:, 1] is more or less orth. to X[:, 0], the predictions of y
        #   are influenced by  X[:, 1], creating this randomness for each x position in the plots:
        sns.lineplot(x='X', y='y', hue='functions', data=df_xb[df_xb['functions'] != 'current'], ax=ax)
        sns.lineplot('X', y='y', color='red', alpha=0.5, data=df_xb[df_xb['functions'] == 'current'],
                     ax=ax, label='current')

        plt2 = fig

        if path is None:
            plt2.show()
        else:
            plt2.savefig('{}_Xb.pdf'.format(path), bbox_inches='tight')


if __name__ == '__main__':
    h = Orth_GAM_Reg(hunits=[2, 1], gam_type='fix_variance', no_basis=10,
                     activation=nn.Identity(), final_activation=nn.Identity())
    n = 1000
    X, Z, y = h.sample_model(n)
    h.reset_parameters()

    import matplotlib

    matplotlib.use('TkAgg')
    h.plot(X, Z, y, path=None)

    # import matplotlib
    # matplotlib.use('TkAgg')
    # h.plot(X, Z, y, chain=chain)
    # print()
    #

    # check sampling ability.
    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 10000

    trainset = TensorDataset(X, Z, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # torch.autograd.set_detect_anomaly(True)
    h.reset_parameters()
    sampler = Sampler(h, epsilon=0.0002, L=1)

    sampler.sample(trainloader, burn_in, n_samples)

    print(h.true_model)
    print(h.state_dict())
    print(h.init_model)
    h.plot(X, Z, y)
    print(sampler.chain_mat)
    import random

    h.plot(X[:100], Z[:100], y[:100], chain=random.sample(sampler.chain, 100),
           **{'title': 'orthoBNN'})

    h.plot(X[:100], Z[:100], y[:100], chain=list(sampler.chain[-1:]))
    h.plot(X[:100], Z[:100], y[:100], chain=list(h.init_model))
