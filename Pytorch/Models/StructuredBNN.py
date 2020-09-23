import torch
import torch.nn as nn
import torch.distributions as td
import pandas as pd

from Tensorflow.Effects.bspline import get_design
from Pytorch.Models.GAM import GAM
from Pytorch.Models.GAM_Wood import GAM_Wood
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN
from Pytorch.Util.ModelUtil import Model_util

from copy import deepcopy

import matplotlib

matplotlib.use('agg')


class StructuredBNN(nn.Module, Model_util):
    gam_layer = {
        'fix_nullspace': GAM,
        'double_penalty': GAM_Wood
    }

    def __init__(self, hunits=[2, 3, 1], shrinkage='glasso',
                 activation=nn.ReLU(), final_activation=nn.ReLU(),
                 seperated=True, bijected=True, alpha_type='cdf',
                 no_basis=20):
        super().__init__()
        self.no_basis = no_basis
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation
        self.seperated = seperated
        self.bijected = bijected

        # define the model components
        self.bnn = ShrinkageBNN(hunits, activation, final_activation, shrinkage,
                                seperated=seperated, bijected=bijected, prior='flat')
        self.gam = self.gam_layer['fix_nullspace'](no_basis=no_basis, bijected=bijected)

        # notice, that alpha automatically adapts to bijection of tau!
        self.alpha = {  # all of the below are properties!
            'cdf': lambda: self.bnn.layers[0].alpha,
            'Be': lambda: self.bnn.layers[0].alpha_probab,
            'constant': lambda: self.bnn.layers[0].alpha_const
        }[alpha_type]

        self.reset_parameters()

    def forward(self, X, Z):
        return self.bnn.forward(X) + self.alpha() * self.gam.forward(Z)

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        return self.bnn.prior_log_prob() + sum(self.gam.prior_log_prob())

    def likelihood(self, X, Z, sigma=torch.tensor(1.)):
        """:param X: joint matrix, first cols are classic design, the following are
        the bspline extension desing marix"""
        return td.Normal(self.forward(X, Z), sigma)

    def log_prob(self, X, Z, y):
        self.update_distributions()
        return self.prior_log_prob() + self.likelihood(X, Z).log_prob(y).sum()

    def reset_parameters(self, seperated=False, **kwargs):
        # Resample the BNN part
        self.bnn.reset_parameters(seperated)
        self.gam.reset_parameters(**kwargs)
        self.init_model = deepcopy(self.state_dict())

    def update_distributions(self):
        self.gam.update_distributions()
        self.bnn.update_distributions()

    @torch.no_grad()
    def _predict_states(self, X, Z, chain=None):
        """
        predict f(X) based on true, current and chain's states.
        a submethod of plotting

        :param y: Torch.Tensor
        :param chain: list of state_dicts
        :param path: string: system path where to save to
        :param *args: Torch.Tensor(s) upon which the predictions are made using
        the model's forward method. Normal models require this to be the
        design matrix X. In the case of structured BNN this must be X and Z

        :return:
        """

        # (1) PREDICT functions -----------------------------------------------
        df = pd.DataFrame()
        df_gam = pd.DataFrame()

        # predict current state
        df['current'] = self.forward(X, Z).view(X.shape[0], ).numpy()
        df_gam['current'] = self.gam.forward(Z).view(X.shape[0], ).numpy()
        current = deepcopy(self.state_dict())

        # predict true model
        self.load_state_dict(self.true_model)
        df['true'] = self.forward(X, Z).view(X.shape[0], ).numpy()
        df_gam['true'] = self.gam.forward(Z).view(X.shape[0], ).numpy()

        # predict true model
        if hasattr(self, 'init_model'):
            self.load_state_dict(self.init_model)
            df['init'] = self.forward(X, Z).view(X.shape[0], ).numpy()
            df_gam['init'] = self.gam.forward(Z).view(X.shape[0], ).numpy()

        # predict chain
        if chain is not None:
            for i, c in enumerate(chain):
                self.load_state_dict(c)
                df[str(i)] = self.forward(X, Z).view(X.shape[0], ).numpy()
                df_gam[str(i)] = self.gam.forward(Z).view(X.shape[0], ).numpy()

            # return to current state
            self.load_state_dict(current)

        return df, df_gam

    def plot(self, X, Z, y, chain=None, path=None, **kwargs):
        # predict the entire model
        df0, df1 = self._predict_states(X, Z, chain)  # df1 is gam only prediction

        plt1 = self.plot2d(X, y, df0, **kwargs)

        if path is None:
            plt1.show()

        else:
            plt1.savefig('{}.png'.format(path), bbox_inches='tight')

        # predict merely the GAM part to ensure it does not deteriorate
        df1['X'] = X[:, 0].view(X.shape[0], ).numpy()  # explicitly the first var is modeled by gam
        df1 = df1.melt('X', value_name='y')
        df1 = df1.rename(columns={'variable': 'functions'})
        plt2 = self._plot1d(X, y=None, df=df1, **kwargs)

        if path is None:

            plt2.show()
        else:

            plt2.savefig('{}_GAM.png'.format(path), bbox_inches='tight')

    @staticmethod
    def check_chain(chain):
        return Model_util.check_chain_seq(chain)

if __name__ == '__main__':
    from copy import deepcopy

    # parametrize the basic datamodel
    no_basis = 20
    no_in = 2
    no_out = 1
    n = 1000

    # generate data
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)
    Z = torch.tensor(get_design(X[:, 0].numpy(), degree=2, no_basis=no_basis),
                     dtype=torch.float32)

    # X_joint = torch.cat([X, Z], dim=1)
    # X_joint.requires_grad_(True)

    h = StructuredBNN(hunits=[2, 10, 1], bijected=True)
    h.reset_parameters(True)
    h.true_model = deepcopy(h.state_dict())
    # L = h(X).sum()
    # L.backward()

    # h.alpha
    # h.h1.layers

    y = h.likelihood(X, Z).sample()
    # h.plot(X, Z, y) # Fixme: canvas fails

    # check sampling ability.
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, Z, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # h.reset_parameters(False)
    # glasso.plot(X_joint, y)

    torch.autograd.set_detect_anomaly(True)
    h.reset_parameters()
    sampler = Sampler(h, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    print(sampler.chain_mat)
    print(sampler.model.a)
    import random

    h.plot(X[:100], Z[:100], y[:100], chain=random.sample(sampler.chain, 100),
           **{'title': 'structuredBNN'})

    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA

    num_samples = 1000
    step_size = 0.01
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    L = 24
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    h.reset_parameters()
    sgnht = SGNHT(h, trainloader,
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  L=L,
                  num_chains=num_chains)
    sgnht.sample()

    print()

    # [(name, p.grad) for name, p in self.model.named_parameters()]
