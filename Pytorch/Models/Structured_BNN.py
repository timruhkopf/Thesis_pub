import torch
import torch.nn as nn
import torch.distributions as td
import pandas as pd

from Pytorch.Models.GAM import GAM
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN
from Tensorflow.Effects.bspline import get_design


class Structured_BNN(ShrinkageBNN):

    def __init__(self, gam_param={'xgrid': (0, 10, 0.5), 'order': 1, 'no_basis': 20,
                                  'no_out': 1, 'activation': nn.Identity()},
                 hunits=[2, 10, 1], shrinkage='glasso',
                 activation=nn.ReLU(), final_activation=nn.ReLU(),
                 seperated=True, bijected=True):

        nn.Module.__init__(self)

        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation
        self.seperated = seperated
        self.bijected = bijected

        # define_model (components)
        self.gam = GAM(**gam_param)
        self.bnn = ShrinkageBNN(hunits, activation, final_activation, shrinkage, seperated, bijected)

        self.reset_parameters()

    def forward(self, X, Z, alpha=None):
        """

        :param X: Designmatrix (batch shaped tensor)
        :param Z: Bspline Design matrix expansion of the first column in X
        (batch shaped tensor)
        :param alpha:
        :return:
        """
        if alpha is None:
            alpha = self.alpha

        if alpha < 0. or alpha > 1.:
            raise ValueError('alpha exceeded (0,1) interval')

        return self.bnn.forward(X) + alpha * self.gam(Z)

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        bnn_log_prob = sum([h.prior_log_prob().sum() for h in self.bnn.layers])
        return bnn_log_prob + sum(self.gam.prior_log_prob())

    def likelihood(self, X, Z, sigma=torch.tensor(1.)):
        return td.Normal(self.forward(X, Z, self.alpha), sigma)

    def log_prob(self, X, Z, y):
        return self.prior_log_prob() + self.likelihood(X, Z).log_prob(y).sum()

    def reset_parameters(self, seperated=False, **kwargs):
        # Resample the BNN part
        self.bnn.reset_parameters(seperated)
        self.gam.reset_parameters(**kwargs)

    def update_distributions(self):
        self.gam.update_distributions()
        self.bnn.update_distributions()

    @property
    def alpha(self):
        return self.bnn.layers[0].alpha  # TODO - Try alternative version with Bernoulli

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

        # predict merely the GAM part to ensure it does not deteriorate

        df1['X'] = X[:, 0].view(X.shape[0], ).numpy()  # explicitly the first var is modeled by gam
        df1 = df1.melt('X', value_name='y')
        df1 = df1.rename(columns={'variable': 'functions'})
        plt2 = self._plot1d(X, y=None, df=df1, **kwargs)

        if path is None:
            plt1.show()
            plt2.show()
        else:
            plt1.savefig('{}.png'.format(path), bbox_inches='tight')
            plt2.savefig('{}_GAM.png'.format(path), bbox_inches='tight')


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

    glasso = Structured_BNN(gam_param={'xgrid': (0, 10, 0.5), 'order': 1, 'no_basis': 20,
                                       'no_out': 1, 'activation': nn.Identity()},
                            hunits=[no_in, 10, no_out], shrinkage='glasso',
                            activation=nn.ReLU(), final_activation=nn.ReLU())
    glasso.true_model = deepcopy(glasso.state_dict())
    y = glasso.likelihood(X, Z).sample()

    # check forward path
    glasso(X, Z)  # == glasso.forward(X, Z)

    # check resampling
    print(glasso.log_prob(X, Z, y))
    glasso.reset_parameters()
    print(glasso.log_prob(X, Z, y))


    # check log_prob(X, Z, y)
    print(glasso.prior_log_prob())
    print(glasso.log_prob(X, Z, y))

    # check plotting with current and true model
    # glasso.plot(X, Z, y)

    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, Z, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']
    sampler = Sampler(glasso, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    # check sampling
    # LUDWIG SAMPLER DOEW NOT AS EASILY ALLOW (X, Z, y) or trainloader of this form
    # from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    #
    # num_samples = 1000
    #
    # step_size = 0.01
    # num_steps = 5000  # <-------------- important
    # pretrain = False
    # tune = False
    # burn_in = 2000
    # # num_chains 		type=int, 	default=1
    # num_chains = 1  # os.cpu_count() - 1
    # batch_size = 50
    # L = 24
    # val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    # val_prediction_steps = 50
    # val_converge_criterion = 20
    # val_per_epoch = 200
    #
    # glasso.reset_parameters()
    # sgnht = SGNHT(glasso, X, y, X.shape[0], Z,
    #               step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
    #               L=L,
    #               num_chains=num_chains)
    # sgnht.sample()
    # print(sgnht.chain)

    import random

    glasso.plot(X, y, chain=random.sample(sgnht.chain, 100),
                **{'title': 'G-lasso'})

    # look at the chain of alphas (XOR decision)
    chain = None
    tau = torch.stack([v for odict in chain for k, v in odict.items() if 'tau' == k[-3:]], axis=0)
    true_tau = glasso.true_model['tau']

    if glasso.bijected:
        tau = glasso.bnn.layers[0].dist['tau'].transforms[0]._inverse(tau)
        true_tau = glasso.bnn.layers[0].dist['tau'].transforms[0]._inverse(true_tau)
    1 - glasso.bnn.layers[0].dist['alpha'].cdf(tau)

    # check sampling ability.
