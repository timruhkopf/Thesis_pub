import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as td


from Pytorch.Models.GAM import GAM
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN
from Tensorflow.Effects.bspline import get_design


class Structured_BNN(nn.Module):

    def __init__(self, gam_param={'xgrid': (0, 10, 0.5), 'order': 1, 'no_basis': 20,
                                  'no_out': 1, 'activation': nn.Identity()},
                 hunits=[2, 10, 1], shrinkage='glasso',
                 activation=nn.ReLU(), final_activation=nn.ReLU(),
                 seperated=True, bijected=True):

        nn.Module.__init__(self)

        self.no_basis = gam_param['no_basis']
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation
        self.seperated = seperated
        self.bijected = bijected

        # define_model (components)
        self.bnn = ShrinkageBNN(hunits, activation, final_activation, shrinkage, seperated, bijected)
        self.gam = GAM(**gam_param)
        self.reset_parameters()

    def split_design(self, X):
        """LudwigWinkler's sampler does not readily allow for the log_prob to recieve
        two input tensors. To avoid any issue with this, this function splits a joined
        designmatrix for calculation purposes in the designmatrix X and a
        Bspline-Designmatrix Z (wich describes the Bspline space extention of
        the first column in X)"""
        return X[:, :-self.no_basis], X[:, -self.no_basis:]

    def forward(self, X):
        """

        :param X: Designmatrix (batch shaped tensor)
        :param Z: Bspline Design matrix expansion of the first column in X
        (batch shaped tensor)
        :return:
        """
        # X, Z = self.split_design(X)
        return self.bnn.forward(X[:, :-self.no_basis]) + self.alpha * self.gam(X[:, -self.no_basis:])

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        return self.bnn.prior_log_prob() + sum(self.gam.prior_log_prob())

    def likelihood(self, X, sigma=torch.tensor(1.)):
        """:param X: joint matrix, first cols are classic design, the following are
        the bspline extension desing marix"""
        return td.Normal(self.forward(X), sigma)

    def log_prob(self, X, y):
        return self.prior_log_prob() + self.likelihood(X).log_prob(y).sum()

    def reset_parameters(self, seperated=False, **kwargs):
        # Resample the BNN part
        self.bnn.reset_parameters(seperated)
        self.gam.reset_parameters(**kwargs)

    def update_distributions(self):
        self.gam.update_distributions()
        self.bnn.update_distributions()

    @property
    def alpha(self):
        tau = self.bnn.layers[0].tau.clone().detach()
        tau.requires_grad_(False)
        return tau

    # @property
    # def alpha(self):
    #     return self.bnn.layers[0].alpha  # TODO - Try alternative version with Bernoulli

    @torch.no_grad()
    def _predict_states(self, X, chain=None):
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
        X_split, Z = self.split_design(X)
        # (1) PREDICT functions -----------------------------------------------
        df = pd.DataFrame()
        df_gam = pd.DataFrame()

        # predict current state
        df['current'] = self.forward(X).view(X.shape[0], ).numpy()
        df_gam['current'] = self.gam.forward(Z).view(X.shape[0], ).numpy()
        current = deepcopy(self.state_dict())

        # predict true model
        self.load_state_dict(self.true_model)
        df['true'] = self.forward(X).view(X.shape[0], ).numpy()
        df_gam['true'] = self.gam.forward(Z).view(X.shape[0], ).numpy()

        # predict chain
        if chain is not None:
            for i, c in enumerate(chain):
                self.load_state_dict(c)
                df[str(i)] = self.forward(X).view(X.shape[0], ).numpy()
                df_gam[str(i)] = self.gam.forward(Z).view(X.shape[0], ).numpy()

            # return to current state
            self.load_state_dict(current)

        return df, df_gam

    def plot(self, X, y, chain=None, path=None, **kwargs):
        # predict the entire model
        df0, df1 = self._predict_states(X, chain)  # df1 is gam only prediction

        X, Z = self.split_design(X)
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

    X_joint = torch.cat([X, Z], dim=1)
    X_joint.requires_grad_(True)

    glasso = Structured_BNN(gam_param={'xgrid': (0, 10, 0.5), 'order': 1, 'no_basis': 20,
                                       'no_out': 1, 'activation': nn.Identity()},
                            hunits=[no_in, 10, no_out], shrinkage='glasso',
                            activation=nn.ReLU(), final_activation=nn.ReLU(), bijected=True)
    glasso.reset_parameters(True)
    glasso.true_model = deepcopy(glasso.state_dict())
    y = glasso.likelihood(X_joint).sample()

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter()
    # writer.add_graph(glasso, input_to_model=X_joint, verbose=True) # FAILS unexpectedly
    # writer.close()

    # glasso.plot(X_joint, y)

    # check forward path
    glasso(X_joint)  # == glasso.forward(X, Z)

    # check resampling
    print(glasso.log_prob(X_joint, y))
    glasso.reset_parameters()
    print(glasso.log_prob(X_joint, y))


    # check log_prob(X, Z, y)
    print(glasso.prior_log_prob())
    print(glasso.log_prob(X_joint, y))

    # check plotting with current and true model
    # glasso.plot(X_joint, y)

    # check sampling ability.
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X_joint, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    glasso.reset_parameters(False)
    # glasso.plot(X_joint, y)

    torch.autograd.set_detect_anomaly(True)
    sampler = Sampler(glasso, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    # check sampling
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

    glasso.reset_parameters()
    sgnht = SGNHT(glasso, X_joint, y, X.shape[0],
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  L=L,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)

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
