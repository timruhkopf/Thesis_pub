import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.tri as mtri  # for trisurface with irregular grid
from copy import deepcopy


class Plots:
    @torch.no_grad()
    def predict_states(self, chain=None, *args):
        """
        predict f(X) based on true, current and chain's states

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
        X = args[0]
        # predict current state
        df['current'] = self.forward(*args).view(X.shape[0],).numpy()
        current = deepcopy(self.state_dict())

        # predict true model
        self.load_state_dict(self.true_model)
        df['true'] = self.forward(*args).view(X.shape[0],).numpy()

        # predict chain
        if chain is not None:
            for i, c in enumerate(chain):
                self.load_state_dict(c)
                df[str(i)] = self.forward(*args).view(X.shape[0],).numpy()

            # return to current state
            self.load_state_dict(current)

        return df

    def plot(self, X, y, chain=None, path=None, **kwargs):
        """

        :param X:
        :param y:
        :param chain: list of state_dicts
        :param path: string
        :param kwargs: additional arguments for respective plot function _plot1d/2d
        :return:
        """
        # (2) MELT the frame for plotting format
        df0 = self.predict_states(chain, X)


        if X.shape[1]==1:
            df0['X'] = X.view(X.shape[0], ).numpy()
            df1 = df0.melt('X', value_name='y')
            df1 = df1.rename(columns={'variable': 'functions'})

            plt = self._plot1d(X, y, df1, **kwargs)
        elif X.shape[1] == 2:
            plt = self.plot2d(X, y, df0, **kwargs)

        else:
            print('Could not plot function, as input dim is >2')



        if path is None:
            plt.show()
        else:
            plt.savefig('{}.png'.format(path), bbox_inches='tight')

    def _plot1d(self, X, y=None, df=None, title=''):
        """

        :param X:
        :param y:
        :param df: melted pandas df, containing all functions' predictions
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(hspace=0.5)
        if y is not None:
            sns.scatterplot(
                x=torch.reshape(X, (X.shape[0],)).numpy(),
                y=torch.reshape(y, (y.shape[0],)).numpy(), ax=ax)

        sns.lineplot('X', y='y', hue='functions', alpha=0.5, data=df, ax=ax)
        plt.title('{}'.format(title))
        return fig

    def plot2d(self, X, y, df, title=''):
        X, y = X.numpy(), y.numpy()
        triang = triangulate_remove_artifacts(X[:, 0], X[:, 1], -9.9, 9.9, -9.9, 9.9, plot=False)


        fig = plt.figure()
        plt.title('{}'.format(title))
        plt.axis('off')

        # if multi_subplots:
        #     rows = int(torch.ceil(torch.sqrt(torch.tensor(len(df.keys()), dtype=torch.float32))).numpy())
        #     ax1 = fig.add_subplot(rows, rows, 1, projection='3d')
        # else:
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.text2D(0.05, 0.95, title, transform=ax1.transAxes)

        # ground truth model & data
        ax1.plot_trisurf(triang, df['true'],
                         cmap='jet', alpha=0.4)
        ax1.scatter(X[:, 0], X[:, 1], y,
                    marker='.', s=10, c="black", alpha=0.5)
        ax1.view_init(elev=40, azim=-45)

        import matplotlib.cm as cm
        colors = cm.rainbow(torch.linspace(0, 1, len(df)).numpy())
        for (k, v), c in zip(df.items(), colors):

            if k != 'true':
                ax1.scatter(X[:, 0], X[:, 1], v,
                            marker='.', s=7, color=c, alpha=0.3)
        return plt


def triangulate_remove_artifacts(x, y, xl=0.1, xu=9.9, yl=0.1, yu=9.9, plot=True):
    """
    remove artifacts from triangulation method with some boundary
    :param x:
    :param y:
    :param xl:
    :param xu:
    :param yl:
    :param yu:
    :param plot:
    :return:
    """
    triang = mtri.Triangulation(x, y)
    isBad = np.where((x < xl) | (x > xu) | (y < yl) | (y > yu), True, False)

    mask = np.any(isBad[triang.triangles], axis=1)
    triang.set_mask(mask)

    if plot:
        # look at the triangulation result
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.triplot(triang, c="#D3D3D3", marker='.',
                   markerfacecolor="#DC143C", markeredgecolor="black",
                   markersize=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    return triang

import torch.distributions as td
import torch
import matplotlib.pyplot as plt
def plot_distribution(dist, support=torch.linspace(0., 10., 100)):
    """plotting a distribution on support, as a tool to inspect the shape of a dist.
    since dist.pdf is not available exp(dist.log_prob(support)) is displayed for shape info"""
    y = torch.exp(dist.log_prob(support))
    plt.plot(support.detach().numpy(), y.detach().numpy())
    plt.show()


if __name__ == '__main__':

    plot_distribution(td.Gamma(0.3, 0.1))

    from Pytorch.Layer.Hidden import Hidden
    import torch.distributions as td
    import torch.nn as nn
    no_in = 1
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    reg.true_model = reg.state_dict()

    # reg.W = reg.W_.data
    # reg.b = reg.b_.data
    reg.forward(X=torch.ones(100, no_in))
    reg.prior_log_prob()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()

    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA

    step_size = 0.01
    num_steps = 500  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 200
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    hmc_traj_length = 24
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    reg.reset_parameters()
    sgnht = SGNHT(reg, X, y, X.shape[0],
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  hmc_traj_length=hmc_traj_length,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)


    if no_in == 1:
        kwargs = {}
    elif no_in == 2:
        kwargs = {'title': 'SOMETHING'}
    reg.plot(X, y, chain=sgnht.chain, **kwargs)
