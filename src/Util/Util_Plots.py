import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.tri as mtri  # for trisurface with irregular grid
from copy import deepcopy


class Util_plots:
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
        df['current'] = self.forward(*args).view(X.shape[0], ).numpy()
        current = deepcopy(self.state_dict())

        # predict true model
        self.load_state_dict(self.true_model)
        df['true'] = self.forward(*args).view(X.shape[0], ).numpy()

        if hasattr(self, 'init_model'):
            self.load_state_dict(self.init_model)
            df['init'] = self.forward(*args).view(X.shape[0], ).numpy()

        # predict chain
        if chain is not None:
            for i, c in enumerate(chain):
                self.load_state_dict(c)
                df[str(i)] = self.forward(*args).view(X.shape[0], ).numpy()

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
            # FIXME: for 2d plots if saved to path, ensure they store the data required
            # such that the plot can easily restored & is movable again
            plt.savefig('{}.pdf'.format(path), bbox_inches='tight')

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

        sns.lineplot('X', y='y', hue='functions', alpha=0.5, data=df[df['functions'] != 'current'], ax=ax)
        sns.lineplot('X', y='y', color='red', alpha=0.5, data=df[df['functions'] == 'current'], ax=ax, label='current')
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
                    marker='.', s=10, c="black", alpha=0.5, label='Observed data')
        ax1.view_init(elev=40, azim=-45)

        ax1.scatter(X[:, 0], X[:, 1], df['current'],
                    marker='.', s=7, color='red', alpha=0.3, label='current')

        import matplotlib.cm as cm
        colors = cm.rainbow(torch.linspace(0, 1, len(df)).numpy())
        for (k, v), c in zip(df.items(), colors):

            if k != 'true' and k != 'current':
                ax1.scatter(X[:, 0], X[:, 1], v,
                            marker='.', s=7, color=c, alpha=0.3, label=k)

        ax1.legend(loc='upper left')
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
