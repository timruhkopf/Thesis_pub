from Tensorflow.Effects.bspline import get_design

from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp

# for plotting of functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Tensorflow.Bayesian.plot1d import triangulate_remove_artifacts

from Tensorflow.Util import setkwargs

tfd = tfp.distributions
tfb = tfp.bijectors


class BNN_GAM:
    # parameters to define in the descendent class
    parameters_list = list()
    bijectors = list()
    gam = None
    bnn = None

    def likelihood_model(self, X, Z, bnn_param, gam_param):
        # GAM & BNN prediction sum as mu of Normal.
        # CAREFULL! GAM VALUE is "support" of prediction.
        #  BNN calculates on the residual information (if additional interactions exist)

        return tfd.JointDistributionNamed(OrderedDict(
            sigma=tfd.InverseGamma(1., 1.),
            y=lambda sigma: tfd.Sample(
                tfd.Normal(loc=self.forward(X, Z, bnn_param, gam_param),
                           scale=sigma))))

    def forward(self, X, Z, bnn_param, gam_param):
        # bnn.forward leverages each PriorModels dense method, which in this thesis
        # consist only of W, b or W alone. (essentially deterministic Neural Net parameter)
        nn_param = [{k: v for k, v in hparam.items() if k in ['W', 'b']}
                    for hparam in bnn_param]

        return tf.add(self.bnn.forward(X, nn_param),
                      # TODO: tfp.bijector on [0,1] bnn_param['tausq']*
                      tf.reshape(self.gam.dense(Z, gam_param['W']), (X.shape[0], 1)))

    def _closure_log_prob(self, X, Z, y):
        def _log_prob(*tensorlist):
            # parameter parsing
            gam_param, bnn_param, sigma = self.parse_tensorlist(tensorlist)

            likelihood = self.likelihood_model(X, Z, bnn_param, gam_param)

            return tf.reduce_sum(likelihood.log_prob(y=y, sigma=sigma)) + \
                   self.bnn._layer_log_prob(bnn_param) + \
                   self.gam.rw.prior_log_prob(gam_param)

        return _log_prob

    def parse_tensorlist(self, tensorlist):
        """:returns tuple gam_param, bnn_param, sigma, with the first two being
        dicts of parameter: tensor (cleaned of sigma), sigma is a tensor"""
        gam_param, bnn_param, sigma = tensorlist[:len(self.gam.parameters) - 1], \
                                      tensorlist[len(self.gam.parameters) - 1:-1], \
                                      tensorlist[-1]

        bnn_param = self.bnn.parse_tensorlist(bnn_param)
        gam_param = {k: v for k, v in zip(self.gam.parameters, gam_param)}

        return gam_param, bnn_param, sigma

    def sample(self):
        gam_param = self.gam.rw.sample()
        bnn_param = self.bnn._layer_sample()
        return gam_param, bnn_param

    def flat_initialize(self, X_bnn, gam_ols=None):
        """
        :param ols: {'X':Z, 'y':y} to initialize gam with OLS estimate for W.
        :return: list of tensors
        """
        # delegate initialization
        gam_param = self.gam.rw.sample()
        if gam_ols is not None:
            gam_param['W'] = self.gam.OLS(**gam_ols)
            # gam_param['sigma']

        self.gam_init = gam_param

        # remove sigma from gam & flatten gam_param
        gam_param = list(v for k, v in gam_param.items() if k != 'sigma')

        # CAREFULL THIS ALSO APPENDS SIGMA of BNN's Normal likelihood!
        bnn_param = self.bnn.flat_initialize(X_bnn)
        return gam_param + bnn_param

    def plot2d_functions(self, gmrf2d, X_gam=None, y_gam=None, title='Effect', gam_funcs={}, bnn_funcs={}):
        """plotting gam (1d) & bnn model (2d)'s TRUE partial residuals plots.
        they are exacly known!"""

        # for irregular grid data make use of Delauny triangulation & trisurf:
        # https://fabrizioguerrieri.com/blog/2017/9/7/surface-graphs-with-irregular-dataset

        df = pd.DataFrame({'X': tf.reshape(X_gam, (X_gam.shape[0],)).numpy()})
        for name, var in gam_funcs.items():
            df[name] = tf.reshape(var, (var.shape[0],)).numpy()
        df = df.melt('X', value_name='y')
        df = df.rename(columns={'variable': 'functions'})

        (meshx, meshy), _ = gmrf2d.gmrf.grid
        # meshx, meshy = np.meshgrid(x, y, indexing='ij')
        gridxy = np.stack((meshx, meshy), axis=-1)

        fig = plt.figure()
        plt.title('{}'.format(title))

        # plot coefficents without TP-Splines
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_wireframe(meshx, meshy, gmrf2d.gmrf.z.reshape(meshx.shape), color='C1')
        ax1.set_title('Coefficents at grid position')

        # plot the bnn mean surface
        ax2 = fig.add_subplot((222), projection='3d')
        x1 = gmrf2d.X[:, 0]
        x2 = gmrf2d.X[:, 1]
        mu = gmrf2d.gmrf.surface(gmrf2d.X)
        y = gmrf2d.y
        triang = triangulate_remove_artifacts(x1, x2, 0.1, 9.9, 0.1, 9.9, plot=False)

        ax2.plot_trisurf(triang, mu, cmap='jet', alpha=0.4)  # groud truth function (approximation
        ax2.scatter(x1, x2, y, marker='.', s=10, c="black", alpha=0.5)
        ax2.view_init(elev=60, azim=-45)

        for name, prediction in bnn_funcs.items():
            ax2.plot_trisurf(triang, prediction, cmap='jet', alpha=0.4)

        # ax2.set_title('TE-Spline with plugged-in gmrf-coef.')

        # ax2.plot_surface(meshx, meshy, gmrf2d.gmrf.surface(gridxy), rstride=8, cstride=8, alpha=0.3)
        # ax2.scatter(gmrf2d.X[:, 0].numpy(), gmrf2d.X[:, 1].numpy(), gmrf2d.y)

        # plot the gam functions
        ax3 = fig.add_subplot(224)
        sns.scatterplot(
            x=tf.reshape(X_gam, (X_gam.shape[0],)).numpy(),
            y=y_gam.numpy() + surface.y - surface.mu, ax=ax3)  # FIXME!!!!
        sns.lineplot('X', y='y', hue='functions', alpha=0.7, data=df, ax=ax3)

        plt.show()

