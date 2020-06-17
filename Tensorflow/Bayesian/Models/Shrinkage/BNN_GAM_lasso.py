from Tensorflow.Bayesian.Models.Base.BNN import BNN
from Tensorflow.Bayesian.Models.Base.GAM_RW import GAM_RW
from Tensorflow.Bayesian.PriorModels.HiddenGroupLasso import HiddenGroupLasso
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


class BNN_GAM_lasso:
    @setkwargs
    def __init__(self, no_basis=20, hunits=[2, 10, 9, 8, 1], activation='relu'):  # TODO gam_RW parameters
        """
              model specification:
              :param hunits: list: number of hidden units per layer. first element is
              input shape. last is output_shape.
              :param activation: activation function for
              """

        self.gam = GAM_RW(no_basis)
        self.bnn = BNN(hunits, activation)
        self.bnn.layers[0] = HiddenGroupLasso(*hunits[:2], activation='identity')

        # prior
        self.parameters_list = self.bnn.parameters[:-1] + self.gam.parameters
        self.bijectors = self.bnn.bijectors[:-1] + self.gam.bijectors[:-1]  # removing the sigmas

        # Likelihood
        self.parameters_list.append('sigma')
        self.bijectors.append(tfb.Exp())

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
            y=y_gam.numpy() + surface.y - surface.mu, ax=ax3)
        sns.lineplot('X', y='y', hue='functions', alpha=0.7, data=df, ax=ax3)

        plt.show()


if __name__ == '__main__':
    # (USING INIT SURFACE AS TARGET) ------------------------------------------
    from Tensorflow.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    no_basis = 15
    n = 600
    # setting up the model
    model = BNN_GAM_lasso(no_basis, hunits=[2, 20, 10, 1], activation='sigmoid')

    # set up the bnn design matrix
    X_bnn = tfd.Uniform(0., 10.).sample((n, 2))
    Z = tf.convert_to_tensor(
        get_design(X_bnn[:, 1].numpy(), degree=2, no_basis=no_basis),
        tf.float32)

    # true_param_gam, _ = model.gam.sample_model(Z)
    # gam_mu = model.gam.dense(Z, true_param_gam['W'])

    gam_param, bnn_param = model.sample()

    likelihood = model.likelihood_model(X_bnn, Z, bnn_param, gam_param)
    like = likelihood.sample()
    y, sigma = like['y'], like['sigma']
    mu = model.forward(X_bnn, Z, bnn_param, gam_param)

    model.unnormalized_log_prob = model._closure_log_prob(X=X_bnn, Z=Z, y=y)

    # using the true function!
    init_tensorlist = [gam_param] + bnn_param
    init_tensorlist = [tensor for dict in init_tensorlist for tensor in dict.values()]
    init_tensorlist.append(sigma)

    model.unnormalized_log_prob(*init_tensorlist)

    # param at init
    # gam_param, bnn_param, _ = model.parse_tensorlist(init_tensorlist)

    # init = model.bnn.forward(X_bnn, bnn_param)
    mu = tf.reshape(mu, (n,))
    y = tf.reshape(y, (n,))

    fig = plt.figure()
    plt.axis('off')
    s = 'GroupLasso: lam:{:10.2f}, tausq:{:10.2f}, \nGam tausq:{:10.2f}'
    plt.title(s.format(bnn_param[0]['lam'], bnn_param[0]['tausq_group'], gam_param['tau']))

    x1 = X_bnn[:, 0]
    x2 = X_bnn[:, 1]  # gam / bnn axis
    triang = triangulate_remove_artifacts(x1, x2, 0.1, 9.9, 0.1, 9.9, plot=False)

    # plot the bnn& gam mean surface
    ax0 = fig.add_subplot((131), projection='3d')
    ax0.plot_trisurf(triang, mu, cmap='jet', alpha=0.7)
    ax0.scatter(x1, x2, y, marker='.', s=10, c="black")

    # plot gam
    ax1 = fig.add_subplot((132), projection='3d')
    ax1.plot_trisurf(triang, model.gam.dense(Z, gam_param['W']), cmap='jet', alpha=0.7)
    ax1.scatter(x1, x2, y, marker='.', s=10, c="black")

    # plot bnn only
    ax2 = fig.add_subplot((133), projection='3d')
    ax2.plot_trisurf(triang, tf.reshape(model.bnn.forward(X_bnn, bnn_param), (n,)), cmap='jet', alpha=0.7)
    ax2.scatter(x1, x2, y, marker='.', s=10, c="black")

    # rotate the axes
    for angle in range(0, 45):
        ax0.view_init(15, angle)
        ax1.view_init(15, angle)
        ax2.view_init(15, angle)

        plt.draw()
        plt.pause(.001)

    adHMC = AdaptiveHMC(
        initial=init_tensorlist,
        bijectors=model.bijectors,
        log_prob=model.unnormalized_log_prob)

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(10e2),
        num_results=int(10e3),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    parameters = [tensorlist for tensorlist in zip(*samples)]
    gam_param_samples, bnn_param_samples, sigma_samples = model.parse_tensorlist(parameters[0])

    # mean values
    tf.reduce_mean(samples, axis=0)

    # TODO adjust the below code to accept functions (e.g. posterior mode)
    # fig = plt.figure()
    # plt.axis('off')
    # s = 'GroupLasso: lam:{:10.2f}, tausq:{:10.2f}, \nGam tausq:{:10.2f}'
    # plt.title(s.format(bnn_param[0]['lam'], bnn_param[0]['tausq_group'], gam_param['tau']))
    #
    # x1 = X_bnn[:, 0]
    # x2 = X_bnn[:, 1]  # gam / bnn axis
    # triang = triangulate_remove_artifacts(x1, x2, 0.1, 9.9, 0.1, 9.9, plot=False)
    #
    # # plot the bnn& gam mean surface
    # ax0 = fig.add_subplot((131), projection='3d')
    # ax0.plot_trisurf(triang, mu, cmap='jet', alpha=0.7)
    # ax0.scatter(x1, x2, y, marker='.', s=10, c="black")
    #
    # # plot gam
    # ax1 = fig.add_subplot((132), projection='3d')
    # ax1.plot_trisurf(triang, model.gam.dense(Z, gam_param['W']), cmap='jet', alpha=0.7)
    # ax1.scatter(x1, x2, y, marker='.', s=10, c="black")
    #
    # # plot bnn only
    # ax2 = fig.add_subplot((133), projection='3d')
    # ax2.plot_trisurf(triang, tf.reshape(model.bnn.forward(X_bnn, bnn_param), (n,)), cmap='jet', alpha=0.7)
    # ax2.scatter(x1, x2, y, marker='.', s=10, c="black")
    #
    # # rotate the axes
    # for angle in range(0, 45):
    #     ax0.view_init(15, angle)
    #     ax1.view_init(15, angle)
    #     ax2.view_init(15, angle)
    #
    #     plt.draw()
    #     plt.pause(.001)



    # (USING TE SURFACE AS TARGET) --------------------------------------------
    from Tensorflow.Data.K_Surface import K_surface
    from Tensorflow.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    no_basis = 20
    n = 200
    # setting up the model
    model = BNN_GAM_lasso(no_basis, hunits=[2, 20, 10, 1], activation='tanh')

    # set up gam design matrix
    X_gam = tfd.Uniform(-10., 10.).sample(n)
    Z = tf.convert_to_tensor(
        get_design(X_gam.numpy(), degree=2, no_basis=no_basis),
        tf.float32)

    true_param_gam, _ = model.gam.sample_model(Z)
    gam_mu = model.gam.dense(Z, true_param_gam['W'])

    # set up the bnn design matrix
    surface = K_surface(n)
    X_bnn = surface.X
    sigma = surface.sigma
    mu = surface.mu

    y = tf.reshape(surface.y + gam_mu, (X_bnn.shape[0], 1))
    model.unnormalized_log_prob = model._closure_log_prob(X=surface.X, Z=Z, y=y)

    init_tensorlist = model.flat_initialize(X_bnn, gam_ols={'X': Z, 'y': tf.reshape(y, (y.shape[0],))})
    init_tensorlist[-1] = model.bnn.init_likelihood_sigma(
        tf.concat([X_bnn, tf.reshape(X_gam, (X_gam.shape[0], 1))], axis=1),
        y, cube=3.)

    # Deprec
    # this most likely will be none
    # model.gam.init_likelihood_sigma(
    # X=tf.concat([X_bnn, tf.reshape(X_gam, (X_gam.shape[0], 1))], axis=1), y=y, cube=0.5)
    model.unnormalized_log_prob(*init_tensorlist)

    # param at init
    gam_param, bnn_param, _ = model.parse_tensorlist(init_tensorlist)

    init = model.bnn.forward(X_bnn, bnn_param)
    init = tf.reshape(init, (n,))

    model.plot2d_functions(
        gmrf2d=surface, X_gam=X_gam, y_gam=gam_mu,
        gam_funcs={'true': model.gam.dense(Z, true_param_gam['W']),  # to get the residual
                   'init': model.gam.dense(Z, model.gam_init['W'])},
        bnn_funcs={'init': init})

    adHMC = AdaptiveHMC(
        initial=init_tensorlist,
        bijectors=model.bijectors,
        log_prob=model.unnormalized_log_prob)

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(1000),
        num_results=int(1000),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    gam_param_samples, bnn_param_samples, _ = model.parse_tensorlist(tf.reduce_mean(samples, axis=0))

    model.plot2d_functions(
        gmrf2d=surface, X_gam=X_gam, y_gam=gam_mu,
        gam_funcs={'true': model.gam.dense(Z, true_param_gam['W']),  # to get the residual
                   'init': model.gam.dense(Z, model.gam_init['W'])},
        bnn_funcs={'init': model.bnn.forward(X_bnn, bnn_param), 'post_mean': model.bnn.forward(X_bnn, bnn_param)})

    print()
