import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.tri as mtri  # for trisurface with irregular grid

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def plot_tfd_1D(dist, support=tf.range(0., 20., 0.05)):
    """plotting a 1D tfp.distribution on support (x)"""
    var = dist.prob(support).numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    sns.lineplot(support, var, ax=ax)
    plt.plot()


def plot1d_functions(X, y, confidence=None, **kwargs):
    """
    function to plot 1d Data and the function estimates of the data
    :param X: Tensor (n,1)
    :param y: Tensor (n,1)
    :param confidence: optional dict, containing x, y1, y2 keys to plot
    a shaded region between y1 and y2, representing some quantiles of the
    posterior predictive conditional on x
    # carefull x != X (x should be more fine grained!)
    :param kwargs: {'funcname': Tensor(n,1), ...}

    :example:
    # (1) plot functions
    n = 200
    X = tfd.Uniform(-10., 10.).sample(n)
    Z = tf.convert_to_tensor(
        get_design(X.numpy(), degree=2, no_basis=no_basis),
        tf.float32)
    f_true = gam_rw.dense(Z, true_param['W'])
    f_init = gam_rw.dense(Z, init_param['W'])
    f_ols = gam_rw.dense(Z, gam_rw.OLS(Z, y))

    plot1d_functions(X,y, **{'true':f_true, 'init':f_init, 'ols': f_ols})

    # (2) Confidence bands syntax:
     x = tf.linspace(-10., 10., 100)
    z = tf.convert_to_tensor(
        get_design(x.numpy(), degree=2, no_basis=no_basis),
        tf.float32)
    mi = gam_rw.dense(z, true_param['W'])
    ma = gam_rw.dense(z, init_param['W'])
    plot1d_functions(X, y, confidence={'x': x, 'y1': mi, 'y2': ma},
                     **{'ols': f_ols, 'true': f_true, 'init': f_init})

    """
    # TODO tensorshape inference ( product(*z.shape) ) für jede dieser variablen
    # convert to seaborn format
    df = pd.DataFrame({'X': tf.reshape(X, (X.shape[0],)).numpy()})
    for name, var in kwargs.items():
        df[name] = tf.reshape(var, (var.shape[0],)).numpy()
    df = df.melt('X', value_name='y')
    df = df.rename(columns={'variable': 'functions'})

    # plot the functions
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    sns.scatterplot(
        x=tf.reshape(X, (X.shape[0],)).numpy(),
        y=tf.reshape(y, (y.shape[0],)).numpy(), ax=ax)
    sns.lineplot('X', y='y', hue='functions', alpha=0.5, data=df, ax=ax)
    if confidence is not None:  # plot (optional) confidence bands
        ax.fill_between(**confidence, alpha=0.4, facecolor='lightblue')
    fig.suptitle('Functions of the data')
    plt.plot()


def triangulate_remove_artifacts(x, y, xl=0.1, xu=9.9, yl=0.1, yu=9.9, plot=True):
    # remove artifacts from triangulation method with some boundary
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


if __name__ == '__main__':
    # plot_tfd_1D(dist=tfd.InverseGamma(0.1, 0.1, name='tau'),
    #             support=tf.range(0., 20., 0.05))
    plot_tfd_1D(dist=tfd.Gamma(6, 20))
    plot_tfd_1D(dist=tfd.HalfCauchy(0., 1.))

    plot_tfd_1D(dist=tfd.InverseGamma(1., 1.))
    plot_tfd_1D(dist=tfd.Gamma((2 + 1) / 2., tfd.HalfCauchy(0, 1).sample() ** 2 / 2.))
