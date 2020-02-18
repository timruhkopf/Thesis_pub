import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Data.Effect2D_K import *
from Python.MCMC.MCMC_Samplers import *



class BNN(AdaptiveHMC):
    # on how to write BNN models in Code (later example hast even TB)
    # https://github.com/tensorflow/probability/issues/292

    pass


if __name__ == '__main__':
    # (plot y) -----------------------------------------------------------------
    # class Exampledata(GMRF, AdaptiveHMC):
    #     def __init__(self, xgrid=(0, 10, 1), ygrid=(0, 10, 1), n=1000):
    #         self.X = np.stack([np.random.uniform(low=xgrid[0], high=xgrid[1]-xgrid[2], size=n), \
    #                            np.random.uniform(low=ygrid[0], high=ygrid[1]-ygrid[2], size=n)], axis=-1)
    #
    #         # mu = bspline_k.spl(x) + bspline_cum.spl(y) + gmrf_k.surface(np.stack([x, y], axis=-1))
    #         self.gmrf =GMRF(xgrid, ygrid, lam=1, phi=40, delta=10, radius=10, tau=1, decomp='eigenB')
    #         self.mu = self.gmrf.surface(self.X)  # np.stack([x, y]
    #
    #         self.y = np.random.normal(loc=self.mu, scale=0.1, size=n)
    #
    #         self.plot_y2D(xgrid=xgrid, ygrid=ygrid, effectsurface=self.gmrf)
    #
    # example = Exampledata()

    print('')
