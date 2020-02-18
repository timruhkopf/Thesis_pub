import numpy as np
from Python.Data import *

class Bspline_cum(Effects1D):
    """Bspline generator based on cumulative sum of Normaldistrib"""

    def __init__(self, xgrid, seed=None, n_basis=7, coef_scale=2.,
                 degree=2, **kwargs):
        super(Bspline_cum, self).__init__(xgrid)
        self._random_coef(size=n_basis, loc=0, scale=coef_scale, seed=seed, **kwargs)
        self._generate_bspline(degree)
        self.plot_bspline()

    def _random_coef(self, size, seed=None, **kwargs):
        """
        Frist parameter is drawn with uniform(0,1). Afterwards Normal random
        walk on parameters; Bayesian: normal prior on consecutive
        parameter difference. Frequentist: splines with difference penalty

        :param seed:
        :param size: length of random walk vector
        :param kwargs: parameters of normal distribution
        :return: random walk vector
        """
        if seed is not None:
            np.random.seed(seed=seed)
        coef = np.concatenate([np.random.uniform(0, 1, size=1),
                               np.random.normal(size=size - 1, **kwargs)], axis=0)

        self.z = coef.cumsum()


class Bspline_K(Effects1D):

    def __init__(self, xgrid, no_coef=20, order=2, sig_Q=1, sig_Q0=1, threshold=10 ** -3, degree=2):
        """
        BSPLINE with K matrix & nullspace_penalty sampling
        :param xgrid:
        :param no_coef: number of Bspline coefficents/ no of bases
        :param order: order of difference K_order = D_order @ D_order
        :param sig_Q: (sig_Q*Q + sig_Q0* S0)**-1 with S0 nullspace eigenvector matrix
        :param sig_Q0:
        # with K matrix and proper prior (nullspace penalty) beware, sig_Q & sig_Q0
        # are factors before inverting Q - so smaller values mean larger variance!
        :param threshold: determining numerical zero in eigenvalues
        :param degree: Bspline degree
        """
        super(Bspline_K, self).__init__(xgrid)
        self.Q = diff_mat1D(dim=no_coef, order=order)[1]
        self._sample_with_nullspace_pen(self.Q, sig_Q, sig_Q0, threshold)
        self._generate_bspline(degree)
        self.plot_bspline()


class Bspline_K_cond(Effects1D):
    # with K matrix and proper prior, sampling first & last value beforehand
    def __init__(self, xgrid, no_coef=20, order=2, degree=2, tau=1):
        """
        BSPLINE with K matrix & conditionally sampling values
        :param xgrid:
        :param no_coef: number of Bspline coefficents/ no of bases
        :param order: order of difference K_order = D_order @ D_order
        :param tau:
        :param degree: Bspline degree
        """
        Effects1D.__init__(self, xgrid)
        self.Q = diff_mat1D(dim=no_coef, order=order)[1]

        # workaround as it was originally intended for 2D grid
        self.grid = (None, None), np.arange(0, no_coef, 1)
        # BE CAREFULL, order determines how many values must be sampled due to
        # increasing rank deficiency!
        self._sample_conditional_precision(cond_points=[0, no_coef - 1], tau=tau)

        self._generate_bspline(degree)
        self.plot_bspline()


class Bspline_GMRF(Effects1D):
    """A distance based approach to build GMRF, using nullspace penalty to sample
    from the precision"""
    def __init__(self, xgrid, radius=1, degree=2, sig_Q=1, sig_Q0=0.01, threshold=10 ** -3):
        Effects1D.__init__(self, xgrid)

        gridvec = np.arange(xgrid[0], xgrid[1], xgrid[2])
        self.construct_GMRF_precision(gridvec, radius)

        self._sample_with_nullspace_pen(self.Q, sig_Q, sig_Q0, threshold)
        self._generate_bspline(degree)
        self.plot_bspline()

    def construct_GMRF_precision(self, gridvec, radius):
        # workaround as grid_distances was inteded to be 2D
        from Python.Data.Effect2D import Effects2D
        gridvec = np.stack([gridvec, np.zeros((len(gridvec),))], axis=1)
        Effects2D._grid_distances(self, corrfn='gaussian', lam=1, phi=0, delta=1, gridvec=gridvec)

        # GRF STYLE
        # self.Q = self.kernel_distance

        # GMRF STYLE
        self.Q = Effects2D._keep_neighbour(self, self.kernel_distance, radius=radius, fill_diagonal=True)


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)

    # 1D Cases
    bspline_cum = Bspline_cum(xgrid, coef_scale=0.3)
    bspline_k = Bspline_K(xgrid, no_coef=10, order=1, sig_Q=0.1, sig_Q0=0.01, threshold=10 ** -3)
    bspline_K_cond = Bspline_K_cond(xgrid, no_coef=10, order=2, tau=0.7)

    bspline_dist = Bspline_GMRF(xgrid)  # not pos.-semi-definite!
    print('')
