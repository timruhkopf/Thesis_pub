import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

from Python.SamplerPrecision import SamplerPrecision
from scipy.interpolate import BSpline  # FIXME: replace ndspline
from Python.bspline import diff_mat1D


class Effects1D(SamplerPrecision):
    def __init__(self, xgrid):
        self.xgrid = xgrid
        self.x = np.linspace(start=self.xgrid[0], stop=self.xgrid[1],
                             num=100, endpoint=True)

    def _generate_bspline(self, degree):
        """
        mean value of data distribution is the b-spline value f(x)
        f(x) = \gamma B(degree), with \gamma_j | gamma_(i<J) [ ~ N(0, coef_scale)
        i.e. gamma is a random walk
        y = N(f(x), data_scale)

        :param xgrid: tuple: (start, end) of linspace.
        :param n_basis: number of basis functions
        :param degree: basis degree
        :param coef_scale: variance for random walk on basis' coefficents
        :return:
        (1) vector of size n, that is normally distributed with mean f(x)
        (2) self.spl: spline function; return of BSpline. allows evaluating any x
        (3) self.z: \gamma vector, produced by random walk
        """

        # random b-spline function parametrization
        n_knots = degree + self.z.size + 1

        # function generator for one regressor
        self.spl = BSpline(t=np.linspace(start=self.xgrid[0], stop=self.xgrid[1],
                                         num=n_knots, endpoint=True),
                           c=self.z,
                           k=degree,
                           extrapolate=True)

    def log_prob(self):
        pass

    def plot_bspline(self):
        """plotting the bspline resulting from bspline_param"""
        import pylab  # FIXME remove this for plt.scatter!

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.x, self.spl(self.x))
        pylab.show()


class Bspline_cum(Effects1D):
    """Bspline generator based on cumulative sum of Normaldistrib"""

    def __init__(self, xgrid, seed=None, n_basis=7, coef_scale=2.,
                 degree=2, **kwargs):
        super(Bspline_cum, self).__init__(xgrid)
        self._random_coef(size=n_basis, loc=0, scale=coef_scale, seed=seed, **kwargs)
        self._generate_bspline(degree)
        self.plot()

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

    def plot(self):
        self.plot_bspline()


class Bspline_K(Effects1D):
    # with K matrix and proper prior ( nullspace penalty)
    def __init__(self, xgrid, no_coef=20, order=2, sig_Q=1, sig_Q0=1, threshold=10 ** -3, degree=2):
        """
        BSPLINE with K matrix & nullspace_penalty sampling
        :param xgrid:
        :param no_coef: number of Bspline coefficents/ no of bases
        :param order: order of difference K_order = D_order @ D_order
        :param tau:
        :param sig_Q: (sig_Q*Q + sig_Q0* S0)**-1 with S0 nullspace eigenvector matrix
        :param sig_Q0:
        :param threshold: determining numerical zero in eigenvalues
        :param degree: Bspline degree
        """
        super(Bspline_K, self).__init__(xgrid)
        self.Q = diff_mat1D(dim=no_coef, order=order)[1]
        SamplerPrecision._sample_with_nullspace_pen(self, self.Q, sig_Q, sig_Q0, threshold)
        self._generate_bspline(degree)
        self.plot()

        # CONSIDER: draw the first value with a prior & make conditional GMRF 1d

    def plot(self):
        self.plot_bspline()


class Bspline_K_cond(Effects1D):
    # with K matrix and proper prior ( nullspace penalty)
    def __init__(self, xgrid, no_coef=20, order=2, degree=2, tau=1):
        """
        BSPLINE with K matrix & conditionally sampling values
        :param xgrid:
        :param no_coef: number of Bspline coefficents/ no of bases
        :param order: order of difference K_order = D_order @ D_order
        :param tau:
        :param sig_Q: (sig_Q*Q + sig_Q0* S0)**-1 with S0 nullspace eigenvector matrix
        :param sig_Q0:
        :param threshold: determining numerical zero in eigenvalues
        :param degree: Bspline degree
        """
        Effects1D.__init__(self, xgrid)
        self.Q = diff_mat1D(dim=no_coef, order=order)[1]

        # workaround as it was originally intended for 2D grid
        self.grid = (None, None), np.arange(0, no_coef, 1)
        self._sample_conditional_precision(cond_points=[0, no_coef - 1], tau=tau)

        self._generate_bspline(degree)
        self.plot()

    def plot(self):
        self.plot_bspline()


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)

    # 1D Cases
    # bspline_cum = Bspline_cum(xgrid, coef_scale=0.3)
    # bspline_k = Bspline_K(xgrid)
    bspline_K_cond = Bspline_K_cond(xgrid, no_coef=10, order=2, tau=0.7)
    print('')
