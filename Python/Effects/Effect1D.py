import matplotlib.pyplot as plt
import numpy as np

from Python.Data.SamplerPrecision import SamplerPrecision
from scipy.interpolate import BSpline  # FIXME: replace ndspline

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
        plt.figure()
        plt.plot(self.x, self.spl(self.x), '-', label='line 1', linewidth=2)
        plt.title('Bspline')
        plt.show()
        # FIXME plot Q & eigen values of Q

    def plot_y1D(self):
        pass




