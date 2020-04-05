import numpy as np
from Python.Effects.Effect1D import Effects1D
from Python.Effects.bspline import diff_mat1D


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


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)

    bspline_K_cond = Bspline_K_cond(xgrid, no_coef=10, order=2, tau=0.7)
