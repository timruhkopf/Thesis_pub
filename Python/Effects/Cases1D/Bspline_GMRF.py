import numpy as np
from Python.Effects.Effect1D import Effects1D
from Python.Effects.Effect2D import Effects2D


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

        gridvec = np.stack([gridvec, np.zeros((len(gridvec),))], axis=1)
        Effects2D._grid_distances(self, corrfn='gaussian', lam=1, phi=0, delta=1, gridvec=gridvec)

        # GRF STYLE
        # self.Q = self.kernel_distance

        # GMRF STYLE
        self.Q = Effects2D._keep_neighbour(self, self.kernel_distance, radius=radius, fill_diagonal=True)


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)

    bspline_dist = Bspline_GMRF(xgrid)  # not pos.-semi-definite!
    print('')
