import numpy as np
from Tensorflow.Effects.Effect2D import Effects2D


class GRF(Effects2D):
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1, tau=1,
                 decomp=['eigenB', 'choleskyB'][0]):
        """
        FOLLOWING THE PAPER OF PICHOT (Algorithms for GRF Construction:
        EIGEN, CHOLESKY [& CIRCULANT EMBEDDING], eigendecomp default
        :param xgrid:
        :param ygrid:
        :param corrfn:
        :param lam:
        :param phi:
        :param delta:
        :param tau: variance of the z ~ MVN(0, tau1*I), bevore B^T z is applied
        :param decomp:
        """

        # generate grid
        Effects2D.__init__(self, xgrid, ygrid)

        self._construct_precision_GRF(corrfn, lam, phi, delta)

        self._sample_uncond_from_precisionB(self.Q, tau, decomp)
        self._generate_surface()

        self.plot_interaction(title='{} with {}'.format(self.__class__.__name__, decomp))

    def _construct_precision_GRF(self, corrfn, lam, phi, delta):
        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self.Q = self.kernel_distance
        np.fill_diagonal(self.Q, 1)  # FIXME: Fact checkk if not e.g. rowsum?


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    grf = GRF(xgrid, ygrid, tau=1, decomp='eigenB')
