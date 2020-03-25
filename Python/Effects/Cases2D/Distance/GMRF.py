from Python.Effects.Effect2D import Effects2D


class GMRF(Effects2D):
    # with eigendecomposition
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1, radius=4, tau=1,
                 decomp=['eigenB', 'choleskyB'][0]):
        """
        FOLLOWING PAPER OF PICHOT, BUT INTRODUCING NEIGHBOURHOOD
        :param xgrid:
        :param ygrid:
        :param corrfn:
        :param lam:
        :param phi: angle of rotation matrix, influencing the distance matrix
        :param delta:
        :param radius:
        :param tau:
        :param tau1: variance of the z ~ MVN(0, tau1*I), bevore B^T z is applied
        :param decomp:
        """

        # generate grid
        Effects2D.__init__(self, xgrid, ygrid)

        self._construct_precision_GMRF(radius, corrfn, lam, phi, delta)

        self._sample_uncond_from_precisionB(self.Q, tau, decomp)
        self._generate_surface()

        self.plot_interaction(title='{} with {}'.format(self.__class__.__name__, decomp))

    def _construct_precision_GMRF(self, radius, corrfn, lam, phi,
                                  delta):  # FIXME Refactor such, that GMRF inherritance of GRF method becomes more apparent
        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self.Q = self._keep_neighbour(self.kernel_distance, radius, fill_diagonal=True)


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    gmrf = GMRF(xgrid, ygrid, lam=1, phi=40, delta=10, radius=10, tau=1, decomp='eigenB')
    gmrf = GMRF(xgrid, ygrid, lam=1, phi=70, delta=1, radius=10, tau=1, decomp='choleskyB')
    gmrf = GMRF(xgrid, ygrid, radius=6, tau=1, decomp='choleskyB')

    # (sparse X sampling) ------------------------------------------------------
    gmrf = GMRF(xgrid, ygrid, lam=1, radius=10, tau=1, decomp='eigenB')
    gmrf.sample_from_surface_density(n=1000, q=(0.5, 0.95), factor=4)
    gmrf.plot_rejected_contour(nbins=100)
    gmrf.plot_interaction(title='SparseX')
