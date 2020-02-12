from Python.Effect2D import Effects2D
import numpy as np


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
        self.decomp = decomp

        # generate grid
        super(GRF, self).__init__(xgrid, ygrid)

        self._construct_precision_GRF(corrfn, lam, phi, delta)

        self._sample_uncond_from_precisionB(self.Q, tau, decomp)
        self._generate_surface()

        self.plot()

    def _construct_precision_GRF(self, corrfn, lam, phi, delta):
        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self.Q = self.kernel_distance
        np.fill_diagonal(self.Q, 1)  # FIXME: Fact checkk if not e.g. rowsum?

    def plot(self):
        self.plot_interaction(title='GRF with {}'.format(self.decomp))


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
        self.decomp = decomp
        # generate grid
        super(GMRF, self).__init__(xgrid, ygrid)

        self._construct_precision_GMRF(radius, corrfn, lam, phi, delta)

        self._sample_uncond_from_precisionB(self.Q, tau, decomp)
        self._generate_surface()

        self.plot()

    def _construct_precision_GMRF(self, radius, corrfn, lam, phi,
                                  delta):  # FIXME Refactor such, that GMRF inherritance of GRF method becomes more apparent
        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self.Q = self._keep_neighbour(self.kernel_distance, radius, fill_diagonal=True)

    def plot(self):
        self.plot_interaction(title='GMRF with {}'.format(self.decomp))


class GMRF_cond(GMRF):
    # original function needs refactoring, but works
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1,
                 radius=4, tau=1, no_neighb=4, decomp=['draw_normal', 'cholesky'][0], seed=1337):
        self.decomp = decomp
        super(GMRF_cond, self).__init__(xgrid, ygrid)

        (meshx, meshy), gridvec = self.grid

        # Identify the index positions of points corresponding to
        # no_neighb square at each of the grid's edges
        row_length, col_length = meshx.shape
        edge_ref = [rowbase + col
                    for rowbase in np.linspace(0, (row_length) * (no_neighb - 1), no_neighb, dtype=np.int32)
                    for col in np.arange(0, no_neighb)]
        edge_pos = [0, row_length - no_neighb, (col_length - no_neighb) * row_length,
                    (col_length - no_neighb) * row_length + row_length - no_neighb]
        edges = [x + y for y in edge_pos for x in edge_ref]

        # self._sample_conditional_gmrf(corrfn, lam, no_neighb, decomp, radius, tau, seed)
        self._construct_precision_GMRF(radius, corrfn, lam, phi, delta)
        self._sample_conditional_precision(cond_points=edges, tau=tau)
        self._generate_surface()

        self.plot()

    def plot(self):
        self.plot_interaction(title='Cond_GMRF with {}'.format(self.decomp))


class GMRF_VL(Effects2D):
    # NOT WORKING
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1,
                 radius=4, rho=0.7, tau=1, tau1=1, decomp=['eigenB', 'choleskyB'][0]):
        self.decomp = decomp
        # generate grid
        super(GMRF_VL, self).__init__(xgrid, ygrid)

        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self._construct_precision_GMRF_VL(radius, rho, tau)  # FIXME: CAREFULL: HERE  TAU IS CORRECT DUE TO PRECISION

        self._sample_uncond_from_precisionB(self.Q, tau1, decomp)
        self._generate_surface()

        self.plot()

    def _construct_precision_GMRF_VL(self, radius, rho, tau):
        """
        FOLLOWING FAHRMEIR KNEIB LANG
        :param radius:
        :param rho:
        :param tau:
        :return:
        """
        # ATTEMPT ON GMRF formulation as in LECTURE SLIDES ON GMRF
        # radius : determining the discrete neighborhood structure
        # rho: correlation coefficient, determining how strong the neighbour affects this coef.
        # tau: global variance - how strong
        # mu = 0

        # w_sr \propto exp(-d(s,r)) where d euclidean dist
        # NOTE: d(s,r) is already in self.kernel_distance & all d(s,r)<= radius define s~r neighborhood!
        # 0.5 or 0.25 are radii that define the neighborhood structure on a grid
        # neighbor = squareform(self.dist) <= radius
        # w_sr = np.where(neighbor == False, 0, self.kernel_distance)
        #
        # note that self.Q is not yet finished!
        w_sr = self._keep_neighbour(self.kernel_distance, radius, fill_diagonal=False)

        # np.fill_diagonal(w_sr, 0)

        # w_s+ = sum_r w_sr for all s~r
        # NOTE: each rowsum of squareform(self.kernel_distance) , whose d(s,r) <= radius are w_s+,
        # excluding the point under evaluation
        w_s = w_sr.sum(axis=0)

        # B[s,r] = \beta_sr if s ~ r else 0
        # \beta_sr = \rho * w_sr/ w_s+
        # \rho element [0,1]
        # BS = rho * w_sr.dot(np.diag(w_s ** (-1)))

        BS = rho * np.diag(1 / w_s).dot(w_sr)

        # where SIGMA = diag(tau1, ..., tauS)
        # tau_s = \tau / w_s+
        Sigma_inv = np.diag(w_s / tau)

        self.SIGMA = np.diag(tau / w_s).dot(np.linalg.inv(np.eye(BS.shape[0]) - BS))
        # plt.imshow(self.SIGMA, cmap='hot', interpolation='nearest')

        # Q =(I_S - B) SIGMA⁻1
        self.Q = (np.eye(BS.shape[0]) - BS).dot(Sigma_inv)
        # plt.imshow(self.Q, cmap='hot', interpolation='nearest')

        print('rank of Q: ', np.linalg.matrix_rank(self.Q))
        print('shape of Q: ', self.Q.shape)

    def plot(self):
        self.plot_interaction(title='GMRF_VL with {}'.format(self.decomp))


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    grf = GRF(xgrid, ygrid, tau=1, decomp='eigenB')
    gmrf = GMRF(xgrid, ygrid, lam=1, phi=40, delta=10, radius=10, tau=1, decomp='eigenB')
    gmrf = GMRF(xgrid, ygrid, lam=1, phi=70, delta=1, radius=10, tau=1, decomp='choleskyB')
    gmrf = GMRF(xgrid, ygrid, radius=6, tau=1, decomp='choleskyB')
    gmrf_vl = GMRF_VL(xgrid, ygrid)

    # FIXME: conditional effect's edges are 'edgy'
    cond_gmrf = GMRF_cond(xgrid, ygrid, radius=4, tau=1, no_neighb=4)

    # (sparse X sampling) ------------------------------------------------------
    gmrf = GMRF(xgrid, ygrid, lam=1, radius=10, tau=1, decomp='eigenB')
    gmrf.sample_from_surface_density(n=1000, q=(0.5, 0.95), factor=4)
    gmrf.plot_rejected_contour(nbins=100)
    gmrf.plot_interaction(title='SparseX')
