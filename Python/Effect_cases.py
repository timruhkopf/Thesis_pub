"""
        # NEIGHBORHOOD STRUCTURE RELATED
        :param radius=0.25,

        # PRECISION MATRIX RELATED
        :param rho=0.2:
        :param tau=0.1:

        :return:
        z: grf sample vector.
        R: precisionmatrix, which induced z.
        B: used Decomposition of R to create z
        """
import numpy as np
from Python.Effect import Effects1D, Effects2D
from Python.bspline import diff_mat, penalize_nullspace
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


class GRF(Effects2D):
    # with eigendecomposition
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1, tau=1, tau1=1,
                 decomp=['eigenB', 'choleskyB'][0]):
        """
        FOLLOWING THE PAPER OF PICHOT (Algorithms for GRF Construction:
        EIGEN, CHOLESKY [& CIRCULANT EMBEDDING]
        :param xgrid:
        :param ygrid:
        :param corrfn:
        :param lam:
        :param phi:
        :param delta:
        :param tau:
        :param tau1: variance of the z ~ MVN(0, tau1*I), bevore B^T z is applied
        :param decomp:
        """
        self.decomp = decomp

        # generate grid
        super(GRF, self).__init__(xgrid, ygrid)

        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self._construct_precision_GRF(tau)

        self._sample_uncond_from_precisionB(self.Q, tau1, decomp)
        self._generate_surface()

        self.plot()

    def _construct_precision_GRF(self, tau):
        self.Q = self.kernel_distance
        np.fill_diagonal(self.Q, 1)  # FIXME: Fact checkk if not e.g. rowsum?

        self.Q = tau * self.Q

    def plot(self):
        self.plot_interaction(title='GRF with {}'.format(self.decomp))


class GMRF(Effects2D):
    # with eigendecomposition
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1, radius=4, tau=1, tau1=1,
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

        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self._construct_precision_GMRF(radius, tau)

        self._sample_uncond_from_precisionB(self.Q, tau1, decomp)
        self._generate_surface()

        self.plot()

    def _construct_precision_GMRF(self, radius, tau):
        self.Q = tau * self._keep_neighbour(self.kernel_distance, radius, fill_diagonal=True)

    def plot(self):
        self.plot_interaction(title='GMRF with {}'.format(self.decomp))


class GMRF_K(Effects2D):
    """Construct GMRF from K_order=D_order @ D_order with nullspace penalty on K.
    coef are drawn from the resulting MVN(0, penK)"""

    def __init__(self, xgrid, ygrid, order=2, tau=1, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
        """
        # FOLLOWING FAHRMEIR KNEIB LANG
        :param xgrid:
        :param ygrid:
        :param order:
        :param tau:
        :param sig_Q:
        :param sig_Q0:
        :param threshold:
        """
        # generate grid
        super(GMRF_K, self).__init__(xgrid, ygrid)

        self._construct_precision_GMRF_K(order, tau)
        self._sample_with_nullspace_pen(Q=self.Q, sig_Q=sig_Q, sig_Q0=sig_Q0, threshold=threshold)
        self._generate_surface()

        self.plot()

    def _construct_precision_GMRF_K(self, order, tau):
        (meshx, _), _ = self.grid
        no_coef = meshx.shape[0]

        d, K = diff_mat(dim=no_coef, order=order)  # FIXME: dimensions appropriate determine by grid
        self.Q = tau * np.kron(np.eye(K.shape[0]), K) + np.kron(K, np.eye(K.shape[0]))

        print('rank of Q: ', np.linalg.matrix_rank(self.Q))
        print('shape of Q: ', self.Q.shape)

    def plot(self):
        self.plot_interaction(title='GMRF_K with Nullspace penalty to MVN')


class GMRF_K2(GMRF_K):
    """Construct GMRF from K_order=D_order @ D_order with nullspace penalty on K.
    coef are drawn from the resulting MVN(0, penK)

    different approach: try nullspace penalty & Cholesky afterwards"""

    def __init__(self, xgrid, ygrid, order=2, tau=1, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
        """
        # FOLLOWING FAHRMEIR KNEIB LANG
        :param xgrid:
        :param ygrid:
        :param order:
        :param tau:
        :param sig_Q:
        :param sig_Q0:
        :param threshold:
        """
        # generate grid
        Effects2D.__init__(self, xgrid=xgrid, ygrid=ygrid)

        self._construct_precision_GMRF_K(order, tau)
        Sigma, penQ = penalize_nullspace(self.Q, sig_Q, sig_Q0, threshold)
        self.Sigma = Sigma
        self.penQ = penQ
        self._sample_uncond_from_precisionB(penQ, tau, decomp='choleskyB')
        self._generate_surface()

        self.plot()


class GMRF_VL(Effects2D):
    # NOT WORKING
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1,
                 radius=4, rho=0.7, tau=1, tau1=1, decomp=['eigenB', 'choleskyB'][0]):
        self.decomp = decomp
        # generate grid
        super(GMRF_VL, self).__init__(xgrid, ygrid)

        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self._construct_precision_GMRF_VL(radius, rho, tau)

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
        plt.imshow(self.SIGMA, cmap='hot', interpolation='nearest')

        # Q =(I_S - B) SIGMA⁻1
        self.Q = (np.eye(BS.shape[0]) - BS).dot(Sigma_inv)
        plt.imshow(self.Q, cmap='hot', interpolation='nearest')

        print('rank of Q: ', np.linalg.matrix_rank(self.Q))
        print('shape of Q: ', self.Q.shape)

    def plot(self):
        self.plot_interaction(title='GMRF_VL with {}'.format(self.decomp))


class Cond_GMRF(Effects2D):
    # original function needs refactoring, but works
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1,
                 radius=4, tau=1, no_neighb=4, decomp=['draw_normal', 'cholesky'][0], seed=1337):
        self.decomp = decomp
        super(Cond_GMRF, self).__init__(xgrid, ygrid)

        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self._sample_conditional_gmrf(corrfn, lam, no_neighb, decomp, radius, tau, seed)
        self._generate_surface()

        self.plot()

    def _sample_conditional_gmrf(self, corrfn='gaussian', lam=1, no_neighb=4, decomp=['draw_normal', 'cholesky'][0],
                                 radius=20, tau=0.1, seed=1337):
        """
        conditional sampling of grf (compare Rue / Held slides p.59, eq (10))
        :param corrfn:
        :param lam:
        :param no_neighb:
        :param decomp:
        :param radius:
        :param tau:
        :param seed:
        :return:
        """
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

        # index workaround for deselection
        mask = np.ones(len(gridvec), np.bool)
        mask[edges] = 0
        Q_AAdata = gridvec[mask]
        Q_ABdata = gridvec[~mask]

        # generate Qs and deselect neighbours for GMRF
        # order of function calls important, as grid_distance tampers with distance
        self._grid_distances('gaussian', lam=1, phi=0, delta=1, gridvec=Q_AAdata)
        Q_AA = self._keep_neighbour(self.kernel_distance, radius, fill_diagonal=True)

        self._grid_distances('gaussian', lam=1, phi=0, delta=1, gridvec=Q_ABdata)
        Q_BB = self._keep_neighbour(self.kernel_distance, radius, fill_diagonal=True)

        # euclidean kernel comparison: GRF Qs
        corr = {
            'gaussian': lambda h: np.exp(-(h / lam) ** 2)
            # consider different kernels: exponential
        }[corrfn]

        dist_AB = cdist(XA=Q_AAdata, XB=Q_ABdata, metric='euclidean')
        Q_AB = corr(dist_AB)

        # deselect neighbours (dist_AB not in self!)
        neighbor = dist_AB <= radius
        Q_AB = np.where(neighbor == False, 0, Q_AB)

        # for plotting purposes:
        Q1 = np.concatenate([Q_AA, Q_AB], axis=1)
        Q2 = np.concatenate([Q_AB.T, Q_BB], axis=1)
        self.Q = np.concatenate([Q1, Q2], axis=0)

        if seed is not None:
            np.random.seed(seed=seed)

        if decomp == 'draw_normal':  # flavours of drawing xa
            xb = np.random.multivariate_normal(mean=np.zeros(Q_BB.shape[0]), cov=tau * Q_BB)
            xa = np.random.multivariate_normal(mean=-tau * Q_AB.dot(xb - 0), cov=tau * Q_AA)

            # xa = np.zeros((Q_AA.shape[0],))  # Consider remove this control
        elif decomp == 'cholesky':
            xb = self._sample_backsolve(L=np.linalg.cholesky(tau * Q_BB),
                                        z=np.random.normal(loc=0, scale=1, size=Q_BB.shape[0]),
                                        mu=np.zeros(Q_BB.shape[0]))
            xa = self._sample_backsolve(L=np.linalg.cholesky(tau * Q_AA),
                                        z=np.random.normal(loc=0, scale=1, size=Q_AA.shape[0]),
                                        mu=-Q_AB.dot(xb - 0))

        # join coefficients of xa, xb in to sorted gridvector
        # (which in generate_surface will be destacked appropriately
        z = np.zeros(shape=gridvec.shape[0])
        z[mask] = xa
        z[~mask] = xb

        self.z = z

    def plot(self):
        self.plot_interaction(title='Cond_GMRF with {}'.format(self.decomp))


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
        self.Q = diff_mat(dim=no_coef, order=order)[1]
        Effects2D._sample_with_nullspace_pen(self, self.Q, sig_Q, sig_Q0, threshold)
        self._generate_bspline(degree)
        self.plot()

        # CONSIDER: draw the first value with a prior & make conditional GMRF 1d

    def plot(self):
        self.plot_bspline()


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)
    ygrid = (0, 10, 0.5)

    # (EFFECT SUBCLASSES: GMRF / BSPLINE) --------------------------------------
    # 2D Cases
    # FIXME: check that seeds actually make it reproducible
    grf = GRF(xgrid, ygrid, tau=1, decomp='eigenB')
    gmrf = GMRF(xgrid, ygrid, lam=1, phi=40, delta=10, radius=10, tau=1, tau1=20, decomp='eigenB')
    gmrf = GMRF(xgrid, ygrid, lam=1, phi=70, delta=1, radius=10, tau=1, tau1=20, decomp='choleskyB')
    gmrf = GMRF(xgrid, ygrid, radius=6, tau=1, tau1=1, decomp='choleskyB')
    gmrf_k = GMRF_K(xgrid, ygrid, order=2, tau=1, sig_Q=1, sig_Q0=0.01)
    gmrf_k2 = GMRF_K2(xgrid, ygrid, order=2, tau=1, sig_Q=1, sig_Q0=0.01)
    gmrf_vl = GMRF_VL(xgrid, ygrid)

    # FIXME: conditional effekt's edges are 'edgy'
    cond_gmrf = Cond_GMRF(xgrid, ygrid, radius=4, tau=1, no_neighb=4)

    # 1D Cases
    bspline_cum = Bspline_cum(xgrid, coef_scale=0.3)
    bspline_k = Bspline_K(xgrid)

    # (sparse X sampling) ------------------------------------------------------
    gmrf = GMRF(xgrid, ygrid, lam=1, phi=40, delta=10, radius=10, tau=1, tau1=20, decomp='eigenB')
    gmrf.sample_from_surface_density(n=10000, q=(0.05, 0.95), factor=2)
    gmrf.plot_rejected_contour()

    # (Draw y example) ---------------------------------------------------------
    # sample coordinates
    n = 1000
    x, y = np.random.uniform(low=xgrid[0], high=xgrid[1], size=n), \
           np.random.uniform(low=ygrid[0], high=ygrid[1], size=n)

    mu = bspline_k.spl(x) + bspline_cum.spl(y) + gmrf_k.surface(np.stack([x, y], axis=-1))
    # mu = gmrf_k.surface(np.stack([x, y], axis=-1))

    z = np.random.normal(loc=mu, scale=0.1, size=n)

    # (draw y with heteroscedasticity) -----------------------------------------
    # bspline_k1 = Bspline_K(xgrid, order=2, sig_Q=0.1, sig_Q0=0.1)
    # mu_sigma = bspline_k1.spl(x)  # FIXME: ensure positive values for variance
    # mu_sigma += 1
    # mu_sigma *= 0.2
    # mu = 0.2 * mu
    #
    # mu_sigma = 2 + x * 3
    # z = np.random.normal(loc=mu, scale=mu_sigma)

    # (plot y) -----------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    # ax.plot_wireframe(meshx, meshy, fxy.surface(gridxy))
    ax.scatter(xs=x, ys=y, zs=z, alpha=0.3)
    ax.set_title('N(f(x,y), ...) = z')

    ax1 = fig.add_subplot(222, projection='3d')
    ax1.scatter(xs=x, ys=y, zs=mu, alpha=0.3)
    ax1.set_title('mu')

    # ax2 = fig.add_subplot(223, projection='3d')
    # ax2.scatter(xs=x, ys=y, zs=mu_sigma, alpha=0.3)
    # ax2.set_title('mu_sigma')

    plt.show()

    print('')
