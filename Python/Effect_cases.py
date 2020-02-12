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
from Python.bspline import diff_mat1D, diff_mat2D, penalize_nullspace
from scipy.spatial.distance import cdist


class GRF(Effects2D):
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1, tau=1, tau1=1,
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

        self._construct_precision_GMRF(radius, corrfn, lam, phi, delta, tau)

        self._sample_uncond_from_precisionB(self.Q, tau1, decomp)
        self._generate_surface()

        self.plot()

    def _construct_precision_GMRF(self, radius, corrfn, lam, phi, delta, tau):
        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self.Q = tau * self._keep_neighbour(self.kernel_distance, radius, fill_diagonal=True)

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

        #self._sample_conditional_gmrf(corrfn, lam, no_neighb, decomp, radius, tau, seed)
        self._construct_precision_GMRF(radius, corrfn, lam, phi, delta, tau)
        self._sample_conditional_precision(cond_points=edges, tau=tau)
        self._generate_surface()

        self.plot()

    def plot(self):
        self.plot_interaction(title='Cond_GMRF with {}'.format(self.decomp))


class GMRF_K(Effects2D):
    """Construct GMRF from K_order=D_order @ D_order with nullspace penalty on K.
    coef are drawn from the resulting MVN(0, penK)"""

    def __init__(self, xgrid, ygrid, order=1, tau=1, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
        """
        # FOLLOWING FAHRMEIR KNEIB LANG
        :param xgrid:
        :param ygrid:
        :param order: CURRENTLY HIGHER ORDERS OF K are not supported due to theoretical issues. implementation works fine
        :param tau:
        :param sig_Q:
        :param sig_Q0:
        :param threshold:
        """
        # generate grid
        super(GMRF_K, self).__init__(xgrid, ygrid)

        self._construct_precision_GMRF_K(tau)
        self._sample_with_nullspace_pen(Q=self.Q, sig_Q=sig_Q, sig_Q0=sig_Q0, threshold=threshold)
        self._generate_surface()

        self.plot()

    def _construct_precision_GMRF_K(self, tau):
        (meshx, _), _ = self.grid
        no_coef = meshx.shape[0]

        D1, D2, K = diff_mat2D(dim=no_coef)

        # \gamma^T K \gamma =   \gamma^T (I_(d2) kron K1 + K2 kron I_(d1) ) \gamma
        # with K1 = D1^T D1 and  K2 = D2^T D2 where D1 and D2 are 1st order difference matrices
        # in z1 & z2 direction respectively.

        self.Q = tau * K
        # self.Q = tau * (np.kron(np.eye(K.shape[0]), K) + np.kron(K, np.eye(K.shape[0])))

        print('rank of Q: ', np.linalg.matrix_rank(self.Q))
        print('shape of Q: ', self.Q.shape)

    def plot(self):
        self.plot_interaction(title='GMRF_K with Nullspace penalty to MVN')

class GMRF_K_cond(GMRF_K):
    """due to rank deficiency of K, sampling conditional on edges"""
    def __init__(self, xgrid, ygrid, order=1, tau=1,):
        """
        # FOLLOWING FAHRMEIR KNEIB LANG
        :param xgrid:
        :param ygrid:
        :param order: currently not supported
        :param tau:
        """
        # generate grid
        super(GMRF_K, self).__init__(xgrid, ygrid)
        self._construct_precision_GMRF_K(tau)

        # find edges
        (meshx, meshy), gridvec = self.grid
        row_length, col_length = meshx.shape
        no_neighb = 1
        edges = [0, row_length - no_neighb, (col_length - no_neighb) * row_length,
                 (col_length - no_neighb) * row_length + row_length - no_neighb]

        self._sample_conditional_precision(cond_points=edges, tau=tau)
        self._generate_surface()

        self.plot()

    def plot(self):
        self.plot_interaction(title='conditonal GMRF_K')




class GMRF_K_null_cholesky(GMRF_K):
    """Construct GMRF from K_order=D_order @ D_order with nullspace penalty on K.
    coef are drawn using Cholesky (on full rank K)

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

        self._construct_precision_GMRF_K(tau)
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
        #plt.imshow(self.SIGMA, cmap='hot', interpolation='nearest')

        # Q =(I_S - B) SIGMA⁻1
        self.Q = (np.eye(BS.shape[0]) - BS).dot(Sigma_inv)
        # plt.imshow(self.Q, cmap='hot', interpolation='nearest')

        print('rank of Q: ', np.linalg.matrix_rank(self.Q))
        print('shape of Q: ', self.Q.shape)

    def plot(self):
        self.plot_interaction(title='GMRF_VL with {}'.format(self.decomp))



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
        self.Q = diff_mat1D(dim=no_coef, order=order)[1]
        Effects2D._sample_with_nullspace_pen(self, self.Q, sig_Q, sig_Q0, threshold)
        self._generate_bspline(degree)
        self.plot()

        # CONSIDER: draw the first value with a prior & make conditional GMRF 1d

    def plot(self):
        self.plot_bspline()


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    # (EFFECT SUBCLASSES: GMRF / BSPLINE) --------------------------------------
    # 2D Cases
    # FIXME: check that seeds actually make it reproducible
    # grf = GRF(xgrid, ygrid, tau=1, decomp='eigenB')
    # gmrf = GMRF(xgrid, ygrid, lam=1, phi=40, delta=10, radius=10, tau=1, tau1=20, decomp='eigenB')
    # gmrf = GMRF(xgrid, ygrid, lam=1, phi=70, delta=1, radius=10, tau=1, tau1=20, decomp='choleskyB')
    # gmrf = GMRF(xgrid, ygrid, radius=6, tau=1, tau1=1, decomp='choleskyB')
    # gmrf_condK = GMRF_K_cond(xgrid, ygrid, order=1, tau=1)
    # gmrf_k = GMRF_K(xgrid, ygrid, order=1, tau=1, sig_Q=1, sig_Q0=0.8)
    # gmrf_k2 = GMRF_K_null_cholesky(xgrid, ygrid, order=1, tau=1, sig_Q=1, sig_Q0=0.1)
    # gmrf_vl = GMRF_VL(xgrid, ygrid)

    # FIXME: conditional effekt's edges are 'edgy'
    # cond_gmrf = GMRF_cond(xgrid, ygrid, radius=4, tau=1, no_neighb=4)

    # 1D Cases
    bspline_cum = Bspline_cum(xgrid, coef_scale=0.3)
    bspline_k = Bspline_K(xgrid)

    # (sparse X sampling) ------------------------------------------------------
    gmrf = GMRF(xgrid, ygrid, lam=1, radius=10, tau=1, tau1=20, decomp='eigenB')
    gmrf.sample_from_surface_density(n=1000, q=(0.5, 0.95), factor=4)
    gmrf.plot_rejected_contour(nbins=100)
    gmrf.plot_interaction(title='SparseX')

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
