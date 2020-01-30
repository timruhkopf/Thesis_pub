"""
        # FIXME doc
        Sampling a 2d Gaussian Random Field:
        https://hal.inria.fr/hal-01414707v2/document
        Note on correlation length \lambda:
        "Roughly speaking, the correlation length should be a measure of the
        constraint between height displacements of neighboring points of the
        surface: this constraint is expected to be significant if the points
        are well inside the correlation length and negligible outside it."
        - Giorgio Franceschetti, Daniele Riccio,
            in Scattering, Natural Surfaces, and Fractals, 2007
        https://www.sciencedirect.com/topics/mathematics/correlation-length

        meaning, it is a parameter influencing the "range" of the correlation

        1. generating R_NxN, the covariance structure, solely dependent on the distance
        between the points and are isotropic (Default: phi=0, delta=1):
        c(h) = E(Z_i Z_j) = E(Z_0, Z_k) = c(|x_i - x_j|), with x vectorvalued

        Thereby various distance functions are available

        1.2 anisotropic distance
        anisotropy in grf: simply exchange Euclidean dist
        with a rotation & prolongation matrix (see Fahrmeir Kneib & lang p.516):
        sqrt((u-v)' R(phi)' D(\delta) R(phi) (u-v))

        Note, that anisotropy is basically a directed smoothness constrained on
        the interaction surface

        2. decomposing R = BB' with either
            a) eigendecomposition (U \sqrt(Lambda)) (U \sqrt(Lambda))'
            b) choletzky LL'
            c) circulant embedding, the most performant version,
                R implementation: https://www.jstatsoft.org/article/view/v055i09/v55i09.pdf

        3. sampling a multivariate normal vector \theta of size N,
        with mu=0, cov=I_N (indep)

        4. realization of grf: Z = B\theta

        :param lam: lambda parameter: is covariance length (autocovariance influence)
        :param decomp: choice of decomposition: available are 'eigen' & 'cholesky'
        :param seed:

        # ANISOTROPY RELATED
        :param phi: angle of rotation matrix, influencing the distance matrix
        :param delta: anisotropy ratio influencing the distance matrix via a
                    prolongation matrix.

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
from Python.bspline_data import Effects1D, Effects2D
from Python.bspline import diff_mat


class GRF(Effects2D):
    # with eigendecomposition
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1, tau=1,
                 decomp=['eigenB', 'choleskyB'][0]):
        self.decomp = decomp

        # generate grid
        super(GRF, self).__init__(xgrid, ygrid)

        self.grid_distances(corrfn, lam, phi, delta)
        self.construct_precision_GRF(tau)

        self._sample_uncond_from_precisionB(self.Q, decomp)
        self.generate_surface()

        self.plot()

    def plot(self):
        self.plot_interaction(title='GMRF_VL with {}'.format(self.decomp))


class GMRF(Effects2D):
    # with eigendecomposition
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1, radius=4, tau=1,
                 decomp=['eigenB', 'choleskyB'][0]):
        self.decomp = decomp
        # generate grid
        super(GMRF, self).__init__(xgrid, ygrid)

        self.grid_distances(corrfn, lam, phi, delta)
        self.construct_precision_GMRF(radius, tau)

        self._sample_uncond_from_precisionB(self.Q, decomp)
        self.generate_surface()

        self.plot()

    def plot(self):
        self.plot_interaction(title='GMRF_VL with {}'.format(self.decomp))


class GMRF_K(Effects2D):
    """Construct GMRF from K_order=D_order @ D_order with nullspace penalty on K.
    coef are drawn from the resulting MVN(0, penK)"""

    def __init__(self, xgrid, ygrid, order=2, tau=1, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
        # generate grid
        super(GMRF_K, self).__init__(xgrid, ygrid)

        self.construct_precision_GMRF_K(order, tau)
        self._sample_with_nullspace_pen(Q=self.Q, sig_Q=sig_Q, sig_Q0=sig_Q0, threshold=threshold)
        self.generate_surface()

        self.plot()

    def plot(self):
        self.plot_interaction(title='GMRF_K with Nullspace penalty to MVN')


class GMRF_VL(Effects2D):
    # NOT WORKING
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1,
                 radius=4, rho=0.7, tau=1, decomp=['eigenB', 'choleskyB'][0]):
        self.decomp = decomp
        # generate grid
        super(GMRF_VL, self).__init__(xgrid, ygrid)

        self.grid_distances(corrfn, lam, phi, delta)
        self.construct_precision_GMRF_VL(radius, rho, tau)

        self._sample_uncond_from_precisionB(self.Q, decomp)
        self.generate_surface()

        self.plot()

    def plot(self):
        self.plot_interaction(title='GMRF_VL with {}'.format(self.decomp))


class Cond_GMRF(Effects2D):
    # original function needs refactoring, but works
    def __init__(self, xgrid, ygrid, decomp):
        self.decomp = decomp
        super(Cond_GMRF, self).__init__(xgrid, ygrid)
        pass

    def plot(self):
        self.plot_interaction(title='Cond_GMRF with {}'.format(self.decomp))


class Bspline_cum(Effects1D):
    """Bspline generator based on cumulative sum of Normaldistrib"""

    def __init__(self, xgrid, seed=None, n_basis=7, coef_scale=2,
                 degree=2, **kwargs):
        super(Bspline_cum, self).__init__(xgrid)
        self._random_coef(size=n_basis, loc=0, scale=coef_scale, seed=seed, **kwargs)
        self.generate_bspline(degree)
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
    def __init__(self, xgrid, no_coef=20, order=2, tau=1, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3, degree=2):
        super(Bspline_K, self).__init__(xgrid)
        self.Q = tau * diff_mat(dim=no_coef, order=order)[1]
        Effects2D._sample_with_nullspace_pen(self, self.Q, sig_Q, sig_Q0, threshold)
        self.generate_bspline(degree)
        self.plot()

    def plot(self):
        self.plot_bspline()



if __name__ == '__main__':
    xgrid = (0, 10, 0.5)
    ygrid = (0, 10, 0.5)

    # FIXME: check that seeds actually make it reproducible
    grf = GRF(xgrid, ygrid, tau=1, decomp='eigenB')
    gmrf = GMRF(xgrid, ygrid, radius=4, tau=1, decomp='eigenB')
    # gmrf = GMRF(xgrid, ygrid, radius=4, tau=1, decomp='choleskyB')
    gmrf_k = GMRF_K(xgrid, ygrid, order=2, tau=1, sig_Q=0.01, sig_Q0=0.01)
    gmrf_vl = GMRF_VL(xgrid, ygrid)
    cond_gmrf = Cond_GMRF(xgrid, ygrid)

    bspline = Bspline_cum(xgrid)
    bspline_k = Bspline_K(xgrid)

    # sample coordinates
    n = 1000
    x, y = np.random.uniform(low=xgrid[0], high=xgrid[1], size=n), \
           np.random.uniform(low=ygrid[0], high=ygrid[1], size=n)

    mu = bspline_k.spl(x) + bspline.spl(y) + gmrf_k.surface(np.stack([x, y], axis=-1))
    # mu = gmrf_k.surface(np.stack([x, y], axis=-1))

    z = np.random.normal(loc=mu, scale=0.1, size=n)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(122, projection='3d')
    # ax.plot_wireframe(meshx, meshy, fxy.surface(gridxy))
    ax.scatter(xs=x, ys=y, zs=z, alpha=0.3)
    ax.set_title('N(f(x,y), ...) = z')

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(xs=x, ys=y, zs=mu.real, alpha=0.3)
    ax1.set_title('mu')

    plt.show()

    print('')
