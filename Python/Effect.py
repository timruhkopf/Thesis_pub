"""
Created on Mon Nov 6 2019
@author: T.Ruhkopf
@email:  tim.ruhkopf@outlook.de
"""

from itertools import product as prd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Surface generation
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import eigh
import ndsplines
from scipy.interpolate import BSpline  # FIXME: replace ndspline
from Python.bspline import penalize_nullspace
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# rejection sampling
from scipy.stats import kde


class Effects1D:
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


class Effects2D:
    """
    Possible extensions
    (1) consider random data distribution & building a grf upon it.
    eventhough it is possible to calculate the tensorproduct spline,
    thin plate regression splines are better suited to handle sparsely sampled
    regions via linear extrapolation

    # Consider further note: the grids are only relevant at effect construction,
    # but not #  for data sampling; as consequence, non overlapping grids for
    # the effect with smaller support assume an effectsize #  of zero for
    # points out of their bounds.
    """

    def __init__(self, xgrid, ygrid):
        self.xgrid = xgrid
        self.ygrid = ygrid

        x, y = np.arange(xgrid[0], xgrid[1], xgrid[2]), \
               np.arange(ygrid[0], ygrid[1], ygrid[2])
        self.grid = self._generate_grid(x, y)

    # (helper functions) -------------------------------------------------------
    def _generate_grid(self, x, y):
        xmesh, ymesh = np.meshgrid(x, y)
        # xflat, yflat = xmesh.flatten(), ymesh.flatten()
        # gridvec = np.stack((xflat, yflat), axis=1)

        a = list(prd(x, y))
        gridvec = np.array(a)

        return (xmesh, ymesh), gridvec

    def _grid_distances(self, corrfn, lam, phi, delta, gridvec):
        """
        Sampling a 2d Gaussian Random Field:
        https://hal.inria.fr/hal-01414707v2/document

        1. generating R_NxN, the covariance structure, solely dependent on the distance
        between the points and are isotropic (Default: phi=0, delta=1):
        c(h) = E(Z_i Z_j) = E(Z_0, Z_k) = c(|x_i - x_j|), with x vectorvalued

        Thereby various distance functions are available

        1.2 (optionally) anisotropic distance
        anisotropy in grf: simply exchange Euclidean dist
        with a rotation & prolongation matrix (see Fahrmeir Kneib & lang p.516):
        sqrt((u-v)' R(phi)' D(\delta) R(phi) (u-v))

        Note, that anisotropy is basically a directed smoothness constrained on
        the interaction surface

        :param corrfn:
        :param lam: correlation length \lambda:
        "Roughly speaking, the correlation length should be a measure of the
        constraint between height displacements of neighboring points of the
        surface: this constraint is expected to be significant if the points
        are well inside the correlation length and negligible outside it."
        - Giorgio Franceschetti, Daniele Riccio,
            in Scattering, Natural Surfaces, and Fractals, 2007
        https://www.sciencedirect.com/topics/mathematics/correlation-length
        meaning, it is a parameter influencing the "range" of the correlation
        :param phi: angle of rotation matrix, influencing the distance matrix
        :param delta: anisotropy ratio influencing the distance matrix via a
                    prolongation matrix.
        :param gridvec: array of Vectors, storing all points of the grid (nxdim)
        :return:
        """
        if phi != 0 or delta != 1:
            # anisotropy  # FIXME make this available for conditional
            def rotation(phi):
                return np.array([[np.cos(phi), np.sin(phi)],
                                 [-np.sin(phi), np.cos(phi)]])

            def prolongation(delta):
                return np.diag([delta ** -1, 1])

            r = rotation(phi)
            d = prolongation(delta)
            anisotropy = r.T.dot(d).dot(r)

            self.dist = pdist(X=gridvec, metric=lambda u, v: np.sqrt(((u - v).T.dot(anisotropy).dot((u - v)))))

        else:
            # isotropy
            self.dist = pdist(X=gridvec, metric='euclidean')

        corr = {
            'gaussian': lambda h: np.exp(-(h / lam) ** 2)
            # consider different kernels: exponential
        }[corrfn]
        self.kernel_distance = squareform(corr(self.dist))

    def _keep_neighbour(self, Q, radius, fill_diagonal=True):
        """
        keep neighbours radius based and optionally fill Q's diagonal:
        to introduce Markov property (of local dependence only)

        :param Q: Precision Matrix
        :param radius: defining which values of the precision to keep;
        keep if d <= radius
        :param fill_diagonal:
        :return: Precision Matrix Q
        """
        neighbor = squareform(self.dist) <= radius
        Q = np.where(neighbor == False, 0, Q)
        if fill_diagonal:
            np.fill_diagonal(Q, 1)

        return Q

    # (sampling GMRF) ----------------------------------------------------------
    # ANY such method must produce self.Q & self.z
    def _sample_with_nullspace_pen(self, Q, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
        # TENSORFLOW VERSION
        # trial on null space penalty
        Sigma, penQ = penalize_nullspace(Q, sig_Q, sig_Q0, threshold)
        self.Sigma = Sigma
        self.penQ = penQ
        rv_z = tfd.MultivariateNormalFullCovariance(
            covariance_matrix=self.Sigma,
            loc=0.)
        self.z = rv_z.sample().numpy()

    def _sample_uncond_from_precisionB(self, Q, tau, decomp=['eigenB', 'choleskyB'][0]):
        """
        1. decomposing R = BB' with either
            a) eigendecomposition (U \sqrt(Lambda)) (U \sqrt(Lambda))'
            b) choletzky LL'
            c) circulant embedding, the most performant version,
                R implementation: https://www.jstatsoft.org/article/view/v055i09/v55i09.pdf

        2. sampling a multivariate normal vector \theta of size N,
        with mu=0, cov=tau*I_N (indep)

        3. realization of g(m)rf: Z = B\theta

        :param Q: Precision
        :param tau: variance of MVN vector
        :param decomp: a) or b)
        :stores:
        self.B, the decomposition of Q.
        self.z, the g(m)rf Vector.
        """
        # independent Normal variabels, upon which correlation will be imposed
        theta = np.random.multivariate_normal(
            mean=np.zeros(Q.shape[0]),
            cov=tau * np.eye(Q.shape[0]))

        if decomp == 'eigenB':
            # DEPREC imaginary part!! - FLOATING POINT PRECISION SEEMS TO BE THE ISSUE IN ITERATIVE ALGORITHMS
            # compare: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
            # eigval, eigvec = np.linalg.eig(self.Q)
            # self.B = eigvec.dot(np.diag(np.sqrt(eigval)))

            eigval, eigvec = eigh(Q)
            plt.scatter(np.arange(eigval.size), eigval)

            self.B = eigvec.dot(np.diag(np.sqrt(eigval)))
            self.z = self.B.dot(theta)

        elif decomp == 'choleskyB':
            # RUE 2005 / 99: for decomp look at sample_GMRF
            self.B = np.linalg.cholesky(Q).T
            self.z = self._sample_backsolve(self.B, theta)

        else:
            raise ValueError('decomp is not propperly specified')

    def _sample_backsolve(self, L, z, mu=0):
        """
        # AUXILIARY METHOD
        Following RUE HELD 2005 slide 52 ff. explicitly slide 56
        Solve eq. system L @ x = z for x.
        if z ~ MVN(0,I) and Q = LL^T from cholesky, this allows to generate
        x ~ MVN(0, Q^(-1)

        :param L: upper triangular matrix
        :param z: vector
        :param mu: additional mean vector
        :return x: vector
        """
        if L.shape[1] != z.size:
            raise ValueError('improper dimensions')

        x = np.zeros(L.shape[0])
        x[-1] = z[-1] / L[-1, -1]

        for i in reversed(range(0, L.shape[0] - 1)):
            x[i] = (z[i] - L[i].dot(x)) / L[i, i]

        return x + mu


    def _sample_conditional_precision(self, cond_points, tau):
        _, gridvec = self.grid

        mask = np.ones(len(gridvec), np.bool)

        mask[cond_points] = 0

        selector = np.zeros((len(cond_points), gridvec.shape[0]), dtype=bool)
        selector[np.arange(len(cond_points)), [cond_points]] = True
        Q_BB = selector.dot(self.Q).dot(selector.T)

        intermediate = self.Q[mask, :]
        Q_AA = intermediate[:, mask]
        Q_AB = intermediate[:, ~mask]

        xb = np.random.multivariate_normal(mean=np.zeros(len(cond_points)), cov=0.001 * Q_BB)  # fixme tau * Q_BB
        xa = np.random.multivariate_normal(mean=-tau * Q_AB.dot(xb - 0), cov=tau * np.linalg.inv(Q_AA))

        z = np.zeros(shape=gridvec.shape[0])
        z[mask] = xa
        z[~mask] = xb
        self.z = z

        Q1 = np.concatenate([Q_AA, Q_AB], axis=1)
        Q2 = np.concatenate([Q_AB.T, Q_BB], axis=1)
        self.Q = np.concatenate([Q1, Q2], axis=0)

        plt.imshow(Q_BB, cmap='hot', interpolation='nearest')


    # (class methods) ----------------------------------------------------------
    def log_prob(self):
        pass

    # (Rejection Sampling gmrf density) ----------------------------------------
    def sample_from_surface_density(self, n, q=(0.05, 0.95), factor=2):
        """
        With rejection sampling, the Effect2D Surface can be interpreted as an
        unnormalized density function, of which we can draw samples.
        This functionality allows to draw sparse data (X) regions.

        THEORY: (McKay Rejection Sampling chp.29. p 364)
        P*(x) is some GMRF childclass from Effects2d

        cQ*(x) > P*(x) for all x,

        draw x from proposal actual Q(x), (note that c is a constant factor ensuring to surpass
        reject x if u > P*(x) with u ~ U[0, cQ*(x)]. otherwise accept it and append it to
        {x^(r)}

        note that if i choose cQ*, such that cQ*(x) > P*(x) does not strictly hold for all x,
        this means, that there are some regions, in which the data will always be accepted!
        this is the case, if a peak is cutoff, as no u can be drawn to surpass P*(x).

        On the otherhand, regions with negative values will always be rejected, punching holes
        in the domain

        ENSURE, that the resulting sample size is n!
        :param n: number of observations to be generated
        :param q: tuple of length 2, empirical quantiles on the depth & hight of the surface (evaluated at gridxy).
        Proposals in regions, whose surface value is below the lower quantile value will always be rejected.
        Proposals in regions, whose surface value is above the upper quantile value will always be accepted.
        :param factor: int: greater than zero, influences the speed of sampling.
        :return: A design matrix of 2D points, drawn from the unnormalized (surface) density
        """
        (meshx, meshy), _ = self.grid
        gridxy = np.stack((meshx, meshy), axis=-1)
        P = self.surface(gridxy)

        # correct for the values to be positive mostly
        quant = np.quantile(P, q=q)

        # taking Q to be a uniform, scaling the the proposal Q
        zero, cQ = quant[0], quant[1]

        m = factor * n
        X = np.zeros((n, 2))
        remainder = n
        while remainder > 0:
            # draw uniform proposals from Q(x) = Uniform and eval unnormalized density
            X_full = np.random.uniform(0, 10, size=2 * m).reshape((m, 2))
            P = self.surface(X_full)

            u = np.random.uniform(zero, cQ, size=m)
            keep = u < P
            no_proposals = sum(keep)

            if no_proposals <= remainder:
                X[n - remainder: n - remainder + no_proposals, :] = X_full[keep, :]
            else:
                X[n - remainder:, :] = X_full[keep, :][:remainder, :]
                break

            remainder = remainder - no_proposals
            m = factor * remainder

        self.X = X

    def plot_rejected_contour(self, nbins = 300):
        """Method to plot the result from sample_from_surface_density.
        Plotting Gaussian Kernel Density Estimate"""
        x, y = self.X[:, 0], self.X[:, 1]

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # # Make the plot
        # plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
        # plt.show()
        #
        # # Change color palette
        # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
        # plt.show()

        plt.contourf(xi, yi, zi.reshape(xi.shape), 20, cmap='RdGy')
        plt.colorbar()
        plt.scatter(x, y, alpha=0.1)
        plt.show()


    def plot_interaction(self, title):  # , pred_error=None):
        """
        # consider plotting both 3d graphics (grf & Tp-BSpline):
            # https://matplotlib.org/mpl_examples/mplot3d/subplot3d_demo.py
        # CONSIDER CONTOUR PLOTs only
        """
        # Plot the grid points with plugged-in gmrf-coef (at the grid points!).
        (meshx, meshy), _ = self.grid
        # meshx, meshy = np.meshgrid(x, y, indexing='ij')
        gridxy = np.stack((meshx, meshy), axis=-1)

        fig = plt.figure()

        plt.title('{}'.format(title))

        # plot coefficents without TP-Splines
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_wireframe(meshx, meshy, self.z.reshape(meshx.shape), color='C1')
        ax1.set_title('Coefficents at grid position')

        # plot TP-splines with plugged in coeff
        ax2 = fig.add_subplot((222), projection='3d')
        ax2.set_title('TE-Spline with plugged-in gmrf-coef.')

        ax2.plot_wireframe(meshx, meshy, self.surface(gridxy), color='C1')

        ax3 = fig.add_subplot(223)
        # plotting the correlation matrix used for sampling effects:
        ax3.imshow(self.Q, cmap='hot', interpolation='nearest')
        ax3.set_title('Precision matrix Q')

        plt.show()

        # Deprec after removing the workaround of fitting TE to GRF instead of plugging in coef
        # fig = plt.figure()
        #
        # (xmesh, ymesh), gridvec = self.grid
        # gridxy = np.stack((xmesh, ymesh), axis=-1)
        # # spline =  self.surface(gridxy) FIXM: this is new version!
        #
        # # DEPREC: scipy.interpol.bivariate Bspline input format:
        # spline = self.surface(xi=gridvec[:, 0], yi=gridvec[:, 1])
        # coord_grf = (xmesh, ymesh,
        #              self.z.reshape((xmesh.shape[0], ymesh.shape[0])).T)
        # # fIXM validate, that [0] is correct for rectangle shaped grid
        # coord_teBspline = (xmesh, ymesh,
        #                    spline.reshape((xmesh.shape[0], ymesh.shape[0])).T)
        #
        # if coord_grf is not None:
        #     if coord_teBspline is not None:
        #         ax = fig.add_subplot(121, projection='3d')
        #     else:
        #         ax = fig.add_subplot(111, projection='3d')
        #     ax.set_title('grf')
        #
        #     # Plot grf in wireframe
        #     X, Y, Z = coord_grf
        #     ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, alpha=0.7)
        #
        # # optionally plot the Bspline estimate as surface
        # if coord_teBspline is not None:
        #     if coord_grf is not None:
        #         ax1 = fig.add_subplot(122, projection='3d')
        #     else:
        #         ax1 = fig.add_subplot(111, projection='3d')
        #
        #     X, Y, Z = coord_teBspline
        #     ax1.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #                      linewidth=0, antialiased=False, alpha=0.7)
        #     ax1.set_title('B-spline estimate')
        #
        #     plt.show()


if __name__ == '__main__':
    pass