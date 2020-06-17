from itertools import product as prd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Surface generation
from scipy.spatial.distance import pdist, squareform
import ndsplines

# rejection sampling
from scipy.stats import kde
from Tensorflow.Effects.SamplerPrecision import SamplerPrecision


class Effects2D(SamplerPrecision):
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

        self.grid = self._generate_grid(xgrid, ygrid)

    # (helper functions) -------------------------------------------------------
    def _generate_grid(self, xgrid, ygrid):
        x, y = np.arange(xgrid[0], xgrid[1], xgrid[2]), \
               np.arange(ygrid[0], ygrid[1], ygrid[2])
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

    # (class methods) ----------------------------------------------------------
    def log_prob(self):
        pass

    def _generate_surface(self, l=2):
        """
        Generate a 2d Surface on a square grid from TE-Splines whose
        coefficients originated from a Random field.

        :param l: The ND-splines degreee
        :return: ndsplines.NDSpline.__call__ object, allows to evaluate the
        exact surface value: fxy.surface(np.stack([x, y], axis=-1))

        """

        # fahrmeir : d = l + m - 1
        # ndspline package : l : degree, m: number of kappas
        # v = m - l - 1
        # v is the number of coefficients derived from degree l and number of knots m

        # given some shape of coef (v,v) and a spline degree, derive what m is:
        v = int(np.sqrt(self.z.shape))
        m = v + l + 1

        # spanning the knots
        x = np.linspace(self.xgrid[0], self.xgrid[1], m)
        y = np.linspace(self.ygrid[0], self.ygrid[1], m)

        # Tensorproduct Splines with plugged in coefficents
        coeff = self.z.reshape((v, v))  # FIXME: this assumes a square grid!
        a = ndsplines.NDSpline(knots=[x, y], degrees=[l, l],
                               coefficients=coeff)

        # INTERPOLATION Function for Datapoints
        self.surface = a.__call__

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

    def plot_rejected_contour(self, nbins=300):
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


if __name__ == '__main__':
    pass
