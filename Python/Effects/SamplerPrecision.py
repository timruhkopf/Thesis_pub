import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import eigh



class SamplerPrecision:
    # (sampling GMRF) ----------------------------------------------------------
    # ANY such method must produce self.Q & self.z
    def _sample_with_nullspace_pen(self, Q, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
        """see penalize nullspace for doc"""
        Sigma, penQ = penalize_nullspace(Q, sig_Q, sig_Q0, threshold)
        self.Sigma = Sigma
        self.penQ = penQ
        self.z = np.random.multivariate_normal(mean=np.zeros((self.Sigma.shape[0],)), cov=self.Sigma)

        # Deprec
        # rv_z = tfd.MultivariateNormalFullCovariance(  # FIXME remove tf dependence
        #     covariance_matrix=self.Sigma,
        #     loc=0.)
        # self.z = rv_z.sample().numpy()

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
            plt.scatter(np.arange(eigval.size), eigval)  # fixme save for later

            self.B = eigvec.dot(np.diag(np.sqrt(eigval)))
            self.z = self.B.dot(theta)

        elif decomp == 'choleskyB':
            # RUE 2005 / 99: for decomp look at sample_GMRF
            self.B = np.linalg.cholesky(Q).T  # FIXME: check if .T is correct here
            self.z = self.B.T.dot(theta)
            # Deprec: backsolve produces bad gmrfs
            # self.z = _sample_backsolve(L= self.B, z=theta)

        else:
            raise ValueError('decomp is not propperly specified')

    def _sample_conditional_precision(self, cond_points, tau):
        """

        :param cond_points: list of index positions in grid to be sampled
        :param tau:
        :return:
        """
        _, gridvec = self.grid

        mask = np.ones(len(gridvec), np.bool)
        mask[cond_points] = 0

        selector = np.zeros((len(cond_points), gridvec.shape[0]), dtype=bool)
        selector[np.arange(len(cond_points)), [cond_points]] = True
        Q_BB = selector.dot(self.Q).dot(selector.T)

        intermediate = self.Q[mask, :]
        Q_AA = intermediate[:, mask]
        Q_AB = intermediate[:, ~mask]

        xb = np.random.multivariate_normal(mean=np.zeros(len(cond_points)),
                                           cov=0.1 * np.linalg.inv(Q_BB))
        xa = np.random.multivariate_normal(mean=-tau * Q_AB.dot(xb - 0),
                                           cov=tau * np.linalg.inv(Q_AA))  # fixme definiteness

        z = np.zeros(shape=gridvec.shape[0])
        z[mask] = xa
        z[~mask] = xb
        self.z = z

        Q1 = np.concatenate([Q_AA, Q_AB], axis=1)
        Q2 = np.concatenate([Q_AB.T, Q_BB], axis=1)
        self.Q = np.concatenate([Q1, Q2], axis=0)

        plt.imshow(Q_BB, cmap='hot', interpolation='nearest')  # fixme:save for later


# Deprec
# def _sample_backsolve(L, z, mu=0):
#     """
#     # AUXILIARY METHOD
#     Following RUE HELD 2005 slide 52 ff. explicitly slide 56
#     Solve eq. system L @ x = z for x.
#     if z ~ MVN(0,I) and Q = LL^T from cholesky, this allows to generate
#     x ~ MVN(0, Q^(-1)
#
#     :param L: upper triangular matrix
#     :param z: vector
#     :param mu: additional mean vector
#     :return x: vector
#     """
#     if L.shape[1] != z.size:
#         raise ValueError('improper dimensions')
#
#     x = np.zeros(L.shape[0])
#     x[-1] = z[-1] / L[-1, -1]
#
#     for i in reversed(range(0, L.shape[0] - 1)):
#         x[i] = (z[i] - L[i].dot(x)) / L[i, i]
#
#     return x + mu


def penalize_nullspace(Q, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
    """
    Nullspace penalty for rank deficient Precision matrix, to get rank sufficent Covariance matrix

    SIMON WOOD comments in [WOOD (on general)] SmoothModels:
    Adding such a penalty to all the smooth terms in the model
    allows smoothing parameter selection to remove terms from the
    model altogether.

    :param Q: Precision Matrix
    :param sig_Q: inverse variance factor (sig_Q * Q)
    :param sig_Q0: penalty factor
    :param threshold: numerical value, determining which eigenvals are numerical zero

    :return: Covariance : inverse of the resulting penalized precision matrix:
    (sig_Q * Q + sig_Q0 * S0)**-1 with S0 = U0 @ U0.T, where U0 corresponds
    to the matrix of Eigenvectors corresponding to those Eigenvalues < threshold
    """
    eigval, eigvec = eigh(Q)

    # (numeric precision) null space eigenvectors
    U0 = eigvec[:, eigval < threshold]
    S0 = U0.dot(U0.T)
    penQ = sig_Q * Q + sig_Q0 * S0
    penSIGMA = np.linalg.inv(penQ)

    print('Eigenvalues: ', eigval, '\n')
    print('Nullspace Matrix: ', U0)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(np.arange(eigval.size), eigval)
    ax1.set_title('eigenvalues of Q')

    ax2 = fig.add_subplot(222)
    ax2.imshow(penSIGMA, cmap='hot', interpolation='nearest')
    ax2.set_title('penSIGMA')

    ax3 = fig.add_subplot(223)
    ax3.imshow(Q, cmap='hot', interpolation='nearest')
    ax3.set_title('Q')
    plt.show()

    return penSIGMA, penQ
