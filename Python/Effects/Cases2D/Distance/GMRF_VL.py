import numpy as np
from Python.Effects.Effect2D import Effects2D


class GMRF_VL(Effects2D):
    # NOT WORKING
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1,
                 radius=4, rho=0.7, tau=1, tau1=1, decomp=['eigenB', 'choleskyB'][0]):
        # generate grid
        super(GMRF_VL, self).__init__(xgrid, ygrid)

        self._grid_distances(corrfn, lam, phi, delta, gridvec=self.grid[1])
        self._construct_precision_GMRF_VL(radius, rho, tau)  # FIXME: CAREFULL: HERE  TAU IS CORRECT DUE TO PRECISION

        self._sample_uncond_from_precisionB(self.Q, tau1, decomp)
        self._generate_surface()

        self.plot_interaction(title='{} with {}'.format(self.__class__.__name__, decomp))

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


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    gmrf_vl = GMRF_VL(xgrid, ygrid)
