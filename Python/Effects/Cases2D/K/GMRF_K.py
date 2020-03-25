import numpy as np
from Python.Effects.Effect2D import Effects2D
from Python.Effects.bspline import diff_mat2D


class GMRF_K(Effects2D):
    """Construct GMRF from K_order=D_order @ D_order with nullspace penalty on K.
    coef are drawn from the resulting MVN(0, penK)"""

    def __init__(self, xgrid, ygrid, order=1, sig_Q=0.01, sig_Q0=0.01, threshold=10 ** -3):
        """
        # FOLLOWING FAHRMEIR KNEIB LANG
        :param xgrid:
        :param ygrid:
        :param order: CURRENTLY HIGHER ORDERS OF K are not supported due to theoretical issues. implementation works fine
        :param sig_Q:
        :param sig_Q0:
        :param threshold:
        """
        # generate grid
        super(GMRF_K, self).__init__(xgrid, ygrid)

        self._construct_precision_GMRF_K()
        self._sample_with_nullspace_pen(Q=self.Q, sig_Q=sig_Q, sig_Q0=sig_Q0, threshold=threshold)
        self._generate_surface()

        self.plot_interaction(title='{} with {}'.format(self.__class__.__name__, 'Nullspace penalty to MVN'))

    def _construct_precision_GMRF_K(self):
        (meshx, _), _ = self.grid
        no_coef = meshx.shape[0]

        D1, D2, K = diff_mat2D(dim=no_coef)
        self.Q = K

        print('rank of Q: ', np.linalg.matrix_rank(self.Q))
        print('shape of Q: ', self.Q.shape)



if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    gmrf_k = GMRF_K(xgrid, ygrid, order=1, sig_Q=1, sig_Q0=0.8)

    print('')
