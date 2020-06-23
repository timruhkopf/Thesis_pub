from Tensorflow.Effects.Effect1D import Effects1D
from Tensorflow.Effects.bspline import diff_mat1D


class Bspline_K(Effects1D):

    def __init__(self, xgrid, no_coef=20, order=2, sig_Q=1, sig_Q0=1, threshold=10 ** -3, degree=2):
        """
        BSPLINE with K matrix & nullspace_penalty sampling
        :param xgrid:
        :param no_coef: number of Bspline coefficents/ no of bases
        :param order: order of difference K_order = D_order @ D_order
        :param sig_Q: (sig_Q*Q + sig_Q0* S0)**-1 with S0 nullspace eigenvector matrix
        :param sig_Q0:
        # with K matrix and proper prior (nullspace penalty) beware, sig_Q & sig_Q0
        # are factors before inverting Q - so smaller values mean larger variance!
        :param threshold: determining numerical zero in eigenvalues
        :param degree: Bspline degree
        """
        super(Bspline_K, self).__init__(xgrid)
        self.Q = diff_mat1D(dim=no_coef, order=order)[1]
        self._sample_with_nullspace_pen(self.Q, sig_Q, sig_Q0, threshold)
        self._generate_bspline(degree)
        self.plot_bspline()


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)

    bspline_k = Bspline_K(xgrid, no_coef=10, order=1, sig_Q=0.1, sig_Q0=0.01, threshold=10 ** -3)
