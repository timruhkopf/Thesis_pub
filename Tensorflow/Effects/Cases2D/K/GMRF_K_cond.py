from Tensorflow.Effects import Effects2D
from Tensorflow.Effects.Cases2D.K.GMRF_K import GMRF_K

class GMRF_K_cond(Effects2D):
    """due to rank deficiency of K, sampling conditional on edges"""

    def __init__(self, xgrid, ygrid, no_neighb=1, order=1, tau=1, ):
        """
        # FOLLOWING FAHRMEIR KNEIB LANG
        :param xgrid:
        :param ygrid:
        :param order: currently not supported
        :param tau:
        """
        # generate grid
        Effects2D.__init__(self,xgrid, ygrid)
        GMRF_K._construct_precision_GMRF_K(self)

        # find edges
        (meshx, meshy), gridvec = self.grid
        row_length, col_length = meshx.shape
        edges = [0, row_length - no_neighb, (col_length - no_neighb) * row_length,
                 (col_length - no_neighb) * row_length + row_length - no_neighb]

        self._sample_conditional_precision(cond_points=edges, tau=tau)
        self._generate_surface()

        self.plot_interaction(title='{}'.format(self.__class__.__name__))

if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    gmrf_condK = GMRF_K_cond(xgrid, ygrid, no_neighb=1, order=1, tau=3)