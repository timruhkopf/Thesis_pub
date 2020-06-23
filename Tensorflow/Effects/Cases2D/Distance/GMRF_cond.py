import numpy as np
from Tensorflow.Effects.Cases2D.Distance.GMRF import GMRF
from Tensorflow.Effects import Effects2D


class GMRF_cond(Effects2D):  # FIXME GMRF_cond produces periodical results
    # original function needs refactoring, but works
    def __init__(self, xgrid, ygrid, corrfn='gaussian', lam=1, phi=0, delta=1,
                 radius=4, tau=1, no_neighb=4, decomp=['draw_normal', 'cholesky'][0], seed=1337):

        Effects2D.__init__(self, xgrid, ygrid)
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

        # self._sample_conditional_gmrf(corrfn, lam, no_neighb, decomp, radius, tau, seed)
        GMRF._construct_precision_GMRF(self, radius, corrfn, lam, phi, delta)
        self._sample_conditional_precision(cond_points=edges, tau=tau)
        self._generate_surface()

        self.plot_interaction(title='{} with {}'.format(self.__class__.__name__, decomp))


if __name__ == '__main__':
    xgrid = (0, 10, 0.5)  # FIXME: due to _generate_surface, these must span an square grid!
    ygrid = xgrid

    # FIXME: conditional effect's edges are 'edgy'
    cond_gmrf = GMRF_cond(xgrid=xgrid, ygrid=ygrid, radius=4, tau=1, no_neighb=1)
