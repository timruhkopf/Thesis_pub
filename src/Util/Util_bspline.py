import numpy as np
from scipy.interpolate import BSpline
from collections import deque


# CONSIDER: def fnc: Recursive Bspline Definiton, returning a callable,
# in order to be plugged into eval_basis instead of BSpline.basis_element

def window(seq, n=2):
    """
    Moving Window Sequences
    generator object to yield a FIFO sequence on seq of length n

    # CODE SOURCE: https://stackoverflow.com/a/6822761
    :param seq: list or ndarray
    :param n: length of yielded sequences
    """
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win


def eval_basis(x, knots=np.arange(0, 20, 1), degree=2):
    """
    Design Vector in Basis representation for a single observation point.
    :param x: int or float: observation point, at which to evaluate the basis fnc
    :param knots: ndarray of knots. Be Carefull, to put sufficient outter knots!
    :param degree: Basis degree
    :return: ndarray z (dim==1), which is x evaluated at all Bspline Basis
             B_{ks,..., k(s+t)}(x) spanned on the knots

    """

    # from Bspline.basis_element doc on why n=degree + 2:
    #     The order of the b-spline, `k`, is inferred from the length of `t` as
    #         ``len(t)-2``. The knot vector is constructed by appending and prepending
    #         ``k+1`` elements to internal knots `t`.

    # vector of callable B_{ks,..., k(s+t)}(x) with appropriate
    Z = [BSpline.basis_element(t=seq, extrapolate=False) for seq in window(knots, n=degree + 3)]
    z = np.stack([B(x) for B in Z])

    return np.nan_to_num(z)


def get_design(X, degree, no_basis):
    """
    Broadcast eval_basis to a 1dim array X, to obtain the corresponding
    Designmatrix Z in Basis representation

    :param X:
    :param degree:
    :param no_basis: number of basis functions (dim of gamma)
    :return:

    # consider eval_basis decorator - to ensure, The callable Basis Vector is caluclated only once
    """

    # construct degree and X's support dependent number of outer knots
    # FIXME: +-1 is not reasoned!, but introduced to make rowsum Z == 1
    l_knot = X.min() - 1  # - degree - 1
    u_knot = X.max() + 1  # + degree + 2

    # generate apporpriate knots & associated metrics
    no_inner_knots = no_basis - degree + 1
    total_no_knots = no_inner_knots + 2 * degree

    h = (u_knot - l_knot) / no_inner_knots

    knots = np.linspace(l_knot - 2 * h, u_knot + 2 * h, num=total_no_knots + 1)
    # num_basis = knots.shape[0] - (degree + 2)

    # obtain designmatrix
    Z = np.zeros((X.__len__(), no_basis))
    for i, obs in enumerate(X):
        Z[i, :] = eval_basis(obs, knots=knots, degree=degree)

    return Z


def diff_mat1D(dim, order=1):
    """
    :param dim: the dimension of gamma vector (i.e. number of basis dimensions)
    :param order: difference order D_r = D_1 [:-r-1, :-r-1] @ D_r-1
    :return: tupel: difference matrix of order,
    difference penalty matric of this order
    """
    # first order difference matrix: shape: (dim-1) x (dim)
    d1 = np.diag(np.repeat(-1, dim), k=0) + np.diag(np.repeat(1, dim - 1), k=1)
    d1 = d1[:-1, :]

    # d1.shape == (dim - 1, dim)

    # higher order difference matrices
    r = 1
    dr = d1
    while r < order:
        dr = d1[:-r, :-r].dot(dr)
        r += 1
        # dr.shape == (dim - r, dim)

    K = dr.T.dot(dr)
    return dr, K


def diff_mat2D(dim):
    """
    \gamma^T K \gamma =   \gamma^T (I_(d2) kron K1 + K2 kron I_(d1) ) \gamma
    with K1 = D1^T D1 and  K2 = D2^T D2 where D1 and D2 are 1st order difference matrices
    in z1 & z2 direction respectively.

    This function assumes a square grid

    :param dim: row length of grid (1d dim)

    :return: D1 (2Drowwisediff), D2 (2Dcolumnwisediff), K (2Dprecision)
    """
    # dim = int(np.sqrt(dim))  # FIXME: ASSUMING SQUARE GRID!!
    # one dimensional (rowwise) difference matrices
    d1, k1 = diff_mat1D(dim, order=2)
    d2 = d1.T

    D2 = np.kron(d2, np.eye(dim)).T

    # two dimensional difference matrix
    D1 = np.kron(np.eye(dim), d1)

    K1 = np.kron(np.eye(dim), k1)
    K2 = D2.T.dot(D2)

    K = K1 + K2

    return D1, D2, K


if __name__ == '__main__':
    pass

    # analoge version
    # dim = 5
    # d1 = np.diag(np.repeat(-1, dim), k=0) + np.diag(np.repeat(1, dim - 1), k=1)
    # d1 = d1[:-1, :]
    # #d2 = d1.T.dot(d1)[1:-1, :]
    # d2 = d1[:-1,:-1].dot(d1)
    # d3 = d1[:-2,:-2].dot(d2)
    # d4 = d1[:-3, :-3].dot(d3)
