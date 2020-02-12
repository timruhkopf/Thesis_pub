import numpy as np
from scipy.interpolate import BSpline
from collections import deque
from scipy.linalg import eigh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


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
    eval_basis.Z = [BSpline.basis_element(t=seq, extrapolate=False) for seq in window(knots, n=degree + 3)]
    z = np.stack([B(x) for B in eval_basis.Z])

    return np.nan_to_num(z)


def get_design(X, degree):
    """
    Broadcast eval_basis to a 1dim array X, to obtain the corresponding
    Designmatrix Z in Basis representation

    :param X:
    :param basis_param:
    :return:

    # consider eval_basis decorator - to ensure, The callable Basis Vector is caluclated only once
    """

    # construct degree and X's support dependent number of outer knots
    l_knot = X.min() - degree - 1
    u_knot = X.max() + degree + 2

    # generate apporpriate knots & associated metrics
    get_design.degree = degree
    get_design.knots = np.arange(l_knot, u_knot, 1)
    get_design.num_basis = get_design.knots.shape[0] - (degree + 2)

    # obtain designmatrix
    Z = np.zeros((X.__len__(), get_design.num_basis))
    for i, obs in enumerate(X):
        Z[i, :] = eval_basis(obs, knots=get_design.knots, degree=degree)

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
    # consider second order column D (D2)
    # d1, k1 = diff_mat1D(dim, order=2)
    # D2 = np.kron(d1, np.eye(4))

    # D2 = np.kron(d1, np.eye(dim))[:2*dim, :dim**2]

    # two dimensional difference matrix
    D1 = np.kron(np.eye(dim), d1)
    # D2 = np.append(np.diag(np.repeat(-1, dim**2 - dim), k=0), np.zeros((dim**2 - dim, dim)), axis=1) + \
    #     np.append(np.zeros((dim**2 - dim, dim)), np.diag(np.repeat(1, dim**2 - dim), k=0), axis=1)

    K1 = np.kron(np.eye(dim), k1)
    K2 = D2.T.dot(D2)
    # K2 = np.kron(np.eye(dim), K2)
    K = K1 + K2

    return D1, D2, K


if __name__ == '__main__':

    D1, D2, K = diff_mat2D(dim=9, order=1)

    # ( ) Check the diff_mat for different values -----------------------------
    for order in [1, 2, 3, 4, 5]:
        d, K = diff_mat1D(dim=5, order=order)
        print('order {order}\n, D{order}: \n {d}, \n K{order} \n{K}, \n'.format(order=order, d=d, K=K))

    # analoge version
    # dim = 5
    # d1 = np.diag(np.repeat(-1, dim), k=0) + np.diag(np.repeat(1, dim - 1), k=1)
    # d1 = d1[:-1, :]
    # #d2 = d1.T.dot(d1)[1:-1, :]
    # d2 = d1[:-1,:-1].dot(d1)
    # d3 = d1[:-2,:-2].dot(d2)
    # d4 = d1[:-3, :-3].dot(d3)

    # (0) Check eval_basis ----------------------------------------------------
    # carefull to get the boundary regions right! Through in extra kappas, such that the
    lower, upper = 0, 10  # support of x
    degree = 2
    l_knots = lower - degree - 1
    u_knots = upper + degree + 2

    X = np.random.uniform(lower, upper, 5)

    z = eval_basis(x=upper, knots=np.arange(l_knots, u_knots, 1), degree=degree)

    print(z)
    print('rowsum: ', z.sum())

    z = eval_basis(x=lower, knots=np.arange(l_knots, u_knots, 1), degree=degree)

    print(z)
    print('rowsum: ', z.sum())

    # (1) get_design matrix ----------------------------------------------------
    Z = get_design(X, degree=2)
    print(Z, '\n',
          'rowsum: ', Z.sum(axis=1), '\n',
          'number of basis: ', get_design.num_basis)

    print(eval_basis(X[0], knots=get_design.knots, degree=2))

    # (1.1) Least squares example: ---------------------------------------------
    # Data generation
    n = 100
    X = np.stack([np.ones(n), np.random.uniform(0, 10, n)], axis=1)
    beta = np.array([4, -2])
    mu = X.dot(beta)
    y = mu + np.random.normal(loc=0, scale=1, size=n)

    # basis extention & OLS fit
    Z = get_design(X[:, 1], degree=2)
    OLS = lambda Z, y: np.linalg.inv(Z.T.dot(Z)).dot(Z.T).dot(y)
    beta_hat = OLS(Z, y)
    y_hat = Z.dot(beta_hat)

    # Metrics
    resid = y - y_hat
    bias = (mu - y_hat) ** 2
    print('residual sum: ', sum(resid))
    print('bias sum: ', sum(bias))

    import matplotlib.pyplot as plt

    plt.scatter(X[:, 1], y)
    plt.scatter(X[:, 1], y_hat)
    plt.scatter(X[:, 1], mu)
    plt.show()
