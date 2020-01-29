import scipy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import ndsplines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from Python.bspline import diff_mat, eval_basis, get_design

# (Bspline SZENARIO) -----------------------------------------------------------
# set up penalty matrix
d1, K1 = diff_mat(dim=12, order=1)

Q = K1
eigval, eigvec = scipy.linalg.eigh(Q)

# null space eigenvectors
U0 = eigvec[:, eigval < 10 ** -3]
S0 = U0.dot(U0.T)

# draw gamma from its prior
tfd = tfp.distributions
sig = 10
rv_gamma = tfd.MultivariateNormalFullCovariance(loc=0., covariance_matrix=tf.linalg.inv(10*Q + sig * S0))
gamma = rv_gamma.sample().numpy()

# generate data & Spline
lower, upper = 0, 10  # support of x
degree = 2
l_knots = lower - degree - 1
u_knots = upper + degree + 2

X = np.random.uniform(lower, upper, 1000)
Z = get_design(X, degree=2)
mu = Z.dot(gamma)

plt.scatter(X, mu)




# (GMRF SZENARIO) --------------------------------------------------------------

# set up penalty matrix
d1, K1 = diff_mat(dim=15, order=1)

# FIXME do kronecker product for GMRF penalty
# # GMRF: with rankdeficient K:
K = np.kron(np.eye(K1.shape[0]), K1) + np.kron(K1, np.eye(K1.shape[0]))

Q = K

eigval, eigvec = scipy.linalg.eigh(Q)

# null space eigenvectors
U0 = eigvec[:, eigval < 10 ** -3]
S0 = U0.dot(U0.T)

# plt.imshow(Q, cmap='hot', interpolation='nearest')
# plt.show()
#
# plt.imshow(S0, cmap='hot', interpolation='nearest')
# plt.show()
#
# plt.imshow(np.linalg.inv(Q + 1 * S0), cmap='hot', interpolation='nearest')
# plt.show()


tfd = tfp.distributions
sig = 0.5
rv_gamma = tfd.MultivariateNormalFullCovariance(loc=0., covariance_matrix=tf.linalg.inv(Q + sig * S0))
z = rv_gamma.sample().numpy()

# fahrmeir : d = l + m - 1
# ndsplien package : l : degree, m: number of kappas
# v = m - l - 1

# v is the number of coefficients derived from degree l and number of knots m
# NOTE: given some shape of coef (v,v) and a spline degree, derive what m is:
l = 2
v = int(np.sqrt(z.shape))
m = v + l + 1

coeff = z.reshape((v, v))

# spanning the knots
x = np.linspace(0, 10, m)
y = np.linspace(0, 10, m)
meshx, meshy = np.meshgrid(x, y, indexing='ij')
gridxy = np.stack((meshx, meshy), axis=-1)

a = ndsplines.NDSpline(knots=[x, y], degrees=[l, l],
                       coefficients=coeff)

# Plot the grid points with plugged-in gmrf-coef at the above knots.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if True:
    # FIXME: CUTTING AWAY THE EXTREME EDGES IN CHOLESKY!
    ax.plot_wireframe(meshx[2:20, 2:20], meshy[2:20, 2:20], a(gridxy)[2:20, 2:20], color='C1')
else:
    ax.plot_wireframe(meshx, meshy, a(gridxy), color='C1')

ax.set_title('TE-Spline with plugged-in gmrf-coef.')
plt.show()
