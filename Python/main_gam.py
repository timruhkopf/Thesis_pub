import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from Python.bspline import get_design, diff_mat
from Python.bspline_data import randomData

def joint_log_prob(Z, Q, sigma, gamma, tau):  # beta
    """
    :param cholQ: cholesky decomposition of precision matrix
    :param gamma: Bspline parameter vector
    :param tau: variance parameter
    """

    if Z.shape[1] != gamma.shape[0]:
        raise ValueError('Z & gamma dim are not matching')
    elif Q.shape[0] != Q.shape[1]:
        raise ValueError('Q is not square')
    elif gamma.shape[0] != Q.shape[0]:
        raise ValueError('Q & gamma dim are not matching')

    # smoothing parameter \lambda : The larger the variance \tau² of the prior distribution is relative to the variance
    # of the residuals \sigma, the less the estimation will be penalized. Consequently, we always have to interpret the
    # value of \tau^2 relative to the variance \sigma^2 that is associated with the measurement error \epsilon. We can
    # then refer \ lambda to as the noise-to-signal ratio

    # deterministic part of the gamma prior distribution (as defined in Appendix B.3.2) corresponds to a constant vector
    # representing the level of function f. With the partially improper prior, we define a flat prior for this level.
    rv_tau = tfd.InverseGamma(0.001, 0.001)
    rv_sigma = tfd.InverseGamma(0.001, 0.001)
    # rv_beta = tfd.Normal(0, 1000)
    eta = tf.linalg.matvec(Z, gamma)  # X @ beta
    rv_observed = tfd.Normal(loc=eta, scale=sigma * tf.ones([eta.shape[0]],
                                                            dtype=tf.float32))  # CONSIDER: should gamma be rv_gamma object?

    summands = [rv_tau.log_prob(tau),
                # rv_beta.log_prob(beta),
                rv_sigma.log_prob(sigma),
                # gamma's prior log_prob:
                tf.cast(tf.rank(Q) / 2, dtype=tf.float32),
                -1 * tf.math.log(tau),
                - 1 / (2 * tau) * tf.tensordot(tf.tensordot(gamma, Q, 1), gamma, 1),
                tf.reduce_sum(rv_observed.log_prob(eta))]

    summands = [tf.reshape(tensor, (1,)) for tensor in summands]

    for i, summ in enumerate(summands):
        if any(np.isnan(summ.numpy())):
            print(i, summ)
            print('sigma: {}, tau: {}'.format(sigma, tau))

    print('sigma: {}, tau: {}'.format(sigma, tau))
    return tf.add_n(summands)


# (generating data) --------------------------------------------------------
n = 4000
xgrid = (0, 10, 0.25)
fx = randomData(xgrid=xgrid,
                bspline_param={'n_basis': 7, 'coef_scale': 0.5, 'degree': 2},
                plt_it=False)

x1 = np.random.uniform(low=xgrid[0], high=xgrid[1], size=n)
mu = fx.spl(x1)
y = np.random.normal(loc=mu, scale=0.3, size=n)

plt.scatter(x1, y, alpha=0.4)
plt.scatter(x1, mu, s=2)

# (unnormalized_log_prob Example) ------------------------------------------
gamma = np.random.uniform(0, 2, 13)
Z = get_design(x1, degree=2)
sigma, tau = 10, 1

d1, K1 = diff_mat(dim=12, order=1)

# # GMRF: with rankdeficient K:
# K = np.kron(np.eye(K1.shape[0]), K1) + np.kron(K1, np.eye(K1.shape[0]))
# eigval, eigvec = np.linalg.eig(K)  # complex solution!

# convert to
sigma = tf.convert_to_tensor(sigma, tf.float32, name='sigma')
tau = tf.convert_to_tensor(tau, tf.float32, name='tau')
gamma = tf.convert_to_tensor(gamma, tf.float32, name='gamma')
Q = tf.convert_to_tensor(K1, tf.float32, name='Q')
Z = tf.convert_to_tensor(Z, tf.float32, name='Z')

# define closure, but with converted data & pen matrix
unnormalized_log_prob = lambda *args: joint_log_prob(Z, Q, *args)
# print(unnormalized_log_prob(sigma, gamma, tau))

# FIXME: Bijectors for variances
# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Exp(),  # Sigma: Maps R+ to R.
    tfp.bijectors.Identity(),  # gamma: Maps R to R.
    tfp.bijectors.Exp()  # tau: Maps R+ to R.
]

# Initialize the HMC transition kernel.
num_results = int(10e3)
num_burnin_steps = int(1e3)
bijected_hmc = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_log_prob,
        num_leapfrog_steps=3,
        step_size=1.),
    bijector=unconstraining_bijectors)

adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    bijected_hmc,
    num_adaptation_steps=int(num_burnin_steps * 0.8))

# Set the chain's start state.
initial_chain_state = [
    10. * tf.ones([1], dtype=tf.float32, name="init_sigma"),
    tf.reduce_mean(tf.cast(x1, dtype=tf.float32)) * tf.ones([13], dtype=tf.float32, name="init_gamma"),
    10 * tf.ones([1], dtype=tf.float32, name="init_tau")
]

[sigma, gamma, tau], is_accepted = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=initial_chain_state,
    kernel=adaptive_hmc,
    trace_fn=None,
    parallel_iterations=1
)

print('')
