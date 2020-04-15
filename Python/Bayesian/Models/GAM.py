import tensorflow as tf
import tensorflow_probability as tfp

from collections import OrderedDict

from Python.Effects.bspline import diff_mat1D, get_design
from Python.Bayesian.Models.Regression import Regression

tfd = tfp.distributions
tfb = tfp.bijectors


class GAM(Regression):
    def __init__(self, precision=diff_mat1D(dim=20, order=1)[1][1:, 1:], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.precision = tf.convert_to_tensor(precision, tf.float32)
        # due to lack of numerical precision, casting and rounding
        self.cov = tf.cast(tf.cast(tf.linalg.inv(self.precision), tf.float16), tf.float32)
        # self.cov = tf.linalg.inv(self.precision)
        assert tf.reduce_all(tf.transpose(self.cov) == self.cov)  # symmetry assertion
        self.cov_cholesky = tf.linalg.cholesky(self.cov)

        # FIXME: hyperparameter
        tau = tf.constant([5.])

        self.joint = tfd.JointDistributionNamed(OrderedDict(
            # Spline model
            # tau=tfd.InverseGamma(1., 1., name='tau'),
            W0=tfd.Uniform(-1., 1., name='w0'),
            W=tfd.MultivariateNormalTriL(  # lambda tau: # fixme: hyperparameter
                loc=tf.repeat([0.], self.cov.shape[0]),
                scale_tril=tau * self.cov_cholesky,  # fixme: tau or tau **-1
                name='w'),

            # y's variance prior
            # Consider moving sigma to seperate model, making GAM a layer
            #  detached from likelihood. problem: listparser does not know sigma
            sigma=tfd.InverseGamma(1., 1., name='sigma')))

    def likelihood_model(self, Z, W0, W, sigma):
        return tfd.Sample(tfd.Normal(
            loc=self.dense(Z, tf.concat([tf.reshape(W0, (1,)), W], axis=0)),  # mu
            scale=sigma, name='y'),
            sample_shape=1)

    def DeBoor_streaming(self):
        """might allow for adaptive knot choice by model -- to achieve the necessary
        flexibility in those regions that require it."""
        pass


if __name__ == '__main__':
    from inspect import getfullargspec
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    # NOTICE:Precision is rank deficient - and of order 1 -
    #  so one row & col is removed.
    #  the input_shape = dim of diff_mat1D before "reduction" to dim-1
    gam = GAM(
        precision=diff_mat1D(dim=20, order=1)[1][1:, 1:],
        input_shape=20, no_units=1, activation='identity')

    # get design matrix
    # NOTICE: no_basis = dim !
    # CAREFULL: .numpy()! get_design works on numpy tensors only!
    X = tfd.Uniform(0., 10.).sample(1000)
    Z = tf.convert_to_tensor(
        get_design(X=X.numpy(), degree=2, no_basis=20),
        dtype=tf.float32)  # CAREFULL: default tf.float64

    tf.reduce_sum(Z, axis=1)  # not exacly one but numerically

    # sample y:
    true_param = gam.joint.sample()
    like_param = {k: true_param[k] for k in getfullargspec(gam.likelihood_model).args[2:]}
    likelihood = gam.likelihood_model(Z, **like_param)
    y = likelihood.sample()

    # set up estimation
    init_param = gam.joint.sample()
    bijectors = {
        'W0': tfb.Identity(), 'W': tfb.Identity(),
        'tau': tfb.Exp(), 'sigma': tfb.Exp()}
    gam.unnormalized_log_prob = gam._closure_log_prob(Z, y)  # lies in Regression
    # print(gam.unnormalized_log_prob(*init_param.values()))
    # print(gam.unnormalized_log_prob(*true_param.values()))

    # FIXME: true_param to reasonable init_param
    print(true_param)

    # Consider using OLS as init for W, W0 parameters
    # adHMC = AdaptiveHMC(
    #     initial=list(true_param.values()),  # CAREFULL MUST BE FLOAT!
    #     bijectors=[bijectors[k] for k in init_param.keys()],
    #     log_prob=gam.unnormalized_log_prob)

    adHMC = AdaptiveHMC(
        initial=list(init_param.values()),  # CAREFULL MUST BE FLOAT!
        bijectors=[bijectors[k] for k in init_param.keys()],
        log_prob=gam.unnormalized_log_prob)

    # FIXME: sample_chain has no y
    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(2*10e2),
        num_results=int(10e3),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    # check acceptance rate:
    acceptance = tf.reduce_mean(tf.cast(traced.inner_results.is_accepted, tf.float32), axis=0)

    # prediction
    meanPost = adHMC.predict_mean()
    modePost = adHMC.predict_mode(gam.unnormalized_log_prob)

    # plotting

    f = lambda param: gam.dense(
        Z,
        W=tf.concat([tf.reshape(param['W0'], (1,)), param['W']], axis=0))

    f_true = f(true_param)
    f_init = f(init_param)
    f_mean = f(gam.listparser(meanPost))
    f_mode = f(gam.listparser(modePost))

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('init-, true-, mean function & sampled points')

    sns.lineplot(
        x=X.numpy(),
        y=tf.reshape(f_true, (X.shape[0],)).numpy(), ax=ax)
    sns.lineplot(
        x=X.numpy(),
        y=tf.reshape(f_init, (X.shape[0],)).numpy(), ax=ax)
    sns.scatterplot(
        x=X.numpy(),
        y=tf.reshape(y, (X.shape[0],)).numpy(), ax=ax)
    sns.lineplot(
        x=X.numpy(),
        y=tf.reshape(f_mean, (X.shape[0],)).numpy(), ax=ax)

# from collections import OrderedDict
# from inspect import getfullargspec
# class GAM:
#     def __init__(self, precision=diff_mat1D(dim=20, order=1)[1][1:, 1:], activation='identity'):
#         """
#         consider interface design like this:
#         https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization
#         # Consider De Boor's algorithm for stream conversion of X --> Z
#         Gam Unit: takes one dim observation
#
#         :param precision: full rank precision matrix for gamma[1:d] prior
#         gamma[0] is drawn from uniform distribution
#
#
#         """
#         identity = lambda x: x
#         self.activation = {
#             'relu': tf.nn.relu,
#             'tanh': tf.math.tanh,
#             'sigmoid': tf.math.sigmoid,
#             'identity': identity}[activation]
#
#         self.precision = tf.convert_to_tensor(precision, tf.float32)
#
#         # due to lack of numerical precision, casting and rounding
#         self.cov = tf.cast(tf.cast(tf.linalg.inv(self.precision), tf.float16), tf.float32)
#         assert tf.reduce_all(tf.transpose(self.cov) == self.cov)  # symmetry assertation
#         self.cov_cholesky = tf.linalg.cholesky(self.cov)
#
#         # Deprec: since piror is no longer seperated from likelihood
#         # instantate the prior
#         # self.joint_prior = self.prior()
#
#         self.joint = tfd.JointDistributionNamed(dict(
#             # Spline model
#             tau=tfd.InverseGamma(1., 1., name='tau'),
#             w0=tfd.Uniform(-1., 1., name='w0'),
#             w=lambda tau: tfd.MultivariateNormalTriL(
#                 loc=tf.repeat([0.], self.cov.shape[0]),
#                 scale_tril=tau * self.cov_cholesky,  # fixme: tau or tau **-1
#                 name='w'),
#
#             # y's variance prior
#             sigma=tfd.InverseGamma(1., 1., name='sigma')))
#
#         print('Resolving hierarchical model (variable, (parents)):\n',
#               self.joint.resolve_graph())
#
#     # @tf.function
#     def dense(self, X, W):
#         return self.activation(tf.linalg.matvec(X, W))
#
#     def likelihood(self, Z, w0, w, sigma):
#         return tfd.Sample(tfd.Normal(
#                 loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0)),  # mu
#                 scale=sigma, name='y'),
#                 sample_shape=1)
#
#     def listparser(self, tensorlist):
#         nameslist = list(self.joint._parameters['model'].keys())
#         return {k: v for k, v in zip(nameslist, tensorlist)}
#
#     def _closure_log_prob(self, X, y):
#         """A closure, to preset X, y in this model and
#         match HMC's expected model format"""
#
#         @tf.function  # NOTICE: seems to be ignored by autograph ( Cause: expected exactly one node node, found [<gast.gast.FunctionDef object at 0x7f59e006d940>, <gast.gast.Return object at 0x7f59e006da90>] )
#         def BNN_log_prob(*tensorlist):
#             """unnormalized log posterior value: log_priors + log_likelihood"""
#
#             param = self.listparser(tensorlist)
#
#             like_param = {k: param[k] for k in getfullargspec(self.likelihood_model).args[2:]}
#             likelihood = self.likelihood_model(X, **like_param)
#
#             val = self.joint.log_prob(**param) + \
#                   tf.reduce_sum(likelihood.log_prob(y))
#             print('logposterior: {}'.format(val))
#             return val
#
#
#
#
#     # Experimental (1) joint OrderedDict, joint prior + external likelihood
#     # def prior(self):
#     #     """defines the joint prior of all hierarchy levels.
#     #     :return tfd.JointDistributionNamed, which has sample & log_prob method"""
#     #     joint = tfd.JointDistributionNamed(OrderedDict(
#     #         # Spline model
#     #         tau=tfd.Gamma(0.1, 0.1, name='tau'),
#     #         w0=tfd.Uniform(-1000., 1000., name='w0'),
#     #         w=lambda tau: tfd.MultivariateNormalTriL(
#     #             loc=tf.repeat([0.], self.cov.shape[0]),
#     #             scale_tril=tau * tf.linalg.cholesky(self.cov),
#     #             name='w'),
#     #
#     #         # variance of Likelihood model
#     #         sigma=tfd.Gamma(0.1, 0.1, name='sigma')
#     #     ))
#     #
#     #     print('Resolving hierarchical model (variable, (parents)):\n',
#     #           joint.resolve_graph())
#     #     return joint
#     #
#     # def likelihood(self, Z, w0, w, sigma):
#     #     # CAREFULL due to argparsing, the ordering of the intersection of
#     #     #  self.joint_prior._parameters['model'].keys() must be the of
#     #     #  likelihood's parameters.
#     #     mu = self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0))
#     #     return tfd.Normal(loc=mu, scale=sigma, name='y') # tfd.Sample(tfd.Normal....)
#     #
#     # def _closure_log_prob(self, Z, y):
#     #     """A closure, to preset X, y in this model and match HMC's expected model format"""
#     #
#     #     # @tf.function
#     #     def GAM_log_prob(tau, w0, w, sigma):
#     #         """unnormalized log posterior value: log_priors + log_likelihood"""
#     #
#     #         gam = self.gam(Z)
#     #
#     #         return gam.log_prob(tau, w0, w, sigma, y)
#     #
#     #         # Deprec assuming paramlist is ordered dict
#     #         # likelihood = self.likelihood(
#     #         #     Z=Z, **{k: v for k, v in paramlist.items() if k in arguments})
#     #
#     #
#     #         # PREVIOUSLY:
#     #         # prior_keys = list(self.joint_prior._parameters['model'].keys())
#     #         # param = self.argparser(paramlist, prior_keys=prior_keys)
#     #         #
#     #         # likelihood = self.likelihood(Z=Z, **param)
#     #         #
#     #         # # FIXME: traceback : warning do not 'graph_parents' : tensors are strings
#     #         # return (self.joint_prior.log_prob(OrderedDict((k, v) for k, v in zip(prior_keys, paramlist))) + \
#     #         #         tf.reduce_sum(likelihood.log_prob(y)))
#     #
#     #     return GAM_log_prob
#
#     # EXPERIMENTAL (2), but working: joint sequential
#     # joint = tfd.JointDistributionSequential([
#     #     # global smoothness of Spline:
#     #     # FIXME: uninformative a, b for tfd.Gamma (tau & sigma)
#     #     tfd.Gamma(0.1, 0.1, name='tau'),  # tau
#     #
#     #     # gamma vector prior
#     #     # FIXME: tau**-1 or tau * ??
#     #     lambda tau: tfd.MultivariateNormalTriL(
#     #         loc=tf.repeat([0.], self.cov.shape[0]),
#     #         scale_tril=tau * tf.linalg.cholesky(self.cov),
#     #         name='w'),
#     #     tfd.Uniform(-1000., 1000., name='w0'),
#     #
#     #     # variance prior
#     #     tfd.Gamma(0.1, 0.1, name='sigma'),
#     #
#     #     # likelihood model
#     #     lambda sigma, w0, w: tfd.Sample(
#     #         tfd.Normal(
#     #             loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0)),  # mu
#     #             scale=sigma,
#     #             name='y'))
#     # ])
#
#
#     #@tf.function
#     def joint_log_prob(self, tau, w0, w, sigma,  y, Z):
#         # fixme: tfd.Sample should alreay reduce sum! however, the log_probs are correct!
#         # notice self.joint.sample() samples 100 observation at once! not single!
#         #  due to likelihood loc = Z @ w = f, a vector! so single sample of likelihood is of dim(f)
#         val = tf.reduce_sum(self.model(Z).log_prob(tau=tau, w0=w0, w=w, sigma=sigma, y=y))
#         print('\n\nlogprob value={}\n'.format(val))
#         print('tau = {tau},\nw0 = {w0},\nw={w},\nsigma = {sigma}\n'.format(
#             tau=tau, w0=w0, w=w, sigma=sigma, #y=y, Z=Z
#             # FIXME: tau & sigma both diverge to infinity! -- resulting log posterior is nan
#         ))
#         return val
#
#         # Consider Validation of log posterior value
#         # print(tfd.Gamma(0.1, 0.1, name='tau').log_prob(tau) + \
#         #       tfd.Uniform(-1000., 1000., name='w0').log_prob(w0) + \
#         #       tfd.MultivariateNormalTriL(
#         #           loc=tf.repeat([0.], self.cov.shape[0]),
#         #           scale_tril=tau * tf.linalg.cholesky(self.cov),
#         #           name='w').log_prob(w) + \
#         #       tfd.Gamma(0.1, 0.1, name='sigma').log_prob(sigma) + \
#         #       tfd.Normal(
#         #           loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0))[1],  # mu
#         #           scale=sigma, name='y').log_prob(y[1])
#         #       )
#
#     # def stream_designX(self):
#     #     """
#     #     consider tf.while_loop
#     #
#     #     # Douzette Masterthesis: on Spline
#     #     /home/tim/PycharmProjects/Thesis/Literature/[MAthesis Douzette] NN & Splines.pdf
#     #     with a C implementation  of (more efficient) De Boor Algorithm e.g. in C
#     #     https://github.com/AndreDouzette/BsplineNetworks/blob/master/splineOp/fixedSpline.cc"""
#     #     pass
#
#     # @tf.function
#     # def argparser(self, paramlist, prior_keys):
#     #     """parse paramlist input to ensure, that likelihood recieves only those
#     #     parameters immediately applicable to it:"""
#     #     arguments = getfullargspec(self.likelihood).args[2:]
#     #
#     #
#     #     return {prior_keys[k]: tensor for k, tensor in enumerate(paramlist) if prior_keys[k] in arguments}
#
#
# if __name__ == '__main__':
#     # TODO Precision Identity:  linear regression example!
#     gam = GAM()
