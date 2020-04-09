import tensorflow as tf
import tensorflow_probability as tfp

from Python.Effects.bspline import diff_mat1D, get_design

# from collections import OrderedDict
# from inspect import getfullargspec

tfd = tfp.distributions
tfb = tfp.bijectors


class GAM:
    def __init__(self, precision=diff_mat1D(dim=20, order=1)[1][1:, 1:]):
        """
        consider interface design like this:
        https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization
        Gam Unit: takes one dim observation

        :param precision: full rank precision matrix for gamma[1:d] prior
        gamma[0] is drawn from uniform distribution
        """
        # Consider De Boor's algorithm for stream conversion of X --> Z

        self.precision = tf.convert_to_tensor(precision, tf.float32)
        # print(tf.linalg.matrix_rank(self.precision))

        # due to lack of numerical precision, casting and rounding
        self.cov = tf.cast(tf.cast(tf.linalg.inv(self.precision), tf.float16), tf.float32)
        assert tf.reduce_all(tf.transpose(self.cov) == self.cov)  # symmetry assertation
        self.cov_cholesky = tf.linalg.cholesky(self.cov)

        # Deprec: since piror is no longer seperated from likelihood
        # instantate the prior
        # self.joint_prior = self.prior()

    @staticmethod
    @tf.function
    def dense(Z, w):
        """extra method, as it is used in likelihood & prediction"""
        return tf.linalg.matvec(Z, w)

    def model(self, Z):
        self.joint = tfd.JointDistributionNamed(dict(
            # Spline model
            tau=tfd.InverseGamma(1., 1., name='tau'),
            w0=tfd.Uniform(-1., 1., name='w0'),
            w=lambda tau: tfd.MultivariateNormalTriL(
                loc=tf.repeat([0.], self.cov.shape[0]),
                scale_tril=tau * self.cov_cholesky, # fixme: tau or tau **-1
                name='w'),

            # y's variance prior
            sigma=tfd.InverseGamma(1., 1., name='sigma'),

            # Likelihood model
            y=lambda w0, w, sigma: tfd.Sample(tfd.Normal(
                loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0)),  # mu
                scale=sigma, name='y'),
                sample_shape=1)

            # trial:
            # y=lambda w0, w, sigma: tfd.Normal(
            #     loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0)),  # mu
            #     scale=sigma, name='y')

            # Experimental (0) delegate likelihood from joint
            # y=lambda w0, w, sigma: tfd.Sample(self.likelihood(
            #     Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0), sigma
            # )
            ))

        print('Resolving hierarchical model (variable, (parents)):\n',
              self.joint.resolve_graph())
        return self.joint

    # Experimental (0)
    # def likelihood(self, Z, w, sigma):
    #     return tfd.Normal(
    #         loc=self.dense(Z, w),  # mu
    #         scale=sigma, name='y')

    # Experimental (1) joint OrderedDict, joint prior + external likelihood
    # def prior(self):
    #     """defines the joint prior of all hierarchy levels.
    #     :return tfd.JointDistributionNamed, which has sample & log_prob method"""
    #     joint = tfd.JointDistributionNamed(OrderedDict(
    #         # Spline model
    #         tau=tfd.Gamma(0.1, 0.1, name='tau'),
    #         w0=tfd.Uniform(-1000., 1000., name='w0'),
    #         w=lambda tau: tfd.MultivariateNormalTriL(
    #             loc=tf.repeat([0.], self.cov.shape[0]),
    #             scale_tril=tau * tf.linalg.cholesky(self.cov),
    #             name='w'),
    #
    #         # variance of Likelihood model
    #         sigma=tfd.Gamma(0.1, 0.1, name='sigma')
    #     ))
    #
    #     print('Resolving hierarchical model (variable, (parents)):\n',
    #           joint.resolve_graph())
    #     return joint
    #
    # def likelihood(self, Z, w0, w, sigma):
    #     # CAREFULL due to argparsing, the ordering of the intersection of
    #     #  self.joint_prior._parameters['model'].keys() must be the of
    #     #  likelihood's parameters.
    #     mu = self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0))
    #     return tfd.Normal(loc=mu, scale=sigma, name='y') # tfd.Sample(tfd.Normal....)
    #
    # def _closure_log_prob(self, Z, y):
    #     """A closure, to preset X, y in this model and match HMC's expected model format"""
    #
    #     # @tf.function
    #     def GAM_log_prob(tau, w0, w, sigma):
    #         """unnormalized log posterior value: log_priors + log_likelihood"""
    #
    #         gam = self.gam(Z)
    #
    #         return gam.log_prob(tau, w0, w, sigma, y)
    #
    #         # Deprec assuming paramlist is ordered dict
    #         # likelihood = self.likelihood(
    #         #     Z=Z, **{k: v for k, v in paramlist.items() if k in arguments})
    #
    #
    #         # PREVIOUSLY:
    #         # prior_keys = list(self.joint_prior._parameters['model'].keys())
    #         # param = self.argparser(paramlist, prior_keys=prior_keys)
    #         #
    #         # likelihood = self.likelihood(Z=Z, **param)
    #         #
    #         # # FIXME: traceback : warning do not 'graph_parents' : tensors are strings
    #         # return (self.joint_prior.log_prob(OrderedDict((k, v) for k, v in zip(prior_keys, paramlist))) + \
    #         #         tf.reduce_sum(likelihood.log_prob(y)))
    #
    #     return GAM_log_prob

    # EXPERIMENTAL (2), but working: joint sequential
    # joint = tfd.JointDistributionSequential([
    #     # global smoothness of Spline:
    #     # FIXME: uninformative a, b for tfd.Gamma (tau & sigma)
    #     tfd.Gamma(0.1, 0.1, name='tau'),  # tau
    #
    #     # gamma vector prior
    #     # FIXME: tau**-1 or tau * ??
    #     lambda tau: tfd.MultivariateNormalTriL(
    #         loc=tf.repeat([0.], self.cov.shape[0]),
    #         scale_tril=tau * tf.linalg.cholesky(self.cov),
    #         name='w'),
    #     tfd.Uniform(-1000., 1000., name='w0'),
    #
    #     # variance prior
    #     tfd.Gamma(0.1, 0.1, name='sigma'),
    #
    #     # likelihood model
    #     lambda sigma, w0, w: tfd.Sample(
    #         tfd.Normal(
    #             loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0)),  # mu
    #             scale=sigma,
    #             name='y'))
    # ])


    #@tf.function
    def joint_log_prob(self, tau, w0, w, sigma,  y, Z):
        # fixme: tfd.Sample should alreay reduce sum! however, the log_probs are correct!
        # notice self.joint.sample() samples 100 observation at once! not single!
        #  due to likelihood loc = Z @ w = f, a vector! so single sample of likelihood is of dim(f)
        val = tf.reduce_sum(self.model(Z).log_prob(tau=tau, w0=w0, w=w, sigma=sigma, y=y))
        print('\n\nlogprob value={}\n'.format(val))
        print('tau = {tau},\nw0 = {w0},\nw={w},\nsigma = {sigma}\n'.format(
            tau=tau, w0=w0, w=w, sigma=sigma, #y=y, Z=Z
            # FIXME: tau & sigma both diverge to infinity! -- resulting log posterior is nan
        ))
        return val

        # Consider Validation of log posterior value
        # print(tfd.Gamma(0.1, 0.1, name='tau').log_prob(tau) + \
        #       tfd.Uniform(-1000., 1000., name='w0').log_prob(w0) + \
        #       tfd.MultivariateNormalTriL(
        #           loc=tf.repeat([0.], self.cov.shape[0]),
        #           scale_tril=tau * tf.linalg.cholesky(self.cov),
        #           name='w').log_prob(w) + \
        #       tfd.Gamma(0.1, 0.1, name='sigma').log_prob(sigma) + \
        #       tfd.Normal(
        #           loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0))[1],  # mu
        #           scale=sigma, name='y').log_prob(y[1])
        #       )

    # def stream_designX(self):
    #     """
    #     consider tf.while_loop
    #
    #     # Douzette Masterthesis: on Spline
    #     /home/tim/PycharmProjects/Thesis/Literature/[MAthesis Douzette] NN & Splines.pdf
    #     with a C implementation  of (more efficient) De Boor Algorithm e.g. in C
    #     https://github.com/AndreDouzette/BsplineNetworks/blob/master/splineOp/fixedSpline.cc"""
    #     pass

    # @tf.function
    # def argparser(self, paramlist, prior_keys):
    #     """parse paramlist input to ensure, that likelihood recieves only those
    #     parameters immediately applicable to it:"""
    #     arguments = getfullargspec(self.likelihood).args[2:]
    #
    #
    #     return {prior_keys[k]: tensor for k, tensor in enumerate(paramlist) if prior_keys[k] in arguments}


if __name__ == '__main__':
    # TODO Precision Identity:  linear regression example!
    gam = GAM()
