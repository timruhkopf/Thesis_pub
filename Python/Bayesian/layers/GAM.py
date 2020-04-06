import tensorflow as tf
import tensorflow_probability as tfp

from Python.Effects.bspline import diff_mat1D, get_design

tfd = tfp.distributions
tfb = tfp.bijectors


class GAM:
    def __init__(self, precision=diff_mat1D(dim=20, order=1)[1][1:, 1:],):
        """
        consider interface design like this:
        https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization
        Gam Unit: takes one dim observation

        :param precision: full rank precision matrix for gamma[1:d] prior
        gamma[0] is drawn from uniform distribution
        """
        # Consider De Boor's algorithm for stream conversion of X --> Z

        self.precision = tf.convert_to_tensor(precision, tf.float32)
        print(tf.linalg.matrix_rank(self.precision))

        # due to lack of numerical precision, casting and rounding
        self.cov = tf.cast(tf.cast(tf.linalg.inv(self.precision), tf.float16),tf.float32)
        assert tf.reduce_all(tf.transpose(self.cov) == self.cov)  # symmetry assertation

    def model(self, Z):
        """defines the joint model of both priors and likelihood
        Here, due rank deficiency of the precision, the first RV is drawn uniformly"""
        joint = tfd.JointDistributionSequential([
            # global smoothness of Spline:
            # FIXME: uninformative a, b for tfd.Gamma (tau & sigma)
            tfd.Gamma(0.1, 0.1, name='tau'),  # tau

            # gamma vector prior
            # FIXME: tau**-1 or tau * ??
            lambda tau: tfd.MultivariateNormalTriL(
                loc=tf.repeat([0.], self.cov.shape[0]),
                scale_tril=tau * tf.linalg.cholesky(self.cov),
                name='w'),
            tfd.Uniform(-1000., 1000., name='w0'),

            # variance prior
            tfd.Gamma(0.1, 0.1, name='sigma'),

            # likelihood model
            lambda sigma, w0, w: tfd.Sample(tfd.Normal(
                loc=self.dense(Z, tf.concat([tf.reshape(w0, (1,)), w], axis=0)),  # mu
                scale=sigma,
                name='y'))
        ])

        print(joint.resolve_graph())
        # x = joint.sample()
        # joint.log_prob(x)
        return joint


    @staticmethod
    @tf.function
    def dense(Z, w):
        return tf.linalg.matvec(Z, w)

    # consider for ease of problem switching use this
    # def likelihood_model(self, Z, w, sigma):
    #     """homoscedastic model"""
    #     mu = self.dense(Z, w)
    #     y = tfd.Normal(loc=mu, scale=sigma)
    #     return y, mu

    def _closure_log_prob(self, Z, y):
        """A closure, to preset X, y in this model and match HMC's expected model format"""

        @tf.function
        def GAM_log_prob(tau, w0, w, sigma):
            """unnormalized log posterior value: log_priors + log_likelihood"""
            # Fixme: BNN_log_prob(Ws, bs) sample_chain(current_state= is either
            #  tensor or list of tensors, but not multiple arguments!!!
            # Consider (*args) and argparser(*args)

            # likelihood, _ = self.likelihood_model(Z, w)

            # return (self.prior_w.log_prob(w) +
            #         self.prior_tau.log_prob(tau) +
            #         tf.reduce_sum(likelihood.log_prob(y)))

            self.model()

        return GAM_log_prob

    # def stream_designX(self):
    #     """
    #     consider tf.while_loop
    #
    #     # Douzette Masterthesis: on Spline
    #     /home/tim/PycharmProjects/Thesis/Literature/[MAthesis Douzette] NN & Splines.pdf
    #     with a C implementation  of (more efficient) De Boor Algorithm e.g. in C
    #     https://github.com/AndreDouzette/BsplineNetworks/blob/master/splineOp/fixedSpline.cc"""
    #     pass


if __name__ == '__main__':
    # TODO Precision Identity:  linear regression example!
    gam = GAM()
