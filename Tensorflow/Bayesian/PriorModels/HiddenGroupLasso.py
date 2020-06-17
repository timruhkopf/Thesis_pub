from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Tensorflow.Bayesian.PriorModels.Hidden import Hidden


# CONSIDER: bayesian_penalization.pdf: their priors can be restated as
# gaussian scale mixtures (for a primer on those see:
# https://stats.stackexchange.com/questions/174502/what-are-gaussian-scale-mixtures-and-how-to-generate-samples-of-gaussian-scale
# Notes:
# (1) scale mixture of normals formulation can be computationally more efficient.
# (2) For the full Bayesian approach, we will use standard half-Cauchy
#  priors for the penalty parameters as a robust default prior choice.
#  note that other choices are possible as well.
# (3) The group lasso (Yuan and Lin, 2006) & (Kyung et al. 2010)
#  is a generalization of the lasso primarily aimed at improving performance
#  when predictors are grouped
#  β_g |τ²_g, σ² ∼ MVN(0 , σ² τ²_g I_{m_g} )
#  τ²_g |λ² ~ Ga((m_g + 1)/2,  λ²/2)
#  λ ∼ half-Cauchy(0, 1)
#  where β is split in G subvectors (grouped parameters) with m_g being the dim
#  of subvector β_g. Notice, essentially the group's prior is independent to all
#  other groups' priors. the special case of group size m_g=1 for all g is the
#  basic bayesian lasso
#  CAREFULL: to decide as to whether the group (of weights from a variable -
#   in my context), either there is a single group & the independence assumption
#   is used extensively (noting, that this will allow to proceed with multiple
#   gam effects natrually) OR shrink all of the other effects too!

# (4) SPIKE & SLAB priors: a prior mixture of flat & spiked (point mass) distributions to
#  model sparse high-dimensional data. they put a positive mass at zero (to
#  induce sparsity) and a heavy tailed continuous distribution (to identify
#  the non-zero coefficients). In the formulation of bayesian_penalization.pdf
#  the binary hyperparametervector gamma, which assigns, which coefficient has a spike
#  and which a slab prior, this may just as well be abused to make one entry of
#  gamma correspond to a group!

# CAREFULL: HERNANDEZ on SPIKE & SLAB
#  Posterior distributions generated by spike-and-slab priors are often multi-modal
# The relevant features are identified by using standard spike-and-slab priors
# on the coefficients of each task, where these priors share the same binary
# latent variables. In particular, it does not allow to introduce prior
# knowledge on groups of features that are believed to be jointly relevant or
# irrelevant for the same task.

# RIDGE PRIOR (bayesian_penalization.pdf Eq. (8))
# instead of fixing the tau values consider, that ridge priors for some
# β_j |λ, σ² ~ N(0, σ²/λ)
# λ ~ halfCauchy(0,1)
# IS THERE A GROUP VERSION?

class HiddenGroupLasso(Hidden):
    def __init__(self, input_shape, no_units=10, activation='relu'):
        """
        The group lasso (Yuan and Lin, 2006) & (Kyung et al. 2010)
        is a generalization of the lasso primarily aimed at improving performance
        when predictors are grouped
        β_g |τ²_g, σ² ∼ MVN(0 , σ² τ²_g I_{m_g} )
        τ²_g |λ² ~ Ga((m_g + 1)/2,  λ²/2)
        λ ∼ half-Cauchy(0, 1)
        where β is split in G subvectors (grouped parameters) with m_g being the dim
        of subvector β_g. Notice, essentially the group's prior is independent to all
        other groups' priors. the special case of group size m_g=1 for all g is the
        basic bayesian lasso
        In this implementation, the first variable is assigned the Group Lasso prior only!
        """

        super().__init__(input_shape, no_units, activation)
        # self.input_shape  == m_g

        self.no_shrinked = 1  # in this implementation only one shrinked variable is considered!
        tau = tf.constant([5.])
        self.joint = tfd.JointDistributionNamed(OrderedDict(
            lam=tfd.HalfCauchy(0., 1.),
            tausq_group=lambda lam: tfd.Gamma((input_shape + 1) / 2., lam ** 2 / 2.),

            # FIXME: SIGMA does not exist here (likelihood's sigma) but by
            #  literature should be to ensure unimodality! would be:
            # W_shrinked=lambda sigma, tausq_group: tfd.Normal(0., sigma * tausq_group),
            W_shrinked=lambda tausq_group: tfd.Normal(
                tf.repeat(0., no_units), 1. * tausq_group),

            # the rest is usual Hidden Unit
            # tau=tfd.InverseGamma(1., 1.),

            # deprec
            # W=tfd.Sample(  # consider lambda tau:
            #     distribution=tfd.Normal(0., tau),
            #     sample_shape=(no_units, input_shape - self.no_shrinked)),
            # b=tfd.Normal(loc=tf.repeat(0., no_units), scale=1.)

            W=tfp.distributions.Sample(
                tfd.Normal(0., tau), sample_shape=(no_units, input_shape - self.no_shrinked), validate_args=False,
                name=None),
            b=tfp.distributions.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=no_units)

        ), name='GroupLasso')

        # notice, that W_shrinked is not available to "public" to ensure a
        # unique interface for self.dense across all Hidden classes. i.e. W is
        # a single matrix!
        self.parameters = list(self.joint._parameters['model'].keys())
        self.parameters.remove('W_shrinked')

        self.bijectors = {
            'lam': tfb.Exp(),
            'tausq_group': tfb.Exp(),
            'W': tfb.Identity(),
            'b': tfb.Identity()}
        self.bijectors = [self.bijectors[k] for k in self.parameters]

    @tf.function
    def sample(self):
        param = self.joint.sample()
        param['W'] = tf.concat(
            [tf.reshape(tensor=param['W_shrinked'], shape=(1, self.no_units, 1)),
             param['W']], axis=2)
        del param['W_shrinked']
        return param

    # @tf.function
    def prior_log_prob(self, param):
        # notice the argument parsing of param!
        param = self.parse(param)
        return tf.reduce_sum(self.joint.log_prob(**param))

    def parse(self, param):
        """Since W_shrinked is incorporated in W for outer 'appearence' - i.e.
        all models, that employ HiddenGroupLasso layer, this method provides the
        means to split W in its components for log_prob"""
        param['W_shrinked'] = param['W'][:, :, self.no_shrinked]
        param['W'] = param['W'][:, :, self.no_shrinked:]

        return param

    # @tf.function
    def dense(self, X, **kwargs):
        return self.activation(
            tf.linalg.matmul(
                a=X,
                b=kwargs['W'][0],
                transpose_b=True) + \
                kwargs['b']

        )

        # deprec
        # + tf.reshape(kwargs['b'], (*kwargs['b'].shape, 1))

        # tf.concat(
        #     [kwargs['W'], tf.reshape(kwargs['W_shrinked'], (*kwargs['W_shrinked'].shape, 1))], axis=2),


if __name__ == '__main__':
    gl = HiddenGroupLasso(input_shape=5, no_units=10, activation='relu')
    gl.init = gl.sample()
    gl.prior_log_prob(gl.init)
    print('')