import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class AdaptiveHMC:
    # FIXME: FAIL OF SAMPLING ALWAYS SAME  VALUE MUST LIE IN HERE NOT XBETA!!
    """intended to hold the adaptive HMC sampler and to be inherited by a model class
    object, that has unnormalized_log_prob, bijectors and initial self attributes"""

    def __init__(self, initial, bijectors, log_prob, num_burnin_steps=int(1e3), num_leapfrog_steps=3):
        self.initial = initial
        bijected_hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=1.),
            bijector=bijectors)

        self.adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            bijected_hmc,
            num_adaptation_steps=int(num_burnin_steps * 0.8))

    @tf.function
    def sample_chain(self, num_burnin_steps=int(1e3), num_results=int(10e3)):
        samples = tfp.mcmc.sample_chain(  # samples, *tup
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=self.initial,
            kernel=self.adaptive_hmc)
        # trace_fn=lambda _, pkr: pkr.inner_results.is_accepted ) # increases number of outputs

        # (TRACE FUNCTION RELATED) ---------------------------------------------
        # Trace function has a "current" look inside the state of the chain and
        # the inner_kernels result and allows choosing which values to build
        # traces of. Those values are returned in nested tupel "inner_kernel"
        # parameters, (inner_kernel) = tfp.mcmc.sample_chain()

        # TRACE WITH TENSORBOARD:
        # https://github.com/tensorflow/probability/issues/356
        # def trace_fn(weights, results):
        #    with tf.compat.v2.summary.record_if(tf.equal(results.step % 100, 0)):
        #      tf.compat.v2.summary.histogram(weights, step=results.step)
        #    return ()

        self.samples = samples[0]

        # tf.print(samples.__len__())
        # tf.print(samples[1])

        # seriously depending on trace_fn
        is_accepted = samples[1].inner_results.inner_results.is_accepted
        rate_accepted = tf.reduce_mean(tf.cast(is_accepted, tf.float32), axis=0)

        sample_mean = tf.reduce_mean(samples[0], axis=0)
        sample_stddev = tf.math.reduce_std(samples[0], axis=0)
        tf.print('sample mean:\t', sample_mean,
                 '\nsample std: \t', sample_stddev,
                 '\nacceptance rate: ', rate_accepted)
        return samples

    def predict(self):
        pass

    # def eval_metrics(self):
    #     Metrics.__init__(self):
