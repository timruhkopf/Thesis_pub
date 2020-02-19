import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from functools import partial


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


    def sample_chain(self, log_dir, num_burnin_steps=int(1e3), num_results=int(10e3)):

        # (SAMPLING RELATED) ----------------------
        def trace_fn(current_state, kernel_results, summary_freq=10):
            """
            Trace function has a "current" look inside the state of the chain and
            the inner_kernels result and allows choosing which values to build
            traces of. Those values are returned in nested tupel "inner_kernel"
            parameters, (inner_kernel) = tfp.mcmc.sample_chain()

            This function allows TB integration during training
            based on https://github.com/tensorflow/probability/issues/356
            """
            #     step = kernel_results.step
            #     with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
            #         tf.print(current_state)
            return kernel_results.inner_results

        @tf.function
        def tfgraph_sample_chain(*args, **kwargs):
            """tf.function wrapped sample_chain. This became necessary, as TB 
            was introduced"""
            return tfp.mcmc.sample_chain(*args, **kwargs)

        chain, traced = tfgraph_sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=self.initial,
            kernel=self.adaptive_hmc,
            trace_fn=trace_fn)  # partial(trace_fn, summary_freq=20)

        # (SUMMARY STATISTICS) -----------------
        # seriously depending on trace_fn
        is_accepted = traced.inner_results.is_accepted
        rate_accepted = tf.reduce_mean(tf.cast(is_accepted, tf.float32), axis=0)
        chain_accepted = chain[tf.reduce_all(is_accepted, axis=1)]
        # FIXME: PECULARITY: not all parameters of a state need be rejected!

        sample_mean = tf.reduce_mean(chain_accepted, axis=0)
        sample_stddev = tf.math.reduce_std(chain_accepted, axis=0)
        tf.print('sample mean:\t', sample_mean,
                 '\nsample std: \t', sample_stddev,
                 '\nacceptance rate: ', rate_accepted)

        # (TB RELATED) ---------------------
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():

            for i, col in enumerate(tf.transpose(chain_accepted)):
                name = 'parameter' + str(i) + '_chain'
                namehist = 'parameter' + str(i) + '_hist'

                tf.summary.histogram(name=namehist, data=col, step=0)
                for step, proposal in enumerate(col):
                    tf.summary.scalar(name=name, data=proposal, step=step)



        self.chain = chain
        self.traced = traced

        # tf.print(chain.__len__())
        # tf.print(self.chain)


        return self.chain, traced

    def predict(self):
        pass

    # def eval_metrics(self):
    #     Metrics.__init__(self):
