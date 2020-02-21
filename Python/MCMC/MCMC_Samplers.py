import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class AdaptiveHMC:
    """intended to hold the adaptive HMC sampler and to be inherited by a model class
    object, that has unnormalized_log_prob, bijectors and initial self attributes"""

    def __init__(self, initial, bijectors, log_prob,
                 num_burnin_steps=int(1e3), num_leapfrog_steps=3):
        self.initial = initial

        # FIXME: CHECK DOC OF THESE THREE SAMPLER OBJECTS & PAPERS PROVIDED IN DOC
        bijected_hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=1.),
            bijector=bijectors)

        self.adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            bijected_hmc,
            num_adaptation_steps=int(num_burnin_steps * 0.8))

    def sample_chain(self, logdir, num_burnin_steps=int(1e3), num_results=int(10e3)):
        @tf.function
        def tfgraph_sample_chain(*args, **kwargs):
            """tf.function wrapped sample_chain. This became necessary, as TB 
            was introduced"""
            return tfp.mcmc.sample_chain(*args, **kwargs)

        self.logdir = logdir

        # TB wrapper for Runtime (trace_fn) TB writing!
        # Tracing graph: https://github.com/tensorflow/tensorboard/issues/1961
        writer = tf.summary.create_file_writer(self.logdir)
        with writer.as_default():
            tf.summary.trace_on(graph=True)
            chain, traced = tfgraph_sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=self.initial,
                kernel=self.adaptive_hmc,
                trace_fn=self._sample_chain_trace_fn)  # functools.partial(trace_fn, summary_freq=20)

            tf.summary.trace_export(name='graphingit', step=0)
            tf.summary.trace_off()

        self.chain = chain
        self.traced = traced

        self._sample_chain_write_staticsummary_toTB()

        # (TB HYPERPARAMETERS) ---------------------
        # https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams

        # tf.print(chain.__len__())
        # tf.print(self.chain)

        return self.chain, traced

    def _sample_chain_trace_fn(self, current_state, kernel_results, summary_freq=10):
        """
        Trace function has a "current" look inside the state of the chain and
        the inner_kernels result and allows choosing which values to build
        traces of. Those values are returned in nested tupel "inner_kernel"
        parameters, (inner_kernel) = tfp.mcmc.sample_chain()

        This function allows TB integration during training
        based on https://github.com/tensorflow/probability/issues/356
        """
        step = kernel_results.step
        with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
            name = 'experiment writing during execution'
            # tf.summary.scalar(name=name, data=current_state[0], step=tf.cast(step, tf.int64))
            tf.summary.histogram(name=name, data=current_state, step=tf.cast(step, tf.int64))

        return kernel_results.inner_results

    def _sample_chain_staticsummary_toTB(self):
        # CONSIDER: Move to Metrics class and inherrit Metrics class
        # (TB STATIC STATISTICS RELATED) ---------------------
        # seriously depending on trace_fn
        is_accepted = self.traced.inner_results.is_accepted

        rate_accepted = tf.reduce_mean(tf.cast(is_accepted, tf.float32), axis=0)
        chain_accepted = self.chain[tf.reduce_all(is_accepted, axis=1)]
        # FIXME: PECULARITY: not all parameters of a state need be rejected!

        # TODO Posterior Mode
        sample_mean = tf.reduce_mean(chain_accepted, axis=0)
        sample_stddev = tf.math.reduce_std(chain_accepted, axis=0)
        tf.print('sample mean:\t', sample_mean,
                 '\nsample std: \t', sample_stddev,
                 '\nacceptance rate: ', rate_accepted)

        # FIXME: change sub log_dir for multiple runs!
        writer = tf.summary.create_file_writer(self.logdir)
        with writer.as_default():
            # TODO tfp.stats.auto_correlation !!!!!!!!!!!!!!!!!!!!!!!!!
            for i, col in enumerate(tf.transpose(chain_accepted)):
                name = 'parameter' + str(i) + '_chain'
                namehist = 'parameter' + str(i) + '_hist'

                tf.summary.histogram(name=namehist, data=col, step=0)
                for step, proposal in enumerate(col):
                    tf.summary.scalar(name=name, data=proposal, step=step)

        # TODO saving model with: tf.train.Checkpoint ???
        # TODO how to start from checkpoint: adaptations & overwrite self.initial

    def predict_mode(self):
        # TODO (1) point prediction (posterior Mode? max log-prob param-set)
        pass

    def predict_posterior(self):
        # TODO (2) posterior predictive distribution
        #  (log-prob weighted pred for parameter sets of chain_accepted??)
        # Consider posterior predictive distribution estimation, writing in TF
        # file:///home/tim/PycharmProjects/Thesis/Literature/Bayesian/(Krueger, Lerch)
        # Predictive Inference Based on Markov Chain Monte.pdf
        pass

    # def eval_metrics(self):
    #     Metrics.__init__(self):
