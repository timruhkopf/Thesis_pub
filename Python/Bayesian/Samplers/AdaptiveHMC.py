import tensorflow as tf
import tensorflow_probability as tfp

# traceplot
import seaborn as sns
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors


class AdaptiveHMC:
    # TODO saving model with: tf.train.Checkpoint ???
    # TODO how to start from checkpoint: adaptations & overwrite self.initial

    """intended to hold the adaptive HMC sampler and to be inherited by a model class
    object, that has unnormalized_log_prob, bijectors and initial self attributes"""

    def __init__(self, initial, bijectors, log_prob,
                 num_burnin_steps=int(1e3), num_leapfrog_steps=3):
        assert (all([tensor.dtype == tf.float32 for tensor in initial]))
        self.initial = initial
        self.bijectors = bijectors
        self.log_prob = log_prob

        # FIXME: CHECK DOC OF THESE THREE SAMPLER OBJECTS & PAPERS PROVIDED IN DOC
        bijected_hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=1.),
            bijector=self.bijectors)

        self.adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=bijected_hmc,
            num_adaptation_steps=int(num_burnin_steps * 0.8))

    def sample_chain(self, logdir, num_burnin_steps=int(1e3), num_results=int(10e3)):
        # @tf.function
        def tfgraph_sample_chain(*args, **kwargs):
            """tf.function wrapped sample_chain. This became necessary, as TB 
            was introduced"""
            return tfp.mcmc.sample_chain(*args, **kwargs)

        self.num_results = num_results
        self.logdir = logdir

        # TB wrapper for Runtime (trace_fn) TB writing!
        # Tracing graph: https://github.com/tensorflow/tensorboard/issues/1961
        # profiling in TB https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
        # with tf.python.eager.profiler.Profiler('logdir_path'): # not working as tf.python does not exist

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

        self.chains = chain
        self.traced = traced

        self._sample_chain_staticsummary_toTB()

        return self.chains, traced

    #@tf.function
    def _sample_chain_trace_fn(self, current_state, kernel_results, summary_freq=100):
        """
        Trace function has a "current" look inside the state of the chain and
        the inner_kernels result and allows choosing which values to build
        traces of. Those values are returned in nested tupel "inner_kernel"
        parameters, (inner_kernel) = tfp.mcmc.sample_chain()

        This function allows TB integration during training
        based on https://github.com/tensorflow/probability/issues/356
        """
        step = kernel_results.step
        if step % summary_freq == 0:
            print('\n\nStep {}, Logposterior:{}\n{}'.format(
                step, self.log_prob(*current_state), current_state))  # '\n'.join(current_state)


        with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
            name = 'experiment writing during execution'

            # singlevalued CURRENT_STATE
            # if isinstance(current_state, tf.Tensor):
            #     tf.summary.scalar(
            #         name=name, data=current_state, step=tf.cast(step, tf.int64))

            # multivalued CURRENT_STATE
            if isinstance(current_state, list):
                for variable in current_state:
                    tf.summary.histogram(
                        name=name, data=variable, step=tf.cast(step, tf.int64))

        return kernel_results.inner_results

    # @tf.function
    def _sample_chain_staticsummary_toTB(self):
        # CONSIDER: Move to Metrics class and inherit Metrics class
        # (TB STATIC STATISTICS RELATED) ---------------------
        # seriously depending on trace_fn
        is_accepted = self.traced.inner_results.is_accepted

        rate_accepted = tf.reduce_mean(tf.cast(is_accepted, tf.float32), axis=0)

        # FIXME!!!!! look at shapes of chain - depending on chain of single or multiple
        #  parameter tensors (list of tensors)
        if len(is_accepted.shape) == 1:
            # chain_accepted = self.chain[is_accepted]
            if isinstance(self.chains, list):
                chain_accepted = [chain[is_accepted] for chain in self.chains]
        else:
            # FIXME: PECULARITY: not all parameters of a state (row) need be rejected!
            chain_accepted = self.chains[tf.reduce_all(is_accepted, axis=1)]

        # TODO Posterior Mode
        # sample_mean = tf.reduce_mean(chain_accepted, axis=0)
        # sample_stddev = tf.math.reduce_std(chain_accepted, axis=0)
        # tf.print('sample mean:\t', sample_mean,
        #          '\nsample std: \t', sample_stddev,
        #          '\nacceptance rate: ', rate_accepted)

        # FIXME: change sub log_dir for multiple runs!
        # writer = tf.summary.create_file_writer(self.logdir)
        # with writer.as_default():
        #     # (chain autocorrelation)
        #     # TODO tfp.stats.auto_correlation !!!!!!!!!!!!
        #
        #     # (Chain trace plot)
        #     for i, col in enumerate(tf.transpose(chain_accepted)):
        #         name = 'parameter' + str(i) + '_chain'
        #         namehist = 'parameter' + str(i) + '_hist'
        #
        #         tf.summary.histogram(name=namehist, data=col, step=0)
        #         for step, proposal in enumerate(col):
        #             tf.summary.scalar(name=name, data=proposal, step=step)

    def predict_mode(self, logpost):
        """get parameter set of the max log_posterior value
         estimating the log_posterior probability of all samples
        :param logpost: the models log posterior function
        :return tuple: parameter set of max logposterior."""
        # TODO (1) point prediction (posterior Mode? max log-prob param-set)
        paramsets = [s for s in zip(*self.chains)]
        post = tf.stack(list(
            map(lambda x: logpost(*x), paramsets)), axis=0)
        return paramsets[tf.argmax(post, axis=0).numpy()]

    # def predict_mean(self):
    # carefull, model is not defined here!
    #     meanPost = [tf.reduce_mean(chain, axis=0) for chain in self.chain]
    #     Ws, bs = bnn.argparser(meanPost)
    #     y_map = bnn.forward(X, Ws, bs)

    def predict_mean(self):
        if isinstance(self.chains, list):
            return [tf.reduce_mean(chain, axis=0) for chain in self.chains]
        else:
            tf.reduce_mean(self.chains, axis=0)

    def predict_posterior(self):
        # TODO (2) posterior predictive distribution
        #  (log-prob weighted pred for parameter sets of chain_accepted??)
        # Consider posterior predictive distribution estimation, writing in TF
        # file:///home/tim/PycharmProjects/Thesis/Literature/Bayesian/(Krueger, Lerch)
        # Predictive Inference Based on Markov Chain Monte.pdf
        pass

    # def plot_traces(self, samples):
    #     """
    #     # EXPERIMENTAL Method. not yet adjusted for chain format from adHMC
    #     original source code from: for single Tensor! e.g. beta nxp:
    #     plot_traces(self,'beta', samples=self.chain, num_chain=p
    #     https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Multilevel_Modeling_Primer.ipynb#scrollTo=v0hZwZfQyjsR
    #     """
    #     # FIXME: do i need to change the values of
    #
    #     if isinstance(self.chains, tf.Tensor):
    #         samples = self.chains.numpy()  # convert to numpy array
    #
    #     # DEPREC: plotting the traces - problems:
    #     #  parameters in matrix form!
    #     #  possibly thousands of parameters
    #     # if isinstance(self.chains, list):
    #     #     dim = int(tf.sqrt(len(self.chains)))
    #     #     fig, axes = plt.subplots(dim, dim, sharex='col', sharey='col')
    #     #     x = range(self.num_results)
    #     #
    #     #     for chain, ax in zip(self.chains, axes):
    #     #         sns.lineplot(x, y=tf.chain)
    #     #
    #     #     plt.show()
    #

