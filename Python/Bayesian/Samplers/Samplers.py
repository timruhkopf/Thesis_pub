import tensorflow as tf
import tensorflow_probability as tfp

# traceplot
# import seaborn as sns
# import matplotlib.pyplot as plt

class Samplers:
    """a metaclass for samplers, to provide basic functionality and a consistent
    interface."""

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
                kernel=self.kernel,
                trace_fn=self._sample_chain_trace_fn  # functools.partial(trace_fn, summary_freq=20)
            # TODO :previous_kernel_results
            )
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

        with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
            print('\n\nStep {}, Logposterior:{}\n{}'.format(
                step, self.log_prob(*current_state), current_state))  # '\n'.join(current_state)

            # singlevalued CURRENT_STATE
            # if isinstance(current_state, tf.Tensor):
            #     tf.summary.scalar(
            #         name=name, data=current_state, step=tf.cast(step, tf.int64))

            # multivalued CURRENT_STATE
            if isinstance(current_state, list):
                for variable in current_state:
                    # TODO: name must be variable dependent
                    name = 'experiment writing during execution'
                    tf.summary.histogram(
                        name=name, data=variable, step=tf.cast(step, tf.int64))

        return kernel_results.inner_results

    # CONSIDER:
    # def trace_fn(current_state, kernel_results, summary_freq=10, callbacks=[]):
    #     """SOURCE: https://janosh.io/blog/hmc-bnn
    #     Can be passed to the HMC kernel to obtain a trace of intermediate
    #     kernel results and histograms of the network parameters in Tensorboard.
    #     """
    #     step = kernel_results.step
    #     with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
    #         for idx, tensor in enumerate(current_state):
    #             count = idx // 2 + 1
    #             FIXME: assumption on input format: enumerate (w, b, w, b, ..)
    #             name = ("w" if idx % 2 == 0 else "b") + str(count)
    #             tf.summary.histogram(name, tensor, step=step)
    #         return kernel_results, [cb(*current_state) for cb in callbacks]

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

    # CONSIDER: PREDICTC FROM CHAIN ALEATORIC EPISTEMIC
    # def predict_from_chain(chain, X_test, uncertainty="aleatoric+epistemic"):
    # """Source: https://janosh.io/blog/hmc-bnn
    # Takes a Markov chain of NN configurations and does the actual
    # prediction on a test set X_test including aleatoric and optionally
    # epistemic uncertainty estimation.
    # """
    # err = f"unrecognized uncertainty type: {uncertainty}"
    # assert uncertainty in ["aleatoric", "aleatoric+epistemic"], err
    #
    # if uncertainty == "aleatoric":
    #     post_params = [tf.reduce_mean(t, axis=0) for t in chain]
    #     post_model = build_net(post_params)
    #     y_pred, y_var = post_model(X_test, training=False)
    #
    #     return y_pred.numpy(), y_var.numpy()
    #
    # if uncertainty == "aleatoric+epistemic":
    #     restructured_chain = [
    #         [tensor[i] for tensor in chain] for i in range(len(chain[0]))
    #     ]
    #
    #     def predict(params):
    #         post_model = build_net(params)
    #         y_pred, y_var = post_model(X_test, training=False)
    #         return y_pred, y_var
    #
    #     preds = [predict(chunks(params, 2)) for params in restructured_chain]
    #     y_pred_mc_samples, y_var_mc_samples = tf.unstack(preds, axis=1,)
    #     y_pred, y_var_epist = tf.nn.moments(y_pred_mc_samples, axes=0)
    #     y_var_aleat = tf.reduce_mean(y_var_mc_samples, axis=0)
    #     y_var_tot = y_var_epist + y_var_aleat
    #     return y_pred, y_var_tot

    # CONSIDER RESUME CHAIN:
    # SOURCE: Source: https://janosh.io/blog/hmc-bnn
    # if resume:
    #     prev_chain, prev_trace, prev_kernel_results = resume
    #     step = len(prev_chain)
    #     current_state = tf.nest.map_structure(lambda chain: chain[-1], prev_chain)
    # else:
    #     prev_kernel_results = adaptive_kernel.bootstrap_results(current_state)
    #     step = 0
    #
    # tf.summary.trace_on(graph=True, profiler=False)
    #
    # chain, trace, final_kernel_results = sample_chain(
    #     kernel=adaptive_kernel,
    #     current_state=current_state,
    #     previous_kernel_results=prev_kernel_results,
    #     num_results=num_burnin_steps + num_results,
    #     trace_fn=ft.partial(trace_fn, summary_freq=20),
    #     return_final_kernel_results=True,
    #     **kwargs,
    # )
    # if resume:
    #         chain = nest_concat(prev_chain, chain)
    #         trace = nest_concat(prev_trace, trace)
    # burnin, samples = zip(*[(t[:-num_results], t[-num_results:]) for t in chain])
    #
    # def nest_concat(*args, axis=0):
    #     """Utility function for concatenating a new Markov chain or trace with
    #     older ones when resuming a previous calculation.
    #     """
    #     return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=axis), *args)

    # CONSIDER MAP estimation by max likelihood optimization:
    # SOURCE: Source: https://janosh.io/blog/hmc-bnn
    # import tensorflow as tf
    #
    #
    # def get_map_trace(target_log_prob_fn, state, n_iter=1000, save_every=10, callbacks=[]):
    #     optimizer = tf.optimizers.Adam()
    #
    #     @tf.function
    #     def minimize():
    #         optimizer.minimize(lambda: -target_log_prob_fn(*state), state)
    #
    #     state_trace, cb_trace = [], [[] for _ in callbacks]
    #     for i in range(n_iter):
    #         if i % save_every == 0:
    #             state_trace.append(state)
    #             for trace, cb in zip(cb_trace, callbacks):
    #                 trace.append(cb(state).numpy())
    #         minimize()
    #
    #     return state_trace, cb_trace
    #
    #
    # def get_best_map_state(map_trace, map_log_probs):
    #     # map_log_probs[0/1]: train/test log probability
    #     test_set_max_log_prob_idx = np.argmax(map_log_probs[1])
    #     # Return MAP params that achieved highest test set likelihood.
    #     return map_trace[test_set_max_log_prob_idx]

    # CONSIDER: Train Test split:
    # log_prob_tracers = (
    #     tracer_factory(X_train, y_train),
    #     tracer_factory(X_test, y_test),
    # )

    # CONSIDER effective sampe size(s)
    # def ess(chains, **kwargs):
    #     """Estimate effective sample size of Markov chain(s).
    #     Arguments:
    #         chains {Tensor or list of Tensors}: If list, first
    #             dimension should index identically distributed states.
    #     """
    #     return tfp.mcmc.effective_sample_size(chains, **kwargs)

    # CONSIDER chain convergence measure:
    # def r_hat(tensors):
    #     """https://tensorflow.org/probability/api_docs/python/tfp/mcmc/potential_scale_reduction
    #     """
    #     return [tfp.mcmc.diagnostic.potential_scale_reduction(t) for t in tensors]

