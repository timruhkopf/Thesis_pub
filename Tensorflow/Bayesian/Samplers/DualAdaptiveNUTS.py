from Tensorflow.Bayesian.Samplers.Samplers import Samplers
from Tensorflow.Util import setkwargs

# CAREFULL NOT YET DEBUGGED!
class DualAdaptiveNUTS(Samplers):
    @setkwargs
    def __init__(self, initial, bijectors, log_prob, num_adaptation_steps=500):
        assert (all([tensor.dtype == tf.float32 for tensor in initial]))

        bijected = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=log_prob,
                step_size=1e-3),
            bijector=bijectors)

        self.kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=bijected,
            num_adaptation_steps=num_adaptation_steps,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                step_size=new_step_size))
        # step_size_getter_fn=lambda pkr: pkr.step_size,
        # log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio)


if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions

    from Tensorflow.Bayesian.Models.Base.Regression import Regression

    tfd = tfp.distributions
    tfb = tfp.bijectors

    # (0) generate data ---------------------------------------
    xgrid = (0, 10)
    n = 100
    X = tf.stack(
        values=[tf.ones((n,)),
                tfd.Uniform(xgrid[0], xgrid[1]).sample((n,))],
        axis=1)

    # no units must correspond to X.shape[1]
    p = X.shape[1]
    reg = Regression(input_shape=p, no_units=1, activation='identity')
    param = {'tau': tf.constant(0.5), 'W': tf.constant([-0.5, 2]), 'sigma': tf.constant(0.25)}
    y = reg.likelihood_model(X, W=param['W'], sigma=param['sigma']).sample()

    # fixme: reshape y !
    # y = tf.reshape(y, (100, 1))
    reg.unnormalized_log_prob = reg._closure_log_prob(X, y)

    num_burnin_steps = 1000
    num_results = 2000
    num_chains = 64
    step_size = 0.1
    # Or, if you want per-chain step size:
    # step_size = tf.fill([num_chains], step_size)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=reg.unnormalized_log_prob,
        num_leapfrog_steps=2,
        step_size=step_size)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))

    def tracer(a, pkr):
        return [pkr.inner_results.accepted_results.step_size, pkr.inner_results.log_accept_ratio]

    # The chain will be stepped for num_results + num_burnin_steps, adapting for
    # the first num_adaptation_steps.
    samples, [step_size, log_accept_ratio] = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=list(param.values()),
        kernel=kernel,
        trace_fn=tracer)


    # ~0.75
    p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
        log_accept_ratio, 0.)))










    from Tensorflow.Bayesian.Models.Base.Regression import Regression

    tfd = tfp.distributions
    tfb = tfp.bijectors

    # (0) generate data ---------------------------------------
    xgrid = (0, 10)
    n = 100
    X = tf.stack(
        values=[tf.ones((n,)),
                tfd.Uniform(xgrid[0], xgrid[1]).sample((n,))],
        axis=1)

    # no units must correspond to X.shape[1]
    p = X.shape[1]
    reg = Regression(input_shape=p, no_units=1, activation='identity')
    param = {'tau': tf.constant(0.5), 'W': tf.constant([-0.5, 2]), 'sigma': tf.constant(0.25)}
    y = reg.likelihood_model(X, W=param['W'], sigma=param['sigma']).sample()

    # fixme: reshape y !
    # y = tf.reshape(y, (100, 1))
    reg.unnormalized_log_prob = reg._closure_log_prob(X, y)

    # DualAdNUTS = DualAdaptiveNUTS(
    #     initial=list(param.values()),  # CAREFULL MUST BE FLOAT!
    #     bijectors=[tfb.Exp(), tfb.Identity(), tfb.Exp()],
    #     log_prob=reg.unnormalized_log_prob)
    # DualAdNUTS.sample_chain(
    #     num_burnin_steps=int(1 * 10e0),
    #     num_results=int(10e1),
    #     logdir='/home/tim/PycharmProjects/Thesis/TFResults',
    #     parallel_iterations=1)

    # bijected = tfp.mcmc.TransformedTransitionKernel(
    #     inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
    #         target_log_prob_fn=reg.unnormalized_log_prob,
    #         num_leapfrog_steps=2,
    #         step_size=0.1),
    #     bijector=[tfb.Exp(), tfb.Identity(), tfb.Exp()])
    # kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #     inner_kernel=bijected, num_adaptation_steps=400)
    #
    # # The chain will be stepped for num_results + num_burnin_steps, adapting for
    # # the first num_adaptation_steps.
    # samples, [step_size, log_accept_ratio] = tfp.mcmc.sample_chain(
    #     num_results=1000,
    #     num_burnin_steps=500,
    #     current_state=list(param.values()),
    #     kernel=kernel,
    #     # trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
    #     #                          pkr.inner_results.log_accept_ratio])
    # )
    print()
