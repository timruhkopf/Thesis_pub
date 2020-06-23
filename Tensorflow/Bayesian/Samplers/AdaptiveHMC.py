import tensorflow as tf
import tensorflow_probability as tfp
from Tensorflow.Bayesian.Samplers.Samplers import Samplers
from Tensorflow.Util import setkwargs


class AdaptiveHMC(Samplers):
    @setkwargs
    def __init__(self, initial, bijectors, log_prob,
                 num_adaptation_steps=int(1e3), num_leapfrog_steps=3):
        assert (all([tensor.dtype == tf.float32 for tensor in initial]))

        # FIXME: CHECK DOC OF THESE THREE SAMPLER OBJECTS & PAPERS PROVIDED IN DOC
        bijected_hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=1.),
            bijector=bijectors)

        self.kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=bijected_hmc,
            num_adaptation_steps=num_adaptation_steps)

if __name__ == '__main__':
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

    adHMC = AdaptiveHMC(initial=list(param.values()),  # CAREFULL MUST BE FLOAT!
                        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Exp()],
                        log_prob=reg.unnormalized_log_prob,
                        num_leapfrog_steps=3)

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(1 * 10e0),
        num_results=int(10e1),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults',
        parallel_iterations=10)



    print()