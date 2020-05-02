import tensorflow as tf
import tensorflow_probability as tfp
from Python.Bayesian.Samplers.Samplers import Samplers
from Python.Util import setkwargs


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
                step_size=new_step_size),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio)
