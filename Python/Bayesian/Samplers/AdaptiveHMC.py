import tensorflow as tf
import tensorflow_probability as tfp
from Python.Bayesian.Samplers.Samplers import Samplers


class AdaptiveHMC(Samplers):
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

        self.kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=bijected_hmc,
            num_adaptation_steps=int(num_burnin_steps * 0.8))
