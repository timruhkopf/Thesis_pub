from Tensorflow.Bayesian.Samplers.Samplers import Samplers
from Tensorflow.Util import setkwargs
import tensorflow as tf
import tensorflow_probability as tfp

class MALA(Samplers):
    @setkwargs
    def __init__(self, initial, bijectors, log_prob,):
        """https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/MetropolisAdjustedLangevinAlgorithm"""
        self.kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
                target_log_prob_fn=log_prob,
                step_size=0.75,
                seed=42),

        # Notice on sampling mala
        # samples = tfp.mcmc.sample_chain(
        #     num_results=1000,
        #     current_state=dtype(1),
        #     kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        #         target_log_prob_fn=target_log_prob,
        #         step_size=0.75,
        #         seed=42),
        #     num_burnin_steps=500,
        #     trace_fn=None,
        #     parallel_iterations=1)

        # Notice on HMC's bijecting! is this required for mala?
        # tfp.mcmc.TransformedTransitionKernel(
        #             inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        #                 target_log_prob_fn=log_prob,
        #                 num_leapfrog_steps=num_leapfrog_steps,
        #                 step_size=1.),
        #             bijector=bijectors)


if __name__ == '__main__':
    pass