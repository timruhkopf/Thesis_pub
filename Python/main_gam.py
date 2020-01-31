import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from Python.bspline import get_design, diff_mat
from Python.Effect_sub import *

class HMC():
    def __init__(self, num_burnin_steps=int(1e3), num_leapfrog_steps=3):

        bijected_hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.unnormalized_log_prob,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=1.),
            bijector=self.bijectors)

        self.adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            bijected_hmc,
            num_adaptation_steps=int(num_burnin_steps * 0.8))

    def sample_chain(self, num_burnin_steps=int(1e3), num_results=int(10e3)):
        parameters, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=self.initial, # initial_chain_state,
            kernel=self.adaptive_hmc,
            trace_fn=None,
            parallel_iterations=1
        )
        self.parameters = parameters
        self.is_accepted = is_accepted
        return self.parameters, self.is_accepted

    def store(self):
        pass

    def plot_residuals(self):
        pass

class Xbeta(HMC, Effects1D):

    def __init__(self, ):
        # data generation
        self.X = None
        self.Bspline = Bspline_K(self.xgrid)
        self.mu = self.Bspline.spl(self.X)
        self.y

        # bijectors
        self.initial = None
        self.bijectors = None

        self.unnormalized_log_prob =self.unnorm_log_prob() # FIXME

    def unnorm_log_prob(self, beta):
        def xbeta_log_prob(beta, X=self.X, y=self.y):

            return
        return xbeta_log_prob(beta)

if __name__ == '__main__':
    xbeta = Xbeta()
    xbeta.sample_chain()