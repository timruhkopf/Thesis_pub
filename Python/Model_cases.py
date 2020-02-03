import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from Python.bspline import get_design, diff_mat
from Python.Effect_cases import *


class AdaptiveHMC():
    """intended to hold the adaptive HMC sampler and to be inherited by a model class
    object, that has unnormalized_log_prob, bijectors and initial self attributes"""

    def __init__(self, initial, bijectors, num_burnin_steps=int(1e3), num_leapfrog_steps=3):
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
            current_state=self.initial,
            kernel=self.adaptive_hmc,
            trace_fn=None,
            parallel_iterations=1
        )
        self.parameters = parameters
        self.is_accepted = is_accepted
        return self.parameters, self.is_accepted


class Xbeta(AdaptiveHMC, Effects1D):

class Xbeta(adaptiveHMC, Effects1D):

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
    xbeta = Xbeta(xgrid=(0,10,0.5))
    #xbeta.sample_chain()




