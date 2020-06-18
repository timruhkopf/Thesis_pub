from Tensorflow.Bayesian.Samplers import Samplers
import tensorflow as tf
import tensorflow_probability as tfp

class pSGLD(Samplers):
    def __init__(self):
        """wrapper for pSGLD to inherit methods from Sampler
        https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/StochasticGradientLangevinDynamics"""
        tfp.optimizer.sgld
        pass

if __name__ == '__main__':
    pass