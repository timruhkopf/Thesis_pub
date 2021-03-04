import abc
from copy import deepcopy
import torch.nn as nn


class Model(nn.Module):
    """optional (but useful) methods are in comments"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        nn.Module.__init__(self)
        self.true_model = deepcopy(self.state_dict())

    @abc.abstractmethod
    def define_model(self):
        pass

    @abc.abstractmethod
    def update_distributions(self):
        pass

    @abc.abstractmethod
    def forward(self, X, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset_parameters(self, *args, **kwargs):
        self.init_model = deepcopy(self.state_dict())
        pass

    # @abc.abstractmethod
    # def prior_log_prob(self):
    #     pass

    # @abc.abstractmethod
    # def likelihood(self, X, *args, **kwargs):
    #     pass

    @abc.abstractmethod
    def log_prob(self):
        # while not strictly required, a prior_log_prob & likelihood method
        # may help to formulate the model
        pass

    @abc.abstractmethod
    @staticmethod
    def check_chain(chain):
        pass

    # @abc.abstractmethod
    # def sample_model(self):
    #     pass
