import abc
from copy import deepcopy
import torch.nn as nn


class Model(nn.Module):
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
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset_parameters(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def log_prob(self):
        pass

    @abc.abstractmethod
    @staticmethod
    def check_chain(chain):
        pass
