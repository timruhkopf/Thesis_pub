import torch
import torch.distributions as td
import torch.nn as nn
from functools import partial

from Pytorch.Models.BNN import BNN
from Pytorch.Models.GAM import GAM
import Pytorch.Layer as layer


class Shrinkage_BNN(nn.Module):
    # hard inheritance due to init & bnn & Shrinkage_BNN with nn.Module resolvement issue
    # furthermore, avoiding inheritance avoids BNN.layers to register to parameters
    vec_to_attrs = BNN.vec_to_attrs
    likelihood= BNN.likelihood
    log_prob = BNN.log_prob
    prior_log_prob = BNN.prior_log_prob
    __call__ = BNN.__call__

    # available shrinkage layers
    shrinkage = {
        'glasso': layer.Group_lasso,
        'gspike': layer.Group_SpikeNSlab,
        'ghorse': layer.Group_HorseShoe
    }

    def __init__(self, hunits=[2, 10, 1], activation=nn.ReLU(), final_activation=nn.Identity(),
                 shrinkage='glasso', **gam_param):
        """
        Shrinkage BNN is a regular BNN, but uses a shrinkage layer, which in the
        current implementation shrinks the first variable in the X vector,
        depending on whether or not it helps predictive performance.

        :param hunits: hidden units per layer
        :param activation: nn.ActivationFunction instance
        :param final_activation: nn.ActivationFunction instance
        :param shrinkage: 'glasso', 'gsplike', 'ghorse' each of which specifies the
        grouped shrinkage version of lasso, spike & slab & horseshoe respectively.
        See detailed doc in the respective layer. All of which assume the first variable
        to be shrunken; i.e. provide a prior log prob model on the first column of W
        :param gam_param:
        """
        super().__init__()
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation

        self.layers = nn.Sequential(self.shrinkage[shrinkage](hunits[0], hunits[1], True, activation),
            *[layer.Hidden(no_in, no_units, True, activation)
              for no_in, no_units in zip(self.hunits[1:-2], self.hunits[2:-1])],
            layer.Hidden(hunits[-2], hunits[-1], bias=False, activation=final_activation))

    def sample(self, seperate=True):
        """
        Sample the entire Prior Model
        :param seperate: True indicates whether or not, the true underlying model
        has a joint or disjoint effect (i.e. shrinkage or no shrinkage)
        :return: list of randomly sampled parameters
        """
        if seperate:
            pass
        else:
            pass


if __name__ == '__main__':
    sbnn = Shrinkage_BNN()

    # sampling with effect
    sbnn.sample(seperate=True)
    sbnn.sample(seperate=False)

    X_dist = td.Uniform(torch.tensor([-10., -10]), torch.tensor([10., 10]))
    X = X_dist.sample(torch.Size([100]))
    X.detach()
    X.requires_grad_()

    y = sbnn.likelihood(X).sample()
    y.detach()
    y.requires_grad_()
