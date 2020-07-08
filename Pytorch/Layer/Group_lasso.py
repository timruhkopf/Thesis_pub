import torch
import torch.nn as nn

from Pytorch.Layer.Hidden import Hidden


class Group_lasso(Hidden):
    # inherritance allows to keep dense method & "activ" activation functions in one place!
    # IDEALLY MAKE IT AUTOGRAD READY IF NECESSARY BY SAMPLER - INHERRIT nn.Module?
    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        self.bias = False  # to meet the Hidden default standard!
