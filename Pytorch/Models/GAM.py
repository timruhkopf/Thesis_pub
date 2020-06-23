import torch

from Pytorch.Layer.Hidden import Hidden
from Tensorflow.Effects.bspline import get_design, diff_mat1D


class GAM(Hidden):
    # inheritance brings "parameters", "bijectors", "activation"
    def __init__(self, no_basis, order=1):
        """
        RandomWalk Prior Model on Gamma
        :param no_basis: number of basis from the de boor basis expansion to expect (=no_columns of X)
        :param order: difference order to create the Precision (/Penalty) matrix K
        """
        super(GAM, self).__init__(no_in=no_basis, no_units=1, bias=False, activation='identity')
        self.order = order
        self.no_basis = no_basis
        self.K = diff_mat1D(no_basis, order)[1]

    def sample(self):
        """sample the prior model"""
        pass

    def log_prob(self):
        pass


if __name__ == '__main__':
    pass
    # dense example

    # sampling Example

    # log prob example

    # plot 1D
