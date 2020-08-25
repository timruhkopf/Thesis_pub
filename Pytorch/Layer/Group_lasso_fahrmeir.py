import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Group_lasso import Group_lasso


class Group_lasso_farhmeir(Group_lasso):
    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """Same as Group lasso, but with slightly different distributional assumptions
        on the parameters tau and lamb. might help """
        super().__init__(no_in, no_out)

        # overwrite default distrib. assumption
        self.dist['lamb'] = td.Gamma(0.01, 0.01)
        self.dist['tau'] = td.Exponential(0.5 * self.lamb ** 2)

        self.true_model = None

    def update_distributions(self):
        """due to the hierarchical stucture, the distributions parameters must be updated
        Note, that this function is intended to be called immediately after vec_to_attr
        in order to get thr correct log prob. This function does not """
        self.dist['tau'].rate = 0.5 * self.lamb ** 2
        self.dist['W_shrinked'].scale = self.tau

    def reset_parameters(self, seperated=False):
        """sampling method to instantiate the parameters"""

        self.lamb = self.dist['lamb'].sample()
        self.dist['tau'].rate = 0.5 * self.lamb ** 2

        if seperated:
            self.tau = torch.tensor(0.001)
        else:
            self.tau = self.dist['tau'].sample()

        self.dist['W_shrinked'].scale = self.tau
        self.W.data = torch.cat(
            [self.dist['W_shrinked'].sample().view(1, self.no_out),  # ASSUMING MERELY ONE VARIABLE TO BE SHRUNKEN
             self.dist['W'].sample().view(self.no_in - 1, self.no_out)],
            dim=0)

        self.b = self.dist['b'].sample()

        # setting the nn.Parameters's starting value
        self.lamb_.data = self.lamb
        self.tau_.data = self.tau
        self.W_.data = self.W
        self.b_.data = self.b


if __name__ == '__main__':
    pass
