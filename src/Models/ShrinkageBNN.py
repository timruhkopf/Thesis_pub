import torch.nn as nn

from src.Models.BNN import BNN
from src.Layer import *


class ShrinkageBNN(BNN):
    # available shrinkage layers
    shrinkage_type = {
        'glasso': GroupLasso,
        'ghorse': GroupHorseShoe,
    }

    def __init__(self, hunits=(2, 10, 1), activation=nn.ReLU(), final_activation=nn.Identity(),
                 shrinkage='ghorse', no_shrink=1, separated=False):
        """
        Shrinkage BNN is a regular BNN, but uses a shrinkage layer, which in the
        current implementation shrinks the first variable in the X vector,
        depending on whether or not it helps predictive performance.


        :param activation: nn.ActivationFunction instance
        :param final_activation: nn.ActivationFunction instance
        :param shrinkage: 'glasso', 'ghorse' each of which specifies the
        grouped shrinkage version of lasso & horseshoe respectively.
        See detailed doc in the respective layer. All of which assume the first
        to be shrunken; i.e. provide a prior log prob model on the first column of W

        """
        if len(hunits) < 3:
            raise ValueError('In its current form, shrinkage BNN works only for '
                             'multiple layers: increase no. layers via hunits')
        self.shrinkage = shrinkage
        self.separated = separated
        self.no_shrink = no_shrink

        BNN.__init__(self, hunits, activation, final_activation)

    def define_model(self):
        # Defining the layers depending on the mode.
        S = self.shrinkage_type[self.shrinkage]

        self.layers = nn.Sequential(
            S(self.hunits[0], self.hunits[1], True, self.activation, self.no_shrink),
            *[Hidden(no_in, no_units, True, self.activation)
              for no_in, no_units in zip(self.hunits[1:-2], self.hunits[2:-1])],
            Hidden(self.hunits[-2], self.hunits[-1], bias=False, activation=self.final_activation))
