import torch
import torch.distributions as td

from copy import deepcopy

from src.Layer.Hierarchical_GroupLasso import GroupLasso
from src.Util.Util_Distribution import LogTransform


class GroupHorseShoe(GroupLasso):
    def __init__(self, *args, **kwargs):
        """

        Implements the following distributional assumptions:

            (0) shrunken variables
            W_shrunk ~ N(0, lamb * tau**2)
            tau ~ C+(0,1)  Half-Cauchy
            lamb ~ C+(0,1)

            (1) unshrunken variables
            W ~ N(0, 1)


        W is partitioned according to the number of shrinked variables.
        By convention, the first no_shrink-variables are considered shrinked variables.
        The groups of weights (W's rows) that are associated with a shrunken variable
        share a variance parameter tau.
        Across the variables, a second variance parameter Lamb is introduced, that
        corresponds to the sparsity across variables.

        tau & lamb are both bijected parameters.

        :param:
        Same arguments as Group_lasso & Hidden.
        """
        GroupLasso.__init__(self, *args, **kwargs)

        # update distributional assumption
        self.lamb.dist = td.HalfCauchy(torch.tensor([1.]))
        self.lamb.dist = td.TransformedDistribution(self.lamb.dist, LogTransform())

        self.tau.dist = td.HalfCauchy(torch.tensor([1.] * self.no_shrink))
        self.tau.dist = td.TransformedDistribution(self.tau.dist, LogTransform())

        self.update_distributions()
        self.reset_parameters()

        self.true_model = deepcopy(self.state_dict())

    def update_distributions(self):
        # Notice, that lamb & tau have no! hierarchical relation!
        self.W.dist_shrinked.scale = \
            (self.lamb.inv(self.lamb.data) * \
             self.tau.inv(self.tau.data) ** 2).view(-1, 1).expand(self.W.dist_shrinked.batch_shape)

    def reset_parameters(self, separated=False):

        # Setting tau according to seperated
        if separated:
            self.lamb.data = self.lamb.bij(torch.tensor([0.01]))
            self.tau.data = self.tau.bij(torch.tensor([0.01] * self.no_shrink))  # < ---------------------- fix value
        else:
            self.lamb.data = self.lamb.dist.sample()
            self.update_distributions()  # communicate to tau
            self.tau.data = self.tau.dist.sample()
        self.update_distributions()  # to ensure W's shrunken dist is updated before sampling it

        self.reset_Wb()

        self.init_model = deepcopy(self.state_dict())


if __name__ == '__main__':
    gHorse = GroupHorseShoe(no_in=3, no_out=2, bias=True, no_shrink=2)
    gHorse.reset_parameters(separated=True)
    X, y = gHorse.sample_model(200)
    gHorse.plot(X, y)

    gHorse.prior_log_prob()
