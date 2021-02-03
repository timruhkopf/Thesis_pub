import torch
import torch.nn as nn
import torch.distributions as td
from copy import deepcopy

class BaseModel(nn.Module):
    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """

        :param no_in:
        :param no_out:
        :param bias:
        """
        # make use of torch.nn's registry and functionality
        nn.Module.__init__(self)

        self.no_in = no_in
        self.no_out = no_out
        self.has_bias = bias
        self.activation = activation  # Note that this prints out nastily

        self.define_model()
        self.reset_parameters()

        self.true_model = deepcopy(self.state_dict())

    # (Model defining & intialising functions) ---------------------------------
    def define_model(self):
        """
        Distributional assumptions of parameters
        """
        self.var = nn.Parameter(torch.tensor([1.]))
        self.var.dist = td.Gamma(torch.tensor([1.]), torch.tensor([1.]))

        self.tau_w = nn.Parameter(torch.tensor([1.]))
        self.tau_w.dist = td.Gamma(torch.tensor([1.]), torch.tensor([1.]))

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W.dist = td.Normal(loc=torch.zeros_like(self.W.data), scale=self.tau_w)

        if self.has_bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))
            self.b.dist = td.Normal(torch.zeros(self.no_out), 1.)

    def forward(self, X):
        """
        Predictive path of the model
        :param X:
        :return:
        """
        XW = X @ self.W
        if self.has_bias:
            XW += self.b
        return self.activation(XW)

    def reset_parameters(self):
        """Initialise the parameters from their respective distributions"""
        for p in self.parameters():
            self.update_distributions()
            p.data = p.dist.sample()

    # (SAMPLING RELATED FUNCTIONS) ---------------------------------------------
    def update_distributions(self):
        """hierarchical models require, that an update of a parameter immediately
        has to be reflected in the distributions of those other parameters that
        depend on this parameter. Ideally, these kind of changes are reflected by
        a computation graph, that indicates which nn.Parameter's distributional
        parameters require an update - or rather point immediately to the underlying
        parameter in a property like fashion. The implementation of such a graph
        is not straight forward. Further, as in case of Multivariate Normal,
        lazy_property prevents from changing the covariance matrix (the likes of
        which are usually desirable).
        """
        self.W.dist = td.Normal(loc=torch.zeros_like(self.W.data), scale=self.tau_w)

    def prior_log_prob(self):
        """calculates log probability of all the parameters under their distributional
        assumption given the current state"""
        for p in self.parameters():
            p.dist.log_prob(p.data)

    def likelihood(self, X):
        """Defines y's distributional assumption
        Here, a heteroscedastic normal distribution.
        :return torch.distribution object"""
        return td.Normal(self.forward(X), scale=torch.sqrt(self.var))

    def log_prob(self, X, y):
        self.update_distributions()  # ensure proper state of hierarchy
        return self.likelihood(X).log_prob(y).sum() + self.prior_log_prob()


if __name__ == '__main__':
    # A simple heteroscedastic regression (Hidden Layer) class with a learnable
    # joint prior variance for the weight parameters
    m = BaseModel(10, 1, True)

    # (SAMPLING) ---------------------------------------------------------------
    # 1) for a detailed example, using all 6 available samplers, see src.Layer.Hidden
    # 2) Be aware, that my work was simulating Data, so for applications,
    #    the interface for the sampler must be revised: think about how the data
    #    is supposed to be presented. Also using any shrinkage or structured model's
    #    implementation assumes a hierarchical structure exclusively for the first
    #    variable!

    # (Grid Search) ------------------------------------------------------------
    # This is very messy and more elegant methods such as BOHB should be used
    # to optimise a model. For the time being i stuck with grid search due to the
    # sensitivity to initialisation, that was persistent.

    # (BIJECTION) --------------------------------------------------------------
    # Notice, that the variance parameters ideally are bijected for sampling to
    # unconstrain the parameter space and avoid invalid solutions. To do so
    # make use of td.TransformedDistribution as in
    from src.Util.Util_Distribution import LogTransform

    tau = torch.tensor([1.])
    tau.dist = td.Gamma(torch.tensor([1.]), torch.tensor([1.]))
    tau.dist = td.TransformedDistribution(tau.dist, LogTransform())

    # with the unbijecting operation
    tau.dist.transforms[0]._inverse(tau)

    # For more details on bijection please revise either src.Layer.GAM or
    # src.Layer.Hierarchical_Group_HorseShoe

    # (Additional structure) ---------------------------------------------------
    # Models such as the Shrinkage models used in this Repo pose additional,
    # hierarchical structure on the weight matrix. This can easily acomplished
    # in this framework. Compare src.Layer.Hierarchical_Group_lasso

    # (Composite Models) -------------------------------------------------------
    # The basic structure of these models (e.g. BNNs) are extremely similar.
    # I highly encourage to read the source code src.Models.BNN. The main
    # difference lies in the BNN.layers attribute, that holds the sequential
    # structure of the BNN using Base models, and the forward path. All other
    # methods are merely delegating to the appropriate BaseModel instance.
