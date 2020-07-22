import torch
import torch.nn as nn
import torch.distributions as td

from hamiltorch.util import flatten, unflatten
from Pytorch.Models.ModelUtil import Vec_Model, Model_util


class Hidden(nn.Module, Vec_Model, Model_util):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """
        Hidden Layer, that provides its prior_log_prob model

        :param no_in:
        :param no_out:
        :param bias: bool, whether or not to use bias in the computation
        :param activation: Any "nn.activation function" instance. Examine
        https://pytorch.org/docs/stable/nn.html?highlight=nn%20relu#torch.nn.ReLU
        should work with their functional correspondance e.g. F.relu. Notably,
        A final Layer in Regression setup looks like
        Hidden(10, 1, bias=False, activation=nn.Identity())
        :param optimbased: bool. A Flag indicating whether or not the used sampler
        makes use of the torch.optim lib -> training based on actual nn.Parameter.
        if the model is trained based on a vector representation of the model, it
        will require parsing of a vec in the log_prob--> and thus optimbased = False

        :notice Due to the inability of writing inplace to nn.Parameters via
        vector_to_parameter(vec, model) whilst keeping the gradients. Using
        vector_to_parameter fails with the sampler.
        The currently expected Interface of log_prob is to recieve a 1D vector
        and use nn.Module & nn.Parameter (surrogate parameters trailing an
        underscore) merely as a skeleton for parsing the 1D
        vector using hamiltorch.util.unflatten (not inplace) - yielding a
        list of tensors. Using the property p_names, the list of tensors can
        be setattr-ibuted and accessed as attribute
        """

        super().__init__()

        self.no_in = no_in
        self.no_out = no_out
        self.bias = bias
        self.activation = activation

        # Add the prior model
        self.dist = {}
        self.define_model()

        # initialize the parameters
        self.reset_parameters()
        self.true_model = None

    # (1) USER MUST DEFINE THESE FUNCTIONS: ------------------------------------
    # TO MAKE THIS A VEC MODEL
    def define_model(self):
        """place to instantiate all necessary nn.Parameter (& in case of non-optim-based
        samplers also the surrogate non-nn parameter.
        Notice, how nn.parameter must have trailing underscore (self.W_) and how
        all actions such as forward, prior_log_prob and log_prob are performed on
        the surrogate parameters."""
        self.tau_w = 1.
        self.dist['W'] = td.MultivariateNormal(
            torch.zeros(self.no_in * self.no_out),
            self.tau_w * torch.eye(self.no_in * self.no_out))  # todo refactor this to td.Normal()

        self.W_ = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W = None

        # add optional bias
        if self.bias:
            self.tau_b = 1.
            self.dist['b'] = td.Normal(0., self.tau_b)
            self.b_ = nn.Parameter(torch.Tensor(self.no_out))
            self.b = None

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W_)
        self.W = self.W_.data

        if self.bias:
            nn.init.normal_(self.b_)
            self.b = self.b_.data

    # (2) DEFINE THESE FUNCTIONS IN REFERENCE TO SURROGATE PARAM: --------------
    # & inherit for layers
    def forward(self, X):
        XW = X @ self.W
        if self.bias:
            XW += self.b
        return self.activation(XW)

    def prior_log_prob(self):
        """evaluate each parameter in respective distrib."""
        param_names = self.p_names
        param_names.remove('W')

        value = torch.tensor(0.)
        if param_names is not None:
            for name in param_names:
                value += self.dist[name].log_prob(self.get_param(name)).sum()

        value += self.dist['W'].log_prob(self.W.view(self.W.nelement()))

        return value

    def likelihood(self, X):
        """:returns the conditional distribution of y | X"""
        # TODO update likelihood to become an attribute distribution,
        #  which is updated via self.likelihood.__init__(newloc, scale)
        #  or even use self.likelihood.loc = newloc
        return td.Normal(self.forward(X), scale=torch.tensor(1.))

    def my_log_prob(self, X, y):
        """
        SG flavour of Log-prob: any batches of X & y can be used
        make sure to pass self.log_prob to the sampler, since self.my_log_prob
        is a convenience mask

        Notice, that self.log_prob has two modes operandi:
        (1) self.log_prob(X,y), which returns the log_prob with current state of
        'parameters'. This is particularly handy with optim based samplers,
        since 'parameters' are literally nn.Parameters and update their state based
        on optim proposals (always up to date)

        (2) self.log_prob(X,y, vec), modus is available iff inherits from VecModel
        (-> and used for vec based samplers such as Hamiltorchs). When vec,
        the vector representation (1D Tensor) of the model is provided, the model's
        surrogate 'parameter' (not nn.Parameter) are updated - and the models state
        under vec is evaluated for X & y

        Irrespective of the mode choice, the self.log_prob operates as a SG-Flavour,
        i.e. is called with ever new X & y batches. However, working with Samplers
        operating on the entire Dataset at every step, the method can be modified
        calling  self.closure_log_prob(X, y) to fix every consequent call to
        self.log_prob()  or self.log_prob(vec) on the provided dataset X, y
        (using functools.partial)"""
        return  -self.prior_log_prob().sum() - \
               self.likelihood(X).log_prob(y).sum()


if __name__ == '__main__':
    no_in = 10
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    reg.true_model = reg.vec

    # reg.W = reg.W_.data
    # reg.b = reg.b_.data
    reg.forward(X=torch.ones(100, no_in))
    reg.prior_log_prob()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()

    # create 1D vector from nn.Parameters as Init + SG-Mode
    init_theta = flatten(reg)
    print(reg.log_prob(X, y, init_theta))
    print(reg.log_prob(X, y))

    init_theta1 = torch.ones_like(init_theta)
    print(reg.log_prob(X, y))
    print(reg.log_prob(X, y, init_theta1))
    print(reg.log_prob(X, y))  # Carefull: notice the update of state!

    # setting the log_prob to full dataset mode
    reg.closure_log_prob(X, y)
    print(reg.log_prob(init_theta))  # vec_mode: state change!
    print(reg.log_prob())  # optim_mode

    # Estimation example
    import hamiltorch
    import hamiltorch.util

    N = 2000
    step_size = .3
    L = 5

    # HMC (tuned with NUTS stepsize)
    nuts_step_size = 0.014175990596413612
    init_theta = hamiltorch.util.flatten(reg)
    params_hmc = hamiltorch.sample(
        log_prob_func=reg.log_prob, params_init=init_theta, num_samples=N,
        step_size=step_size, num_steps_per_sample=L)

    # HMC NUTS
    N = 2000
    step_size = .3
    L = 5
    burn = 500
    N_nuts = burn + N
    params_hmc_nuts = hamiltorch.sample(log_prob_func=reg.log_prob, params_init=init_theta,
                                        num_samples=N_nuts, step_size=step_size, num_steps_per_sample=L,
                                        sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
                                        desired_accept_rate=0.8)

    # check helper functions
    reg.vec
    reg.parameters_list
    reg.p_names
    reg.parameters_dict
    reg.n_tensors
