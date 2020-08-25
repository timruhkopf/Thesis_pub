import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Util.ModelUtil import Model_util


class Hidden(nn.Module, Model_util):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        """
        Hidden Layer, that provides its prior_log_prob model

        :param no_in:
        :param no_out:
        :param bias: bool, whether or not to use bias in the computation
        :param activation: Any "nn.activation function" instance. Examine
        https://pytorch.org/docs/stable/nn.html?highlight=nn%20relu#torch.nn.ReLU
        should work with their functional correspondance e.g. F.relu. Notably,
        A final Layer in Regression setup looks like e.g.
        Hidden(10, 1, bias=False, activation=nn.Identity())
        """

        super().__init__()

        self.no_in = no_in
        self.no_out = no_out
        self.has_bias = bias
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
        self.tau_w = 1.
        self.dist = {'W': td.MultivariateNormal(
            torch.zeros(self.no_in * self.no_out),
            self.tau_w * torch.eye(self.no_in * self.no_out))}  # todo refactor this to td.Normal()

        self.W = nn.Parameter(torch.Tensor(self.no_in, self.no_out))

        if self.has_bias:
            self.tau_b = 1.
            self.dist['b'] = td.Normal(0., self.tau_b)
            self.b = nn.Parameter(torch.Tensor(self.no_out))

    @torch.no_grad()
    def reset_parameters(self):
        # nn.init.xavier_normal_(self.W)
        # with torch.no_grad():
        nn.init._no_grad_zero_(self.W)
        self.W.add_(self.dist['W'].sample().view(self.W.shape))

        if self.has_bias:
            nn.init.normal_(self.b)

    def update_distributions(self):
        # here no hierarchical distributions exists
        return None

    # (2) DEFINE THESE FUNCTIONS IN REFERENCE TO SURROGATE PARAM: --------------
    # & inherit for layers
    def forward(self, X):
        XW = X @ self.W
        if self.has_bias:
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
        return self.prior_log_prob().sum() + \
               self.likelihood(X).log_prob(y).sum()


if __name__ == '__main__':
    no_in = 2
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    reg.true_model = reg.state_dict()

    # reg.W = reg.W_.data
    # reg.b = reg.b_.data
    reg.forward(X=torch.ones(100, no_in))
    reg.prior_log_prob()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()

    print(reg.log_prob(X, y))

    # Estimation example
    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    num_samples = 1000

    step_size = 0.01
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    hmc_traj_length = 24
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    reg.reset_parameters()
    sgnht = SGNHT(reg, X, y, X.shape[0],
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  hmc_traj_length=hmc_traj_length,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)

    sgnht.model.plot(X, y, **{'title':'Hidden'})