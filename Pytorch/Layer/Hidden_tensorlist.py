import torch
import torch.nn as nn
from torch.nn.modules import Module
import torch.functional as F

import torch.distributions as td
import hamiltorch

inplace_util = False


class Hidden(nn.Module):

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        super().__init__()
        self.no_in = no_in
        self.no_out = no_out
        self.bias = bias
        self.activation = activation

        self.tau_w = 1.
        self.tau_b = 1.
        self.dist = [td.MultivariateNormal(torch.zeros(self.no_in * self.no_out),
                                           self.tau_w * torch.eye(self.no_in * self.no_out)),
                     td.Normal(0., 1.)]

        self.W = nn.Parameter(torch.Tensor(no_out, no_in))

        if bias:
            self.b = nn.Parameter(torch.Tensor(self.no_out))

        self.reset_parameters()

        # occupied space in 1d vector
        self.n_params = sum([tensor.nelement() for tensor in self.parameters()])
        self.n_tensors = len(list(self.parameters()))

        # [tensor.shape for tensor in self.parameters()]
        # [torch.Size([1, 10]), torch.Size([1])]

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)
        if self.bias:
            nn.init.normal_(self.b)

    def forward(self, X, tensorlist):
        if inplace_util:
            return self.activation(X @ self.W.t() + self.b)
        else:
            W, b = tensorlist[0], tensorlist[1]
            return self.activation(X @ W.t() + b)
        # return self.activation(X @ W.t() + b)

    def prior_log_prob(self, tensorlist):
        return sum([dist.log_prob(tensor.view(tensor.nelement()))
                        for tensor, dist in zip(tensorlist, self.dist)])

    def likelihood(self, X, tensorlist):
        return td.Normal(self.forward(X, tensorlist), scale=torch.tensor(1.))

    def log_prob(self, X, y, tensorlist):
        # parsing 1d tensor to list of tensors
        if isinstance(tensorlist, torch.Tensor):  # i.e. tensorlist is 1D vec
            import Pytorch.utils

            if inplace_util:
                Pytorch.utils.unflatten_inplace(tensorlist, self) # immediate unpacking
            else:

                tensorlist = hamiltorch.util.unflatten(self, tensorlist)

                for p, new_p in zip(self.parameters(), tensorlist):
                    p.data = new_p

        return self.prior_log_prob(tensorlist).sum() + \
               self.likelihood(X, tensorlist).log_prob(y).sum()


if __name__ == '__main__':
    from hamiltorch.util import flatten, unflatten

    no_in = 2
    no_out = 10

    reg = Hidden(no_in, no_out, activation=nn.Identity())
    reg.forward(X=torch.ones(100, 2), tensorlist=[torch.ones(20).view(no_out, no_in), torch.ones(1)])
    reg.prior_log_prob(tensorlist=[torch.ones(20).view(no_out, no_in), torch.ones(1)])

    X_dist = td.Uniform(torch.tensor([-10., -10]), torch.tensor([10., 10]))
    X = X_dist.sample(torch.Size([100]))
    X.detach()
    X.requires_grad_()

    y = reg.likelihood(X, unflatten(reg, flatten(reg))).sample()
    y.detach()
    y.requires_grad_()

    init_theta = unflatten(reg, flatten(reg))
    print(reg.log_prob(X, y, init_theta))

    from functools import partial

    reg.log_prob = partial(reg.log_prob, X, y)
    print(reg.log_prob(init_theta))

    import hamiltorch
    import hamiltorch.util

    N = 400
    step_size = .3
    L = 5

    init_theta = hamiltorch.util.flatten(reg)
    params_hmc = hamiltorch.sample(log_prob_func=reg.log_prob, params_init=init_theta, num_samples=N,
                                   step_size=step_size, num_steps_per_sample=L)

    print()
