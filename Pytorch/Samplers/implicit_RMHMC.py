# https://github.com/AdamCobb/hamiltorch
# https://adamcobb.github.io/journal/hamiltorch.html  (vignette)

if __name__ == '__main__':
    import torch
    import torch.distributions as td
    from torch import nn
    from Pytorch.utils import unflatten_inplace, flatten


    # translating from 1d vector to model -- get hamiltorch to work! (requires only log_prob and start vector!
    # still autograd?
    class Regression(nn.Module):
        def __init__(self, wdim=2):
            super(Regression, self).__init__()
            self.wdim

            self.W_dist = td.MultivariateNormal(loc=torch.zeros(wdim), covariance_matrix=torch.eye(wdim))

            self.W = nn.Parameter(torch.randn(5, 1))  # (no. connections, no_nodes)
            self.std_bij = nn.Parameter(torch.ones(1))  # just for demo of flatten / unflatten

            self.likelihood = None

        # @property
        # def std(self):
        #     # inverse transformation using
        #     self.bij
        #     self.std_bij

        def forward(self, X, W):
            return X @ W

        def unbij_homo_scedast(self, X, y, bij_sigma=True):
            SIGMA = td.Gamma(0.1, 0.1)
            if bij_sigma:
                SIGMA = td.TransformedDistribution(SIGMA, td.ExpTransform())

            def log_prob(self, theta):
                W, sigma = theta[:self.wdim], theta[self.wdim:]

                return self.W_dist.log_prob() + \
                       SIGMA.log_prob() + \
                       self.likelihood(self.forward(X, W), sigma).log_prob(y)

            return log_prob



    X = torch.randn(2, 5)  # (no.obs, dim W)
    model = Regression()
    model.forward(X)

    vec = flatten(model)
    vec = torch.ones(6)
    unflatten_inplace(vec, model)
