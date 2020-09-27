import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Util.Util_Distribution import LogTransform


class Group_HorseShoe(Group_lasso):
    prior_log_prob = Group_lasso.prior_log_prob

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True):
        """
        Group Lasso Layer, which is essentially a Hidden Layer, but with a different
        prior structure: W is partitioned columwise - such that all weights outgoing
        the first variable are shrunken by bayesian lasso.
        for params see Hidden
        :param bijected: bool. indicates whether or not the shrinkage variances
        'tau' and 'lamb' are to be bijected i.e. unconstrained on space R.
        as consequence, the self.update_method must change. the self.prior_log_prob
        is automatically adjusted by the jacobian via td.TransformedDistribution.
        """
        self.bijected = bijected
        Hidden.__init__(self, no_in, no_out, bias, activation)
        self.dist['alpha'] = td.HalfCauchy(0.3)

        self.dist['tau'] = td.HalfCauchy(torch.tensor([1.]))
        if self.bijected:
            self.dist['tau'] = td.TransformedDistribution(self.dist['tau'], LogTransform())

        self.reset_parameters()

    def update_distributions(self):
        if self.bijected:
            self.dist['W_shrinked'].scale = \
                self.dist['tau'].transforms[0]._inverse(self.tau.clone().detach())  # CAREFULL, THIS is ESSENTIAL for
            # the model to run -and not invoke a second backward (in which already gradients have been lost)
        else:
            self.dist['W_shrinked'].scale = self.tau


if __name__ == '__main__':
    no_in = 2
    no_out = 1
    n = 1000
    X_dist = td.Uniform(torch.ones(no_in) * -10., torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)

    horse = Group_HorseShoe(no_in, no_out, bias=True, activation=nn.ReLU(), bijected=True)
    horse.reset_parameters(seperated=False)
    horse.true_model = horse.state_dict()
    y = horse.likelihood(X).sample()

    # check reset_parameters &  check prior_log_prob
    horse.reset_parameters(seperated=True)
    horse.plot(X, y, **{'title': 'G-lasso'})
    print(horse.prior_log_prob())

    # check update_distributions
    # horse.reset_parameters()
    print(horse.W[:, 0])
    print(horse.dist['W_shrinked'].scale)
    # value of log prob changed due to change in variance
    print(horse.dist['W_shrinked'].log_prob(0.))
    print(horse.dist['W_shrinked'].cdf(0.))  # location still 0

    # check alpha value
    # horse.alpha

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter()
    # writer.add_graph(horse, input_to_model=X, verbose=True) # FAILS unexpectedly
    # writer.close()

    # check sampling ability.
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # horse.reset_parameters(False)
    # horse.plot(X_joint, y)

    torch.autograd.set_detect_anomaly(True)
    sampler = Sampler(horse, epsilon=0.003, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    import random

    horse.plot(X, y, chain=random.sample(sampler.chain, 30),
               **{'title': 'G-lasso'})

    horse.plot(X, y, chain=sampler.chain[-30:],
               **{'title': 'G-lasso'})
