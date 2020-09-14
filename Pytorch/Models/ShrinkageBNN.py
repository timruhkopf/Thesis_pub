import torch
import torch.distributions as td
import torch.nn as nn

from Pytorch.Models.BNN import BNN
from Pytorch.Layer.Hidden import Hidden, Hidden_flat
from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Layer.Group_HorseShoe import Group_HorseShoe


class ShrinkageBNN(BNN):
    # available shrinkage layers
    shrinkage_type = {
        'glasso': Group_lasso,
        # 'gspike': layer.Group_SpikeNSlab,
        'ghorse': Group_HorseShoe}

    layer_type = {
        'flat': Hidden_flat,
        'normal': Hidden
    }

    def __init__(self, hunits=[2, 10, 1], activation=nn.ReLU(), final_activation=nn.Identity(),
                 shrinkage='glasso', prior='normal', seperated=False, bijected=True, heteroscedast=False):
        """
        Shrinkage BNN is a regular BNN, but uses a shrinkage layer, which in the
        current implementation shrinks the first variable in the X vector,
        depending on whether or not it helps predictive performance.

        :param hunits: hidden units per layer
        :param activation: nn.ActivationFunction instance
        :param final_activation: nn.ActivationFunction instance
        :param shrinkage: 'glasso', 'gsplike', 'ghorse' each of which specifies the
        grouped shrinkage version of lasso, spike & slab & horseshoe respectively.
        See detailed doc in the respective layer. All of which assume the first variable
        to be shrunken; i.e. provide a prior log prob model on the first column of W
        :param gam_param:
        """
        if len(hunits) < 3:
            raise ValueError('In its current form, shrinkage BNN works only for '
                             'multiple layers: increase no. layers via hunits')
        self.shrinkage = shrinkage
        self.seperated = seperated
        self.bijected = bijected
        self.prior = prior

        BNN.__init__(self, hunits, activation, final_activation, heteroscedast)

    def define_model(self):
        # Defining the layers depending on the mode.

        L = self.layer_type[self.prior]
        S = self.shrinkage_type[self.shrinkage]

        self.layers = nn.Sequential(
            S(self.hunits[0], self.hunits[1], True, self.activation, self.bijected),
            *[L(no_in, no_units, True, self.activation)
              for no_in, no_units in zip(self.hunits[1:-2], self.hunits[2:-1])],
            L(self.hunits[-2], self.hunits[-1], bias=False, activation=self.final_activation))

        if self.heteroscedast:
            self.sigma_ = nn.Parameter(torch.Tensor(1))
            self.dist_sigma = td.TransformedDistribution(td.Gamma(0.01, 0.01), td.ExpTransform())
            self.sigma = self.dist_sigma.sample()
        else:
            self.sigma = torch.tensor(1.)


if __name__ == '__main__':
    import random
    from copy import deepcopy

    sbnn = ShrinkageBNN()
    no_in = 2
    no_out = 1
    n = 1000
    # sampling with effect
    # sbnn.reset_parameters(seperated=True)
    sbnn.reset_parameters(seperated=True)

    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n]))
    # sbnn.reset_parameters(seperated=True)
    sbnn.true_model = deepcopy(sbnn.state_dict())
    y = sbnn.likelihood(X).sample()

    sbnn.log_prob(X, y)

    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    sbnn.reset_parameters(seperated=False)
    sbnn.plot(X, y, **{'title': 'Shrinkage BNN @ init'},
              path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/Results/init')

    sampler = Sampler(sbnn, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    import random

    sampler.model.plot(X, y, chain=random.sample(sampler.chain, 30),
                       path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/Results'
                            '/datamodel',
                       **{'title': 'Shrinkage BNN with subsampled chain'})
    sampler.traceplots(path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/Results'
                            '/traceplot')
    sampler.acf_plots(nlags=500,
                      path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/Results'
                           '/acf')

    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA

    step_size = 0.001
    num_steps = 5000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    L = 10
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    sbnn.reset_parameters()
    sgnht = SGNHT(sbnn, trainloader,
                  step_size, num_steps, burn_in, pretrain=pretrain, tune=tune,
                  L=L,
                  num_chains=num_chains)
    sgnht.sample()
    print(sgnht.chain)
    import random

    sgnht.model.plot(X, y, random.sample(sgnht.chain, 30), **{'title': 'Shrinkage BNN with subsampled chain'})
