import torch
import torch.distributions as td
import torch.nn as nn

from Pytorch.Models.BNN import BNN
from Pytorch.Layer.Hidden import Hidden, Hidden_flat
from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Layer.Group_HorseShoe import Group_HorseShoe
from Pytorch.Layer.Hierarchical_Group_HorseShoe import Hierarchical_Group_HorseShoe
from Pytorch.Layer.Hierarchical_Group_lasso import Hierarchical_Group_lasso


class ShrinkageBNN(BNN):
    # available shrinkage layers
    shrinkage_type = {
        'glasso': Group_lasso,
        # 'gspike': layer.Group_SpikeNSlab,
        'ghorse': Group_HorseShoe,
        'multihorse': Hierarchical_Group_HorseShoe,
        'multilasso': Hierarchical_Group_lasso}

    layer_type = {
        'flat': Hidden_flat,
        'normal': Hidden
    }

    def __init__(self, hunits=[2, 10, 1], activation=nn.ReLU(), final_activation=nn.Identity(),
                 shrinkage='ghorse', prior='normal', seperated=False, bijected=True, heteroscedast=False):
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
    import matplotlib

    matplotlib.use('TkAgg')

    no_in = 2
    no_out = 1
    n = 1000

    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n]))

    sbnn = ShrinkageBNN(hunits=[no_in, 10, 5, no_out], shrinkage='multilasso', bijected=True, seperated=True)
    sbnn.reset_parameters(seperated=True)
    sbnn.true_model = deepcopy(sbnn.state_dict())
    y = sbnn.likelihood(X).sample()

    sbnn.log_prob(X, y)

    sbnn.reset_parameters(seperated=False)
    sbnn.init_model = deepcopy(sbnn.state_dict())

    sbnn.plot(X[:400], y[:400])  # **{'title': 'Shrinkage BNN @ init'})

    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    import pickle
    import matplotlib

    matplotlib.use('Agg')

    path = '/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/manualShrinkage_glasso/'
    counter = 0
    while True:
        counter += 1
        basename = path + 'Shrinkage_Hierarchical_glasso{}'.format(str(counter))
        try:
            # generate a new true model
            sbnn = ShrinkageBNN(hunits=[no_in, 10, 5, no_out], shrinkage='multilasso', bijected=True, seperated=True)
            sbnn.reset_parameters(seperated=True)
            sbnn.true_model = deepcopy(sbnn.state_dict())
            y = sbnn.likelihood(X).sample()

            # generate a new init model
            sbnn.reset_parameters()
            sbnn.init_model = deepcopy(sbnn.state_dict())
            burn_in, n_samples = 1000, 1000
            sampler = Sampler(sbnn, epsilon=0.002, L=2)
            sampler.sample(trainloader, burn_in, n_samples)
            sampler.model.check_chain(sampler.chain)

            matplotlib.use('TkAgg')
            sampler.model.plot(X[0:100], y[0:100], sampler.chain[-30:])

            sampler.model.plot(X[0:100], y[0:100], sampler.chain[-30:],
                               path=basename)
        except:
            # sbnn.load_state_dict(sbnn.init_model)
            continue

        try:
            burn_in, n_samples = 10000, 10000
            sampler = Sampler(sbnn, epsilon=0.001, L=2)
            sampler.sample(trainloader, burn_in, n_samples)
            # sampler.model.check_chain(sampler.chain) # already included in .sample()
            sampler.model.plot(X[0:100], y[0:100], sampler.chain[:30])
            sampler.model.plot(X[0:100], y[0:100], sampler.chain[-30:],
                               path=basename)
            sampler.save(basename)
            config = {'n': 1000, 'n_val': 100, 'model_class': ShrinkageBNN, 'model_param':
                dict(hunits=[no_in, 10, 5, no_out], shrinkage='multilasso', bijected=True, seperated=True),
                      'sampler_name': 'RHMC', 'sampler_param': dict(epsilon=0.001, L=2), 'seperated': True}

            sampler.traceplots(path=basename + '_traces_baseline.pdf')
            sampler.traceplots(path=basename + '_traces.pdf', baseline=False)

            with open(basename + '_config.pkl', 'wb') as handle:
                pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(basename + '_chain.pkl', 'wb') as handle:
                pickle.dump(sampler.chain, handle, protocol=pickle.HIGHEST_PROTOCOL)

        except:
            continue

    import random

    sampler.model.plot(X[0:100], y[0:100], sampler.chain[:30])
    sampler.model.plot(X[0:100], y[0:100], random.sample(sampler.chain, 30))
    sampler.model.plot(X[0:100], y[0:100], chain=[sampler.model.init_model])

    print(sampler.chain[0])
    print(sampler.chain[-1])

    sampler.traceplots()
    sampler.acf_plots(nlags=200)

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
