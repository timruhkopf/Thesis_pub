import unittest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.Layer.GAMs.GAM import GAM
from src.Samplers.mygeoopt import myRSGLD
from ..Test_Samplers.Convergence_teardown import Convergence_teardown
from ..Test_Samplers.util import posterior_mean, odict_pmean


class Test_GAM(Convergence_teardown, unittest.TestCase):

    def setUp(self) -> None:
        # Model & Data setup
        n = 1000
        # TODO Test also unbijected, with reset_parameters(tau=value)
        self.model = GAM(no_basis=5, order=1, activation=nn.Identity(), bijected=True)

        X, Z, y = self.model.sample_model(n=n)
        self.model.reset_parameters()
        self.X = X  # for plotting
        self.Z = Z
        self.y = y

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        self.Z.to(device)
        self.y.to(device)

        # self.model.LS = OLS(Z, y)

    def tearDown(self) -> None:
        Convergence_teardown.tearDown(self)

        # check MSE of true and 'posterior mean'
        self.model.load_state_dict(self.model.true_model)
        y_true = self.model.forward(self.Z)

        self.model.load_state_dict(self.model.init_model)
        y_init = self.model.forward(self.Z)

        chain = self.sampler.chain
        self.model.load_state_dict(odict_pmean(chain[:int(len(chain) * 3 / 4): 5]))
        y_pmean = self.model.forward(self.Z)

        losspmean = nn.MSELoss()(y_pmean, y_true)
        lossinit = nn.MSELoss()(y_init, y_true)
        self.assertTrue(losspmean < lossinit, msg='In terms of avg MSE, the model was not improved during '
                                                  'sampling (using posterior mean on last quarter of the '
                                                  'thinned chain as reference)')

        self.assertTrue(losspmean < torch.tensor(0.5),
                        msg='In MSE terms and in comparison of true model to posterior mean'
                            '(on last quarter of the thinned chain as reference),'
                            'the model walked out of "convergence bound 0.5" (i.e. avg MSE difference) ')
        # self.assertTrue(torch.allclose(loss, torch.tensor(1.), rtol=0.2), msg= "check MSE of true and 'posterior "
        #                                                                         "mean' failed")

    def test_samplable(self):
        # Sampler set up
        eps = 0.001
        sampler_param = dict(epsilon=eps)

        burn_in, n_samples = 100, 1000
        batch_size = 100

        # dataset setup
        trainset = TensorDataset(self.Z, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # init sampler & sample
        Sampler = myRSGLD
        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)
        self.sampler.loss = self.sampler.log_probs.detach().numpy()

        self.model.plot(self.X, self.y, self.sampler.chain[::n_samples // 30],
                        title='subsampled chain (every 30th state, sorted)')

        # plot the posterior mean
        pmean = posterior_mean(self.sampler.chain[int(n_samples * 3 / 4)::5])
        tau = pmean[0].reshape((1,))
        W = pmean[1:].reshape(self.model.W.shape)

        from collections import OrderedDict
        self.model.plot(self.X, self.y, chain=[OrderedDict([('tau', tau), ('W', W)])],
                        title='posterior mean on last quarter of the chain; interleave: every 5th')


if __name__ == '__main__':
    unittest.main(exit=False)
