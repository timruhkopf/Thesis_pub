import unittest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.Layer.GAM import GAM
from src.Samplers.mygeoopt import myRSGLD
from ..Test_Samplers.Convergence_teardown import Convergence_teardown
from ..Test_Samplers.util import posterior_mean, chain_mat


class Test_GAM_samplable(unittest.TestCase):

    def test_samplable_unbijected(self):
        self.model = GAM(no_basis=5, order=1, activation=nn.Identity(), bijected=False)

    def test_bijected_samplable(self):
        self.model = GAM(no_basis=5, order=1, activation=nn.Identity(), bijected=True)

    def tearDown(self) -> None:
        # Data setup
        n = 1000
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
        n_samples = len(self.sampler.chain)

        self.model.plot(self.X, self.y, self.sampler.chain[::n_samples // 30],
                        title='subsampled chain (every 30th state, sorted)')

        # plot the posterior mean
        pmean = posterior_mean(self.sampler.chain[int(n_samples * 3 / 4)::5])
        tau = pmean[0].reshape((1,))
        W = pmean[1:].reshape(self.model.W.shape)

        from collections import OrderedDict
        self.model.plot(self.X, self.y, chain=[OrderedDict([('tau', tau), ('W', W)])],
                        title='posterior mean on last quarter of the chain; interleave: every 5th')

        self.X_backup = self.X  # for plotting when in teardown
        self.X = self.Z  # for Convergence teardown

        # did tau ever hit zero or became negative?
        mat = chain_mat(self.sampler.chain)
        taus = mat[:, 0]
        if self.model.bijected:
            self.assertFalse(any(self.model.tau.dist.transforms[0]._inverse(taus) <= 0.),
                             msg='the unbijected sampling did produce negative or 0 values for tau')
        else:
            if any(taus <= 0):
                Warning('for an unbijectd model, sampling produced negative or zero values for tau.'
                        'This can happen, but be careful with the interpretation of the unittest')

        Convergence_teardown.tearDown(self)


if __name__ == '__main__':
    unittest.main(exit=False)
