# https://github.com/hsvgbkhgbv/Thermostat-assisted-continuously-tempered-Hamiltonian-Monte-Carlo-for-Bayesian-learning

import platform

print('python_version ==', platform.python_version())
import torch

print('torch.__version__ ==', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
import time
import argparse
import numpy as np
from tqdm import tqdm

from Pytorch.Samplers.Samplers import Sampler
from thirdparty_repo.hsvgbkhgbv.tacthmc import TACTHMC


class TACT_HMC(TACTHMC, Sampler):
    def __init__(self, model, N, trainloader, eta_theta0=1.7e-8, eta_xi0=1.7e-10,
                 c_theta0=0.1, c_xi0=0.1, gamma_theta0=1, gamma_xi0=1,
                 enable_cuda='store_true', standard_interval=0.1,
                 gaussian_decay=1e-3, version='accurate', temper_model='ABF'):
        """

        :param model: nn.Module child class
        :param N:
        :param eta_theta0:
        :param eta_xi0:
        :param c_theta0:
        :param c_xi0:
        :param gamma_theta0:
        :param gamma_xi0:
        :param enable_cuda:
        :param standard_interval:
        :param gaussian_decay:
        :param version: string: 'accurate', 'approx', changes the dynamics behaviour
        :param temper_model: string: ABF, Metadynamics. the latter is not explicitly supported here!
        """
        # FIXME: TACTHMC.__init__ expects the model to consist of torch linear, conv or lstm
        #   objects only, leading to the following conditional initialization:
        #    self.pattern1 = re.compile(r'linear|conv')
        #    self.pattern2 = re.compile(r'lstm')
        #   for name, module in self.model._modules.items():
        #       if self.pattern1.match(name):
        #   ----------------------------------------
        #   consequently: self.model._modules.items() will have all the right
        #   and explicit parameters that are to be initialized.
        #   However in my case, each layer is not a module, thus
        #   self.model._modules.items() turns out to be the activation function only!
        super(TACTHMC).__init__(model, N, eta_theta0, eta_xi0, c_theta0, c_xi0,
                                gamma_theta0, gamma_xi0, enable_cuda, standard_interval,
                                gaussian_decay, version, temper_model)

        self.trainloader = trainloader
        self.info_string = 'xi:{:+7.4f}; fU:{:+.3E}; r_xi:{:+.3E}; loss:{:6.4f}; ' \
                           'thermostats_param:{:6.3f}; thermostats_xi:{:6.3f}; ' \
                           'tElapsed:{:6.3f}'

    def sample(self, trainloader, device_num=7, num_epochs=2000,
               prior_precision=1e-3, evaluation_interval=50):

        if torch.cuda.is_available():
            torch.cuda.set_device(device_num)
            self.model.cuda()

        self.resample_momenta()
        print(self.model)
        nIter = 0
        tStart = time.time()

        for epoch in tqdm(range(1, 1 + num_epochs)):
            print("#############################################################")
            print("This is the epoch: ", epoch)
            print("#############################################################")

            for i, (X, y) in enumerate(trainloader):
                batch_size = X.data.size(0)
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                self.model.zero_grad()
                loss = self.model.log_prob(X, y)

                # FIXME: check up on the following
                for param in self.model.parameters():
                    loss += prior_precision * torch.sum(param ** 2)

                loss.backward()
                '''update params and xi'''
                self.update(loss)
                nIter += 1
                self.standard_interval
                if nIter % evaluation_interval == 0:
                    print(self.info_string.format(
                        self.model.xi.item(),
                        self.fU.item(),
                        self.model.r_xi.item(),
                        loss.data.item(),
                        self.get_z_u(),
                        self.get_z_xi(),
                        time.time() - tStart))

                    self.resample_momenta()
                    tStart = time.time()


if __name__ == '__main__':
    # INITILIZATION ISSUE:
    # my layers are actually nn.Module's and not as e.g. linear or conv or lstm
    # from .module import Module
    # as a result, the TACTHMC init already fails at finding the parameters!

    class Line(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.Tensor(10, 1))
        def forward(self, X):
            return X @ self.W1

    class Hidden(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 1, True)
            self.line = Line()
            self.W = nn.Parameter(torch.Tensor(10, 1))
            self.I = nn.Identity()

        def forward(self, X):
            return nn.ReLU(self.lin(X))

    model = Hidden()
    list(model.parameters())
    list(model.named_parameters())
    for name, module in model._modules.items():
        print(name, module)  # <<<<---------------------


    from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel
    model = Hidden_ProbModel(no_in=10, no_out=1, bias=True, activation=nn.Identity())
    for name, module in model._modules.items():
        print(name, module)  # <<<<--------------------- FIXME:

    # EXAMPLE: -----------------------------------------------------------------
    import torch.distributions as td
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel

    no_in = 10
    no_out = 1

    model = Hidden_ProbModel(no_in, no_out, bias=True, activation=nn.Identity())

    # model.reset_parameters()
    print(model.parameters_dict)

    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = model.likelihood(X).sample()

    # setting up the sampler
    # trainloader --------------------------------------------------------------
    batch_size = 64
    burn_in = 30000
    epochs = 2000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    cuda_availability = enable_cuda = 'store_true' and torch.cuda.is_available()
    N = len(trainloader.dataset)
