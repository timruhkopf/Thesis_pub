# https://github.com/ruqizhang/csgmcmc

from __future__ import print_function
import sys
import torch.backends.cudnn as cudnn
import numpy as np
import random

from hamiltorch.util import flatten

# from models import *
from Pytorch.Util.DistributionUtil import *

sys.path.append('..')
use_cuda = torch.cuda.is_available()


class CSG_MCMC:  # (Samplers)
    def __init__(self, temperature=1. / 50000, alpha=0.9,
                 weight_decay=5e-4, lr_0=0.5, M=4, epochs=20, num_batch=10):
        """
        Instantiate a CSG_MCMC Sampler; i.e. depending on alpha, a cSGLD or cSGHMC
        :param temperature: default 1 / dataset_size
        :param alpha: float, 1: SGLD, <1 SGHMC
        :param weight_decay:
        :param lr_0: initial learning rate
        :param M: number of cycles
        :param epochs: number of epochs to train
        """
        print('==> INSTANTIATING SAMPLER..')
        self.T = epochs * num_batch  # total number of iterations
        self.M = M
        self.lr_0 = lr_0
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.temperature = temperature
        self.epochs = epochs
        self.num_batch = num_batch

        # set up
        self.net = None
        self.trainloader = None
        self.criterion = None
        self.device_id = 1

        self.use_cuda = torch.cuda.is_available()
        self.chain = list()

    def __repr__(self):
        if self.alpha == 1:
            return 'cSGLD'
        elif self.alpha < 1:
            return 'cSG-HMC'

    def _update_params(self, lr, epoch):
        """
        INTERNAL METHOD
        :param lr:
        :param epoch:
        :return: None (inplace change)
        """
        for p in self.net.parameters():
            # initalize buf attribute depending on available cuda
            if not hasattr(p, 'buf'):
                if self.use_cuda:
                    p.buf = torch.zeros(p.size()).cuda(self.device_id)
                else:
                    p.buf = torch.zeros(p.size())

        for p in self.net.parameters():
            d_p = p.grad.data
            d_p.add_(self.weight_decay, p.data)
            buf_new = (1 - self.alpha) * p.buf - lr * d_p

            if (epoch % 50) + 1 > 45:
                if self.use_cuda:
                    eps = torch.randn(p.size()).cuda(self.device_id)
                else:
                    eps = torch.randn(p.size())

                buf_new += (2.0 * lr * self.alpha * self.temperature / self.datasize) ** .5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

    def _adjust_learning_rate(self, epoch, batch_idx):
        """
        INTERNAL METHOD, cyclic adjustment of learning rate
        :param epoch:

        :param batch_idx:
        :return:
        """
        rcounter = epoch * self.num_batch + batch_idx
        cos_inner = np.pi * (rcounter % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr_0
        return lr

    def _step(self, epoch):
        """
        INTERNAL METHOD
        single epoch of training
        :param epoch: current epoch to adjust learning rate and update params
        :return: None (inplace changes)
        """
        print('\nEpoch: %d' % epoch)
        self.net.train()  # sets the module in training mode

        # TODO update statistics!
        # train_loss = 0
        # correct = 0
        # total = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(self.device_id), targets.cuda(self.device_id)

            self.net.zero_grad()
            lr = self._adjust_learning_rate(epoch, batch_idx)

            # outputs = self.net(inputs)  # Deprec was relevant in the Resnet with multiclass prediction
            loss = self.criterion(inputs, targets)

            loss.backward()
            self._update_params(lr, epoch)

            # TODO update statistics!
            # train_loss += loss.data.item()
            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += predicted.eq(targets.data).cpu().sum()
            # if batch_idx % 100 == 0:
            #     print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def sample_csgmcmc(self, net, trainloader, dir='/home/tim/PycharmProjects/Thesis/Pytorch/Chains', seed=1):
        """
        Like Hamiltorch .sample_model: sampling model based on net (which is required to have a log_prob)
        Also this assumes, that the net's parameters are already initialized.
        :param net:
        :param trainloader:
        :param dir: path to save checkpoints (default None)  # DEPREC: WARNING state_dicts!
        :param seed:
        :return: list of 1D tensors, where each tensor is flattend net.parameters() also stores the state_dict
        """
        self.net = net
        self.trainloader = trainloader
        self.criterion = net.log_prob
        self.datasize = len(trainloader.dataset)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if self.use_cuda:
            self.net.cuda(self.device_id)
            cudnn.benchmark = True
            cudnn.deterministic = True

        # mt is number of cycles
        print('==> COLLECTING SAMPLES')
        for mt, epoch in enumerate(range(self.epochs)):
            self._step(epoch)
            if (epoch % 50) + 1 > 47:  # save 3 models per cycle
                print('save!')
                if self.use_cuda:
                    self.net.cpu()
                # TODO: check for net.state_dict VS. list of Tensors.
                torch.save(self.net.state_dict(), dir + '/cifar_csghmc_%i.pt' % (mt))
                self.chain.append(flatten(model=self.net))
                if self.use_cuda:
                    net.cuda(self.device_id)

        return self.chain


#
# def test(net, testloader, device_id, criterion, epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             if use_cuda:
#                 inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.data.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += targets.size(0)
#             correct += predicted.eq(targets.data).cpu().sum()
#
#             if batch_idx % 100 == 0:
#                 print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
#                       % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss / len(testloader), correct, total,
#         100. * correct / total))


if __name__ == '__main__':
    from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel
    import torch.distributions as td
    import torch.nn as nn

    no_in = 2
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden_ProbModel(no_in, no_out, bias=True, activation=nn.Identity())
    reg.true_model = reg.vec

    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([1000]))
    y = reg.likelihood(X).sample()

    reg.reset_parameters()
    init_theta = reg.vec

    # sampling example
    batch_size = 10

    from torch.utils.data import TensorDataset, DataLoader

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # a = trainloader.__iter__()
    # next(a)

    csgmcmc = CSG_MCMC(epochs=1000)
    chain = csgmcmc.sample_csgmcmc(
        reg, trainloader,
        dir='/home/tim/PycharmProjects/Thesis/Pytorch/Chains', seed=1)
