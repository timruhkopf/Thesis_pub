# https://github.com/ruqizhang/csgmcmc

from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random

from Pytorch.Samplers import Samplers

# from models import *
from Pytorch.utils import *

sys.path.append('..')
use_cuda = torch.cuda.is_available()


class CSG_MCMC(Samplers):
    def __init__(self, temperature=1. / 50000, alpha=0.9,
                 weight_decay=5e-4, lr_0=0.5, M=4, epochs=20):
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
        self.epochs = 20

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
            if not hasattr(p, 'buf'):
                p.buf = torch.zeros(p.size()).cuda(self.device_id)
            d_p = p.grad.data
            d_p.add_(self.weight_decay, p.data)
            buf_new = (1 - self.alpha) * p.buf - lr * d_p
            if (epoch % 50) + 1 > 45:
                eps = torch.randn(p.size()).cuda(self.device_id)
                buf_new += (2.0 * lr * self.alpha * self.temperature / datasize) ** .5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

    def _adjust_learning_rate(self, epoch, num_batch, batch_idx):
        # FIXME: num_batch shouldn't be in call should be in train_loader6
        """
        INTERNAL METHOD, cyclic adjustment of learning rate
        :param epoch:
        :param num_batch:
        :param batch_idx:
        :return:
        """
        rcounter = epoch * num_batch + batch_idx
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
        self.net.train()  # TODO figure out what this is doing

        # TODO update statistics!
        # train_loss = 0
        # correct = 0
        # total = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(self.device_id), targets.cuda(self.device_id)
            self.net.zero_grad()
            lr = self._adjust_learning_rate(epoch, batch_idx)
            outputs = self.net(inputs)

            # FIXME! change criterion to be log prob!
            loss = self.criterion(outputs, targets)

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

    def sample_model(self, net, trainloader, criterion, dir=None, device_id=1, seed=1):
        """
        Like Hamiltorch .sample_model: sampling model based on net (which is required to have a log_prob)
        Also this assumes, that the net's parameters are already initialized.
        :param net:
        :param trainloader:
        :param criterion:
        :param dir: path to save checkpoints (default None)  # DEPREC: WARNING state_dicts!
        :param device_id:
        :param seed:
        :return: list of 1D tensors, where each tensor is flattend net.parameters() also stores the state_dict
        """
        self.net = net
        self.trainloader = trainloader
        self.criterion = criterion
        self.device_id = device_id

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.use_cuda = torch.cuda.is_available()

        if use_cuda:
            self.net.cuda(self.device_id)
            cudnn.benchmark = True
            cudnn.deterministic = True


        # mt is number of cycles
        print('==> COLLECTING SAMPLES')
        for mt, epoch in enumerate(self.epochs):
            self._step(epoch)
            if (epoch % 50) + 1 > 47:  # save 3 models per cycle
                print('save!')
                self.net.cpu()
                # TODO: check for net.state_dict VS. list of Tensors.
                torch.save(self.net.state_dict(), dir + '/cifar_csghmc_%i.pt' % (mt))
                self.chain.append(flatten(model=self.net))
                net.cuda(self.device_id)

        return self.chain

    def sample(self, log_prob, init_param, trainloader, criterion, dir=None, device_id=1, seed=1):
        """
        sample based on the log_prob model
        :param log_prob:
        :param trainloader:
        :param criterion:
        :param dir:
        :param device_id:
        :param seed:
        :return:
        """
        pass

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
    # Data / model related
    net = ResNet18()
    batch_size = 64  # input batch size for training
    datasize = 50000
    num_batch = datasize / batch_size + 1
    criterion = nn.CrossEntropyLoss()


    # # Data
    # print('==> Preparing data..')
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # set up data
    data_path = None  # path to datasets location (default None)
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Model
    csg_ld = CSG_MCMC(temperature=1. / 50000, alpha=0.9, weight_decay=5e-4, lr_0=0.5, M=4, epochs=20)
    csg_ld.sample(net, trainloader, criterion, dir=None, seed=1)

    csg_ld = CSG_MCMC(temperature=1. / 50000, alpha=1., weight_decay=5e-4, lr_0=0.5, M=4, epochs=20)
    csg_ld.sample(net, trainloader, criterion, dir=None, seed=1)

    # DEPREC the train functionallity is now in sample
    # for epoch in range(epochs):
    #     sample(epoch)
    #     test(epoch)
    #     if (epoch % 50) + 1 > 47:  # save 3 models per cycle
    #         print('save!')
    #         net.cpu()
    #         torch.save(net.state_dict(), dir + '/cifar_csghmc_%i.pt' % (mt))
    #         mt += 1
    #         net.cuda(device_id)
