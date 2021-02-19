import torch
import torch.distributions as td
from copy import deepcopy
from tqdm import  tqdm


class Optimizer:
    def __init__(self, model, trainloader):
        """
        Stochastic Gradient Descent
        :param model:
        :param lr: float learning rate
        """
        self.model = model
        self.chain = list()
        self.losses = list()
        self.trainloader = trainloader

        # todo ensure lr is torch tensor float

    def sample(self, loss_closure, steps, lr):
        for step in tqdm(range(steps)):
            X, y = next(self.trainloader.__iter__())
            y_pred = self.model.forward(X)
            # TODO choose loss !
            loss = loss_closure(X, y)
            loss.backward()
            self.losses.append(loss)

            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if step % 100 == 0:
                        print(p.grad)
                    p -= lr * (p.grad + td.Normal(torch.zeros_like(p), 1.).sample())
                    p.grad = None

                self.chain.append(deepcopy(self.model.state_dict()))

# chain = []
# losses = []
# print('Pz_ortho', h.Pz_orth @ X)
# for step in range(100):
#     y_pred = h.forward(X, Z)
#     MSE = ((y_pred - y) ** 2).sum()
#     loss = h.log_prob(X, Z, y)
#     loss.backward()
#     losses.append(loss)
#     print('tau\'s gradient: ', h.gam.tau.grad)
#
#     with torch.no_grad():
#
#         # print('BNN')
#         # for name, p in h.bnn.named_parameters():
#         #     print(name, torch.flatten(p.data), 'grad: ', torch.flatten(p.grad),
#         #           'update: ', torch.flatten(torch.flatten(0.001 * p.grad)))
#         #
#         # print('GAM')
#         for name, p in h.named_parameters():
#             # print(p.grad)
#
#             try:
#                 if name == 'gam.tau':
#                     # update tau more sensibly in smaller steps
#                     print('gam.tau (on R): ', p.data, 'gam.tau_bij (on R+):', h.gam.tau_bij, '\n',
#                           'grad:', p.grad, 'update: ', 0.0001 * p.grad, '\n')
#
#                     print('gam.W.grad:', torch.flatten(h.gam.W.grad), '\n',
#                           'gam.W.update: ', torch.flatten(0.001 * h.gam.W.grad), '\n\n')
#                     p -= 0.0001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()
#                 elif name == 'gam.W':
#                     # default update
#                     print('gam.W.grad:', torch.flatten(h.gam.W.grad), '\n',
#                           'gam.W.update: ', torch.flatten(0.001 * h.gam.W.grad), '\n\n')
#                     p -= 0.001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()
#
#                 else:
#                     print(name, ' grad:', torch.flatten(p.grad), '\n',
#                           name, '.update: ', torch.flatten(0.001 * p.grad), '\n\n')
#                     p -= 0.001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()
#
#
#
#             except:
#                 # tau seems to not be include in prior - since it is not in the loss and has no gradient as
#                 # consequence!
#                 print(name, ' failed to be updated')
#                 continue
#
#         for p in h.parameters():
#             chain.append(deepcopy(h.state_dict()))
#             p.grad = None
