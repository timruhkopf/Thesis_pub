"""
Idea of this  script is to show, that despite Pz is detached i.e. has no gradient,
W1, the first hidden layer's weights still get their gradients.

To proof this a single hidden layer NN, where the first layers output is
orthogonalised to the columnspace of Z, another matrices columnspace.
"""
# most important: loss is actually moving and not stuck indicating a relevant change in the
# W1 & W2 (notice that W3 is analytically determined by Least squares.
import torch
import torch.nn as nn
n = 100
in_dim = 1
X = torch.rand(n, in_dim)
Z_dim = 1
Z = torch.rand(n, Z_dim)

W1_true = torch.rand(in_dim, 5, requires_grad=False)
W2_true = torch.rand(5, 1, requires_grad=False)

W3_true = torch.rand(Z_dim, 1, requires_grad=False)

y = (X @ W1_true) @ W2_true +  Z @  W3_true

W1 = torch.nn.Parameter(torch.rand(in_dim, 5, requires_grad=True))
W2 = torch.nn.Parameter(torch.rand(5, 1, requires_grad=True))

for step in range(100):
    Pz_orth = (torch.eye(Z.shape[0]) - Z @ torch.inverse(Z.t() @ Z) @ Z.t()).detach()
    Pz_orth.shape

    # orthogonalised part of BNN without Z added the analytical LS solution of Z
    # Notice BNN is linear here! since no ReLU - and W1 & W2 $W3 are strictly positive - indicating y strictly pos.
    # indicating ReLU is never used. also NO BIAS! again it merely is tried to figure out how to use Pz without grad
    y_pred = Pz_orth @ (X @ W1) @ W2 + (Z @ torch.inverse(Z.t() @ Z) @ Z.t() @ Z).detach()  # analytical solution  for Z

    # y_pred = (X @ W1) @ W2
    loss = ((y_pred - y) ** 2).sum()

    print('-----', loss)  # < --------------------- MOST IMPORTANT
    loss.backward()

    # check that all but Pz_orth have gradients
    W1.grad
    assert (Pz_orth.grad is None)  # < --------------------- MOST IMPORTANT
    W2.grad

    old_W1 = W1.clone().detach()
    old_W2 = W2.clone().detach()

    with torch.no_grad():
        W1 -= 0.01 * W1.grad
        W2 -= 0.01 * W2.grad


        # print(W1.grad)
        W1.grad = None
        W2.grad = None


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.scatter(X[:, 0], y[:,0])
plt.scatter(X[:, 0], y_pred[:, 0].detach().numpy())
plt.show()
