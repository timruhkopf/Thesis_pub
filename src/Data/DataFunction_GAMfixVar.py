import matplotlib
import torch
import torch.distributions as td

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from numpy import pi

from src.Layer import GAM_fix_var
from src.Util.Util_bspline import get_design

order, degree, no_basis = 1, 2, 5
gam = GAM_fix_var(order=order, no_basis=no_basis)

# print(gam.tau.dist.transforms[0]._inverse(gam.tau))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 1000

x1 = td.Uniform(0., 2. * pi).sample([n, 1])
x2 = td.Uniform(0., 2. * pi).sample([n, 1])

# y = torch.sin(x1) + x1 * torch.cos(x2)
y = torch.sin(x1)
# y = x1 * torch.cos(x2)


ax.scatter(x1.view((n,)).numpy(), x2.view((n,)).numpy(), y.view((n,)).numpy())

x1 = torch.linspace(0., 2. * pi, 1000)
x2 = x1
Z = torch.tensor(
    get_design(x1.view((n,)).numpy(), degree=2, no_basis=no_basis), dtype=torch.float32,
    requires_grad=False)
ax.plot(x1.view((n,)).numpy(),
        torch.zeros_like(x1).view((n,)).numpy(),
        gam.forward(Z).detach().view((n,)).numpy(),
        label='parametric curve')
plt.show()

# estimate model
from src.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

# matplotlib.use('Agg')  # 'TkAgg' for explicit plotting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.get_device_name(0)

X = x1
Z.to(device)
y.to(device)
import torch.nn as nn

sampler_name = ['SGNHT', 'SGLD', 'MALA', 'RHMC', 'SGRLD', 'SGRHMC'][3]
model = gam

sg_batch = 100
for rep in range(3):
    for L in [2, 1]:
        for eps in np.arange(0.005, 0.001, -0.001):
            model.reset_parameters(tau=torch.tensor([0.0001]))  # initialization of variance = 1
            print('init avg MSE:', nn.MSELoss()(y, model.forward(Z)))
            name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))
            print(name)
            sampler_param = dict(
                epsilon=eps,
                num_steps=10000, burn_in=100,
                pretrain=False, tune=False, num_chains=1)

            if sampler_name in ['SGNHT', 'RHMC', 'SGRHMC']:
                sampler_param.update(dict(L=L))

            if sampler_name == 'SGRHMC':
                sampler_param.update(dict(alpha=0.2))

            if 'SG' in sampler_name:
                batch_size = sg_batch
            else:
                batch_size = X.shape[0]

            trainset = TensorDataset(Z, y)

            # Setting up the sampler & sampling
            if sampler_name in ['SGNHT', 'SGLD', 'MALA']:  # geoopt based models
                trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
                Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                           'MALA': MALA,  # step_size
                           'SGLD': SGLD  # step_size
                           }[sampler_name]
                sampler = Sampler(model, trainloader, **sampler_param)
                try:
                    sampler.sample()
                    print('avg MSE:', nn.MSELoss()(y, model.forward(X)))
                    # sampler.save('/home/tim/PycharmProjects/Thesis/Experiments/Results/Results_GAM/')

                    # Visualize the resulting estimation -------------------------
                    import matplotlib

                    matplotlib.use('TkAgg')
                    sampler.model.plot(X[:100], y[:100], sampler.chain[-30:])
                    sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30))

                    sampler.model.plot(X[:100], y[:100], sampler.chain[-30:])
                    sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), )
                    sampler.traceplots(baseline=True)
                    matplotlib.pyplot.close('all')
                except Exception as error:
                    print(name, 'failed')
                    sampler.model.plot(X[:100], y[:100])
                    print(error)

            elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:
                n_samples = sampler_param.pop('num_steps')
                burn_in = sampler_param.pop('burn_in')
                sampler_param.pop('pretrain')
                sampler_param.pop('tune')
                sampler_param.pop('num_chains')

                trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

                Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                           'SGRLD': myRSGLD,  # epsilon
                           'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                           }[sampler_name]
                sampler = Sampler(model, **sampler_param)
                try:
                    sampler.sample(trainloader, burn_in, n_samples)
                    print('avg MSE:', nn.MSELoss()(y, model.forward(Z)))
                    # sampler.save('/home/tim/PycharmProjects/Thesis/Experiments/Results/Results_GAM/{}'.format(name))
                    sampler.traceplots(baseline=True)

                except Exception as e:
                    print(None)
                    print(e)
                #
                #     # Visualize the resulting estimation -------------------------
                #
                #     sampler.model.plot(X[:100], y[:100], sampler.chain[-30:], path=path + name)
                #     sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)
                #     matplotlib.pyplot.close('all')
                # except Exception as error:
                #     print(name, 'failed')
                #     sampler.model.plot(X[:100], y[:100], path=path + 'failed_' + name)
                #     print(error)
