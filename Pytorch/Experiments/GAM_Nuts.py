import torch
import torch.nn as nn
import torch.distributions as td
import os

from Pytorch.Util.GridUtil import Grid


class GAM_Nuts(Grid):
    def main(self, n, steps, bijected=True, model_param={}):
        from Tensorflow.Effects.bspline import get_design
        from Pytorch.Models.GAM import GAM
        from Pytorch.Samplers.Hamil import Hamil

        # set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        # set up data & model
        no_basis = 20
        X_dist = td.Uniform(-10., 10)
        X = X_dist.sample(torch.Size([n]))
        Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis),
                         dtype=torch.float32, requires_grad=False)

        gam = GAM(order=1, bijected=bijected, **model_param)
        gam.to(device)
        gam.reset_parameters()
        gam.true_model = gam.vec
        y = gam.likelihood(Z).sample()

        # send data to device
        Z.to(device)
        y.to(device)

        # sample
        hamil = Hamil(gam, Z, y, torch.ones_like(gam.vec))
        hamil.sample_NUTS(steps)

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        hamil.save(self.pathresults + self.hash + '.model')

        sub_n = 100
        import random
        rand_choices = random.sample(range(len(X)), sub_n)
        rand_chain = random.sample(range(len(hamil.chain)), 20)

        # IN GAM there is always only one dimension!
        gam.plot1d(X[rand_choices], y[rand_choices],
                   path=self.pathresults + self.hash + '_1dplot',
                   true_model=gam.true_model,
                   param=[hamil.chain[i] for i in rand_chain])

        # TODO Autocorrelation time & thinning
        # TODO write out 1d / 2d plots


if __name__ == '__main__':
    gam_unittest = GAM_Nuts(root= \
                                os.getcwd() if os.path.isdir(os.getcwd()) else \
                                    os.getcwd() + '/Pytorch/Experiments')

    # (1) unbijected
    gam_unittest.main(n=1000, steps=10000, bijected=False, model_param={
        'no_basis': 20
    })

    # (2) bijected # FIXME: fails, since prior model is not implemented!
    gam_unittest.main(n=1000, steps=10000, bijected=True, model_param={
        'no_basis': 20
    })

    # (3) penK

    # RECONSTRUCTION FROM MODEL FILES: ----------------------------------------------
    # from Pytorch.Models.GAM import GAM
    # from Pytorch.Samplers.Hamil import Hamil
    #
    # # filter a dir for .model files
    # models = [m for m in os.listdir('results/') if m.endswith('.model')]
    #
    # loaded_hamil = torch.load('results/' +models[0])
    # loaded_hamil.chain

print()
