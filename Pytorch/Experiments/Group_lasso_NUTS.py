import torch
import torch.nn as nn
import torch.distributions as td
import os

from Pytorch.Util.GridUtil import Grid
from Pytorch.Layer.Group_lasso import Group_lasso


class Group_lasso_Nuts(Grid):
    def main(self, n, steps, burn, seperated, bijected=True, model_param={}):
        from Pytorch.Layer.Group_lasso import Group_lasso
        from Pytorch.Samplers.Hamil import Hamil

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        # set up data & model
        # generate true model
        X_dist = td.Uniform(-10. * torch.ones(model_param['no_in']),
                            10 * torch.ones(model_param['no_in']))
        X = X_dist.sample(torch.Size([n]))

        Glasso = Group_lasso(model_param['no_in'], model_param['no_out'], model_param['bias'],
                             activation=nn.Identity(), bijected=bijected)
        Glasso.to(device)
        Glasso.reset_parameters(seperated)
        Glasso.true_model = Glasso.vec
        y = Glasso.likelihood(X).sample()

        Glasso.plot2d(X, y, path=self.pathresults + self.hash + '_2dplot',
                      title='true_model: {}'.format(
                          Glasso.parameters_dict),
                      true_model=Glasso.true_model,
                      multi_subplots=False)

        # send data to device
        X.to(device)
        y.to(device)

        hamil = Hamil(Glasso, X, y, torch.ones_like(Glasso.vec))
        hamil.sample_NUTS(steps, burn)

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        hamil.save(self.pathresults + self.hash + '.model')

        # subsampling the data for the plot
        sub_n = 100
        import random
        rand_choices = random.sample(range(len(X)), sub_n)
        rand_chain = random.sample(range(len(hamil.chain)), 20)

        if model_param['no_in'] == 1:
            Glasso.plot1d(X[rand_choices], y[rand_choices],
                          path=self.pathresults + self.hash + '_1dplot',
                          true_model=Glasso.true_model,
                          param=[hamil.chain[i] for i in rand_chain])
        elif model_param['no_in'] == 2:
            print('currently only one dimensional plots are supported')
            Glasso.plot2d(X, y, path=self.pathresults + self.hash + '_2dplot',
                          true_model=Glasso.true_model,
                          param=[hamil.chain[i] for i in rand_chain],
                          multi_subplots=False)
        else:
            print('no_in is not properly defined for plotting: must be 1 or 2')


if __name__ == '__main__':
    root = os.getcwd() if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Pytorch/Experiments/'

    # unbijected
    Glasso = Group_lasso_Nuts(root)
    Glasso.main(n=1000, steps=10000*5, burn=5000, seperated=True, bijected=False,
                model_param={'no_in': 2, 'no_out': 1, 'bias': False})

    Glasso.main(n=1000, steps=10000*5, burn=5000, seperated=False, bijected=False,
                model_param={'no_in': 2, 'no_out': 1, 'bias': False})

    # bijected
    Glasso.main(n=1000, steps=10000*5, burn=5000, seperated=True, bijected=True,
                model_param={'no_in': 2, 'no_out': 1, 'bias': False})

    Glasso.main(n=1000, steps=10000*5, burn=5000, seperated=False, bijected=True,
                model_param={'no_in': 2, 'no_out': 1, 'bias': False})
