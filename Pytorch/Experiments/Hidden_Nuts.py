import torch
import torch.nn as nn
import torch.distributions as td
import os

from Pytorch.Util.GridUtil import Grid


class Hidden_Nuts(Grid):
    def main(self, n, model_param, steps, sampler_config={}):
        """
        Unittest to both Hidden model and Hamil sampler NUTS
        1d Data.
        :param n: number of data points to generate (X)
        :param model_param: dict: all details to instantiate
        :param steps: number of steps taken by sampler
        :param sampler_config: currently not supported
        :return: None: writes to module's path + '/results folder under the self.hash
        signiture.
        """
        from Pytorch.Layer.Hidden import Hidden
        from Pytorch.Samplers.TRASH.Hamil import Hamil

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        reg_Hidden = Hidden(**model_param)
        reg_Hidden.to(device)

        # generate true model
        X_dist = td.Uniform(-10. * torch.ones(model_param['no_in']),
                            10 * torch.ones(model_param['no_in']))
        X = X_dist.sample(torch.Size([n]))

        reg_Hidden.true_model = reg_Hidden.vec.clone()
        y = reg_Hidden.likelihood(X).sample()

        X.to(device)
        y.to(device)

        # sample
        # TODO REMEMBER TO DO A SUFFICIENT AMOUNT OF BURNIN
        hamil = Hamil(reg_Hidden, X, y, torch.ones_like(reg_Hidden.vec))
        hamil.sample_NUTS(steps, burn=5000)

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        hamil.save(self.pathresults + self.hash + '.model')

        # subsampling the data for the plot
        sub_n = 100
        import random
        rand_choices = random.sample(range(len(X)), sub_n)
        rand_chain = random.sample(range(len(hamil.chain)), 20)

        if model_param['no_in']==1:
            reg_Hidden.plot1d(X[rand_choices], y[rand_choices],
                              path=self.pathresults + self.hash + '_1dplot',
                              true_model=reg_Hidden.true_model,
                              param=[hamil.chain[i] for i in rand_chain ])
        elif model_param['no_in']==2:
            print('currently only one dimensional plots are supported')

        # TODO Autocorrelation time & thinning
        # TODO write out 1d / 2d plots


if __name__ == '__main__':
    hidden_unittest = Hidden_Nuts(
        root= \
            os.getcwd() if os.path.isdir(os.getcwd()) else \
                os.getcwd() + '/Pytorch/Experiments/')  # + '/Pytorch/Experiments/')

    # test run
    hidden_unittest.main(
        n=100,
        model_param={'no_in': 1, 'no_out': 1,
                     'bias': True, 'activation': nn.ReLU()},
        steps=10000,
        sampler_config={}
    )

    # serious run
    hidden_unittest.main(
        n=1000,
        model_param={'no_in': 1, 'no_out': 1,
                     'bias': True, 'activation': nn.Identity()},
        steps=10000,
        sampler_config={}
    )

print()
