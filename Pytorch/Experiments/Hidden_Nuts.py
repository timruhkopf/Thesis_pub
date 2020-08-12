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
        from Pytorch.Samplers.Hamil import Hamil

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)

        reg_Hidden = Hidden(**model_param)
        reg_Hidden.to(device)

        # generate true model
        X_dist = td.Uniform(-10. * torch.ones(model_param['no_in']),
                            10 * torch.ones(model_param['no_in']))
        X = X_dist.sample(torch.Size([n]))

        reg_Hidden.true_model = reg_Hidden.vec
        y = reg_Hidden.likelihood(X).sample()

        X.to(device)
        y.to(device)

        # sample
        hamil = Hamil(reg_Hidden, X, y, torch.ones_like(reg_Hidden.vec))
        hamil.sample_NUTS(steps)

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        hamil.save(self.pathresults + self.hash + '.model')

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
        model_param={'no_in': 2, 'no_out': 1,
                     'bias': True, 'activation': nn.Identity()},
        steps=100,
        sampler_config={}
    )

    # serious run
    hidden_unittest.main(
        n=1000,
        model_param={'no_in': 2, 'no_out': 1,
                     'bias': True, 'activation': nn.Identity()},
        steps=1000,
        sampler_config={}
    )

print()
