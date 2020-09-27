import torch
import torch.distributions as td
from Pytorch.Util.Util_Plots import Util_plots

from math import prod


class Util_Model(Util_plots):

    @property
    def p_names(self):
        return list(self.__dict__['_parameters'].keys())

    def get_param(self, name):
        return self.__dict__['_parameters'][name]

    @property
    def parameters_dict(self):
        # print(self)
        return {name: self.get_param(name) for name in self.p_names}

    @property
    def vec(self):
        """vectorize provides the view of all of the object's parameters in form
        of a single vector. essentially it is hamiltorch.util.flatten, but without
        dependence to the nn.Parameters. instead it works on the """
        return torch.cat([self.get_param(name).view(
            self.get_param(name).nelement())
            for name in self.p_names])

    # LOG-PROB related
    def update_distributions(self):
        raise NotImplementedError('update_distribution function must be specified')

    def likelihood(self, X):
        """:returns the conditional distribution of y | X"""
        # TODO update likelihood to become an attribute distribution,
        #  which is updated via self.likelihood.__init__(newloc, scale)
        #  or even use self.likelihood.loc = newloc
        return td.Normal(self.forward(X), scale=torch.tensor(1.))

    def my_log_prob(self, X, y):
        """Default "user-specified" log_prob function, that assumes
         class hast prior_log_prob() and likelihood(mu=f(X)) function.
         my_log_prob function is subjected to self.log_prob function, which in
         turn is subjected to a sampler"""
        return self.prior_log_prob() + \
               self.likelihood(X).log_prob(y).sum()

    def log_prob(self, X, y, vec=None):
        if hasattr(self, 'update_distributions'):
            # in case of a hierarchical model, the distributions hyperparam are updated,
            # changing the (conditional) distribution
            self.update_distributions()
        return self.my_log_prob(X, y)

    def invert_bij(self, name):
        return self.dist[name].transforms[0]._inverse(self.get_param(name))

    def _chain_predict(self, chain, *args):
        """

        :param chain: list of 1d Tensors
        :param X:
        :return: dict
        """
        if not isinstance(chain, list):
            raise ValueError('chain must be list of state_dicts')

        d = dict()
        if chain is not None:
            for i, p in enumerate(chain):
                # TODO parallelize predictions
                self.load_state_dict(p)
                # self.update_distributions()
                d.update({str(i): self.forward(*args)})

        return d

    @staticmethod
    def check_chain(chain):
        """method, that sampler interfaces call to ensure chain did actually
        do something meaningfull. This is the default method for non nn.Sequential based models.
        Overwrite this method for any nn.Sequential based method"""
        if len(chain) == 1:
            print(chain)
            raise RuntimeError('The chain did not progress beyond first step')

        if any([torch.any(torch.isnan(v)) if v.nelement() != 1 else torch.isnan(v)
                for chain in (chain[-1].values(), chain[0].values())
                for v in chain]):
            print(chain[0], chain[-1])
            raise RuntimeError('first and last entry contain nan')

        if all(all(a == b) for a, b in zip(chain[0].values(), chain[-1].values())):
            print(chain[0], chain[-1])
            raise RuntimeError('first and last entry are the same')

        print('succeeded')

    @staticmethod
    def check_chain_seq(chain):
        if len(chain) == 1:
            print(chain)
            raise RuntimeError('The chain did not progress beyond first step')

        if any([torch.any(torch.isnan(v)) if v.nelement() != 1 else torch.isnan(v)
                for chain in (chain[-1].values(), chain[0].values())
                for v in chain]):
            print(chain[0], chain[-1])
            raise RuntimeError('first and last entry contain nan')

        if all(all(a.view(prod(a.shape)) == b.view(prod(a.shape))) for a, b in
               zip(chain[0].values(), chain[-1].values())):
            print(chain[0], '\n', chain[-1])
            raise RuntimeError('first and last entry are the same')
