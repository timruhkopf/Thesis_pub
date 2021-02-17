import torch
import torch.distributions as td
from src.Util.Util_Plots import Util_plots
import numpy as np
from math import prod


class Util_Model(Util_plots):

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

    def log_prob(self, X, y):
        if hasattr(self, 'update_distributions'):
            # in case of a hierarchical model, the distributions hyperparam are updated,
            # changing the (conditional) distribution
            self.update_distributions()
        return self.my_log_prob(X, y)

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
            print(chain[0], '\n', chain[-1])
            raise RuntimeError('first and last entry contain nan')

        if len(chain) > 2 and all(torch.all(a == b) for a, b in zip(chain[0].values(), chain[-2].values())):
            print(chain[0], '\n', chain[-2])
            # FIXME: as it appears, many chains actually progress usefully,
            #  but 1st and last are same - this seems to be a technical issue yet to be identified!
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
            print(chain[0], '\n', chain[-1])
            raise RuntimeError('first and last entry contain nan')

        if len(chain) > 2 and all(all(a.view(prod(a.shape)) == b.view(prod(a.shape))) for a, b in
                                  zip(chain[0].values(), chain[-2].values())):
            for a, b, c in zip(chain[0].values(), chain[-2].values(), chain[-1].values()):
                print(a.view(prod(a.shape)), '\n', b.view(prod(a.shape)), '\n', c.view(prod(a.shape)), '\n')
            raise RuntimeError('first and last entry are the same')

