import torch

from hamiltorch.util import flatten, unflatten
from functools import partial

class Model_util:
    @property
    def n_params(self):
        # to make it inheritable (is used in "model's" vec_to_params
        return sum([tensor.nelement() for tensor in self.parameters()])

    @property
    def n_tensors(self):
        return len(list(self.parameters()))  # parsing tensorlist

    @property
    def parameters_list(self):
        """due to the differentiation of surrogates e.g. self.W_ and self.W, with
        the former not being updated, but referencing self.parameters(), this function
        serves as self.parameters on the current state parameters self.W
        """
        # print(self, 'id:', id(self))
        return [self.get_param(name) for name in self.p_names]

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

    def log_prob(self, X, y, vec=None):
        if vec is not None:
            self.vec_to_attrs(vec)  # parsing to attributes

        if hasattr(self, 'update_distributions'):
            # in case of a hierarchical model, the distributions hyperparam are updated,
            # changing the (conditional) distribution
            self.update_distributions()

        return self.my_log_prob(X, y)

    def closure_log_prob(self, X=None, y=None):
        """log_prob factory, to fix X & y for samplers operating on the entire
        dataset.
        :returns None. changes inplace attribute log_prob"""
        # FIXME: ensure, that multiple calls to closure do not append multiple
        # X & y s to the function log_prob, causing the call to fail ( to many arguments)
        print('Setting up "Full Dataset" mode')
        self.log_prob = partial(self.log_prob, X, y)


class Vec_Model:
    @property
    def p_names(self):
        return [p[:-1] for p in self.__dict__['_parameters'].keys()]

    def get_param(self, name):
        return self.__getattribute__(name)

    def vec_to_attrs(self, vec):
        """parsing a 1D tensor according to self.parameters & setting them
        as attributes"""
        tensorlist = unflatten(self, vec)
        for name, tensor in zip(self.p_names, tensorlist):
            self.__setattr__(name, tensor)


class Optim_Model:
    @property
    def p_names(self):
        return list(self.__dict__['_parameters'].keys())

    def get_param(self, name):
        return self.__dict__['_parameters'][name]
