import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from functools import partial
from hamiltorch.util import flatten, unflatten
from Pytorch.Util.plotUtil import triangulate_remove_artifacts


class Model_util:
    flatten = flatten

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

    # LOG-PROB related
    def update_distributions(self):
        return None

    def likelihood(self, X):
        """:returns the conditional distribution of y | X"""
        # TODO update likelihood to become an attribute distribution,
        #  which is updated via self.likelihood.__init__(newloc, scale)
        #  or even use self.likelihood.loc = newloc
        return td.Normal(self.forward(X), scale=torch.tensor(1.))

    def my_log_prob(self, X, y):
        """ DEFAULT, but can be easily modified!
        SG flavour of Log-prob: any batches of X & y can be used
        make sure to pass self.log_prob to the sampler, since self.my_log_prob
        is a convenience mask

        Notice, that self.log_prob has two modes operandi:
        (1) self.log_prob(X,y), which returns the log_prob with current state of
        'parameters'. This is particularly handy with optim based samplers,
        since 'parameters' are literally nn.Parameters and update their state based
        on optim proposals (always up to date)

        (2) self.log_prob(X,y, vec), modus is available iff inherits from VecModel
        (-> and used for vec based samplers such as Hamiltorchs). When vec,
        the vector representation (1D Tensor) of the model is provided, the model's
        surrogate 'parameter' (not nn.Parameter) are updated - and the models state
        under vec is evaluated for X & y

        Irrespective of the mode choice, the self.log_prob operates as a SG-Flavour,
        i.e. is called with ever new X & y batches. However, working with Samplers
        operating on the entire Dataset at every step, the method can be modified
        calling  self.closure_log_prob(X, y) to fix every consequent call to
        self.log_prob()  or self.log_prob(vec) on the provided dataset X, y
        (using functools.partial)"""
        return self.prior_log_prob().sum() + \
               self.likelihood(X).log_prob(y).sum()

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

    def invert_bij(self, name):
        return self.dist[name].transforms[0]._inverse(self.get_param(name))

    def _chain_predict(self, chain, X):
        """

        :param chain: list of 1d Tensors
        :param X:
        :return: dict
        """
        d = dict()
        if isinstance(self, Vec_Model):
            if chain is not None:
                for i, p in enumerate(chain):
                    # TODO parallelize predictions
                    self.vec_to_attrs(p)
                    # self.update_distributions()
                    d.update({str(i):self.forward(X)})

        elif isinstance(self, Optim_Model):
            raise NotImplementedError('plot1d not parsing parameters yet')

        return d

    @torch.no_grad()
    def plot1d(self, X, y, true_model=None, param=None, confidence=None, show=True, **kwargs):
        """1D plot of the (current) model.
        :param param: list of 1d torch tensors. optional vector of parameters,
        representing the model of interest, which is to be plotted."""

        # TODO : each dim in y indicates another unit, i.e. another regression Problem
        if X.shape[1] != 1:
            raise ValueError('X is of improper dimensions')

        # FIXME: in GAM case, Z is required for predictions & X is used for plotting
        last_state = self.vec.detach().clone()  # saving the state before overwriting it
        kwargs.update({'true': self._chain_predict([true_model], X)['0']})
        kwargs.update(self._chain_predict(param, X))

        df = pd.DataFrame({'X': torch.reshape(X, (X.shape[0],)).numpy()})
        for name, var in kwargs.items():
            df[name] = torch.reshape(var, (var.shape[0],)).numpy()
        df = df.melt('X', value_name='y')
        df = df.rename(columns={'variable': 'functions'})

        # plot the functions
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(hspace=0.5)
        sns.scatterplot(
            x=torch.reshape(X, (X.shape[0],)).numpy(),
            y=torch.reshape(y, (y.shape[0],)).numpy(), ax=ax)

        sns.lineplot('X', y='y', hue='functions', alpha=0.5, data=df, ax=ax)

        if confidence is not None:  # plot (optional) confidence bands
            ax.fill_between(**confidence, alpha=0.4, facecolor='lightblue')

        fig.suptitle('Functions of the data')

        # reinstate the old state
        self.vec_to_attrs(last_state)

        plt.plot()


    @torch.no_grad()
    def plot2d(self, X, y, true_model=None, param=None, multi_subplots=False, **kwargs):
        """
        :param true_model: 1D vector representing the true model
        :param X: 2d Tensor
        :param y: 1d Tesor
        :param param: list of 1d Tensors, representing the model (e.g. chain)
        :param multi_subplots: bool. whether each p's prediction based on p in param
        is represented as a surface in a subplot or by points in a single plot
        :param kwargs: dict of tensors, each of size y to be plotted
        """
        # TODO: look at previous 2D plot function and trisurfaces!
        if X.shape[1] != 2:
            raise ValueError('X is of improper dimensions')

        last_state = self.vec  # saving the state before overwriting it
        kwargs.update({'true': self._chain_predict([true_model], X)['0']})
        kwargs.update(self._chain_predict(param, X))

        # for irregular grid data make use of Delauny triangulation & trisurf:
        # https://fabrizioguerrieri.com/blog/2017/9/7/surface-graphs-with-irregular-dataset

        df = pd.DataFrame(torch.reshape(X, (X.shape[0], 2)).numpy(), columns=['X1', 'X2'])
        for name, var in kwargs.items():
            df[name] = torch.reshape(var, (var.shape[0],)).numpy()

        # remove artifacts
        triang = triangulate_remove_artifacts(X[:, 0], X[:, 1], -9.9, 9.9, -9.9, 9.9, plot=False)

        fig = plt.figure()
        plt.title('{}'.format('title here'))
        plt.axis('off')


        if multi_subplots:
            rows = int(torch.ceil(torch.sqrt(torch.tensor(len(kwargs.keys()), dtype=torch.float32))).numpy())
            ax1 = fig.add_subplot(rows, rows, 1, projection='3d')
        else:
            ax1 = fig.add_subplot(111, projection='3d')

        # ground truth model & data
        ax1.plot_trisurf(triang, kwargs['true'].view(X.shape[0]),
                         cmap='jet', alpha=0.4)
        ax1.scatter(X[:, 0], X[:, 1], y,
                    marker='.', s=10, c="black", alpha=0.5)
        ax1.view_init(elev=40, azim=-45)


        if multi_subplots:
            # plotting each function as a seperate surface in a subplot
            for num, (k,v) in enumerate(kwargs.items()):
                if k != 'true_function':
                    num += 2
                    ax = fig.add_subplot(rows, rows, num, projection='3d')
                    # ax2.scatter(X[:, 0], X[:, 1], v,
                    #             marker='.', s=10, c="red", alpha=0.3)
                    ax.plot_trisurf(triang, v.view(X.shape[0]),
                                     cmap='jet', alpha=0.4)
                    ax.view_init(elev=40, azim=-45)

        else:
            # plotting each functions predictions in same plot
            import matplotlib.cm as cm
            colors = cm.rainbow(torch.linspace(0, 1, len(kwargs)).numpy())
            for (k, v), c in zip(kwargs.items(), colors):

                if k != 'true':
                    ax1.scatter(X[:, 0], X[:, 1], v,
                                marker='.', s=7, color=c, alpha=0.3)


        plt.show()


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


if __name__ == '__main__':
    from Pytorch.Layer.Hidden import Hidden
    import torch.nn as nn
    import torch.distributions as td

    # no_in = 1
    # no_out = 1
    #
    # # reg.plot1d(X, y)
    # reg = Hidden(no_in, no_out, bias=True, activation=nn.ReLU())
    # X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    # X = X_dist.sample(torch.Size([100]))
    # y = reg.likelihood(X).sample()
    # reg.true_model = reg.vec
    #
    # param= list()
    # for i in range(5):
    #     reg.reset_parameters()
    #     print(reg.vec)
    #     print(reg.forward(X[0:5]))
    #     param.append(reg.vec)
    #
    # d = reg._chain_predict(param, X)
    #
    # reg.plot1d(X, y, true_model=reg.true_model, param=param)

    no_in = 2
    no_out = 1

    # reg.plot1d(X, y)
    reg = Hidden(no_in, no_out, bias=True, activation=nn.ReLU())
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()

    from copy import deepcopy
    reg.true_model = torch.tensor(deepcopy(reg.vec.numpy()))


    param = list()
    for i in range(10):
        reg.reset_parameters()
        param.append(reg.vec)

    d = reg._chain_predict(param, X)

    reg.plot2d(X, y, true_model=reg.true_model,
               param=[torch.ones_like(reg.vec), *param], multi_subplots=True)
