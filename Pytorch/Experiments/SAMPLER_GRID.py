import numpy as np

from inspect import getfullargspec


class SAMPLER_GRID:
    """A class intended to simplify the Experiments, as this functionallity is shared
    across all Experiments"""

    def sampler_init_run(self, trainloader):
        from Pytorch.Samplers.LudwigWinkler import SGLD
        from copy import deepcopy
        import matplotlib.pyplot as plt
        self.model.init_model = deepcopy(self.model.state_dict())

        print('searching for a suitable initilisation')
        sgld = SGLD(self.model, trainloader, epsilon=0.001, num_steps=1000,
                    burn_in=1000, pretrain=False, tune=False, num_chains=1)
        try:
            sgld.sample()

        except:
            print('init run failed')

        data = next(trainloader.__iter__())
        if len(data) == 3:
            X, Z, y = data
            self.model.plot(X, Z, y, path=self.basename + '_init_run', title='')

        else:
            X, y = data
            self.model.plot(X, y, path=self.basename + '_init_run', title='')

        plt.close('all')

    def set_up_sampler(self, model, sampler_name, sampler_param):
        from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
        from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
        from torch.utils.data import TensorDataset, DataLoader

        # (SAMPLER) Select sampler, set up  and sample -------------------------
        # random init state
        if 'mode' in getfullargspec(self.model.reset_parameters).args:
            model.reset_parameters(mode='U-MVN')  # GAM model has multiple ways for init
        else:
            self.model.reset_parameters()

        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('Tkagg')
        # plt.ion()
        from copy import deepcopy
        init = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.model.true_model)
        self.model.plot(*self.data_val, chain=[init], path=self.basename + '_initmodel')
        self.model.load_state_dict(init)

        try:
            batch_size = sampler_param.pop('batch_size')
        except:
            batch_size = self.data[0].shape[0]

        # send data to device
        for tensor in self.data:
            tensor.to(self.device)

        trainset = TensorDataset(*self.data)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # searching for a stable init for sgnht and all RM sampler
        # if sampler_name in ['SGNHT', 'RHMC', 'SGRLD', 'SGRHMC']:
        #     self.sampler_init_run(trainloader)

        if sampler_name in ['SGNHT', 'SGLD', 'MALA']:  # geoopt based models
            Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                       'MALA': MALA,  # step_size
                       'SGLD': SGLD  # step_size
                       }[sampler_name]
            self.sampler = Sampler(model, trainloader, **sampler_param)
            self.sampler.sample()

        elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:
            n_samples = sampler_param.pop('n_samples')
            burn_in = sampler_param.pop('burn_in')

            Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                       'SGRLD': myRSGLD,  # epsilon
                       'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                       }[sampler_name]
            self.sampler = Sampler(model, **sampler_param)
            self.sampler.sample(trainloader, burn_in, n_samples)

        else:
            raise ValueError('sampler_name was not correctly specified')

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        self.sampler.save(self.basename)

        # STATIC METHDODS: ---------------------------------------
        # TODO: Check each Grid to be runnable configs & the defaults of param included in grid defaults
        # ludwig based

    def grid_exec_MALA(self, steps, epsilons=np.arange(0.0001, 0.03, 0.003)):
        for epsilon in epsilons:
            yield dict(epsilon=epsilon, num_steps=steps, pretrain=False,
                       tune=False, burn_in=int(steps * 0.10), num_chains=1)

    def grid_exec_SGLD(self, steps, batch_size, epsilons=np.arange(0.0001, 0.03, 0.003)):
        return self.grid_exec_MALA(steps, epsilons, batch_size)

    def grid_exec_SGNHT(self, steps, batch_size,
                        epsilons=np.arange(0.0001, 0.03, 0.003),
                        Ls=[1, 2, 3, 5, 10]):
        """

        :param steps:
        :param step_sizes:
        :param Ls :hmc_traj_lengths:
        :return: generator of tuples: sampler_name, model_param, sampler_param
        """
        for epsilon in epsilons:
            for L in Ls:
                yield dict(epsilon=epsilon, L=L, num_steps=steps, pretrain=False,
                           tune=False, burn_in=int(steps * 0.10), num_chains=1,
                           batch_size=batch_size)

        # geoopt based

    def grid_exec_RHMC(self, steps,
                       epsilons=np.arange(0.0001, 0.03, 0.003),
                       Ls=[1, 2, 3, 5, 10]):
        """

        :param steps: number of sampler steps
        :param epsilons: step_size
        :param n_stepss: leapfrog steps
        :return: generator of tuples:  model_param, sampler_param
        """

        for epsilon in epsilons:
            for L in Ls:
                yield dict(epsilon=epsilon, L=L, n_samples=steps,
                           burn_in=int(steps * 0.10)
                           )

    def grid_exec_SGRHMC(self, steps, batch_size, epsilons=np.arange(0.0001, 0.03, 0.003),
                         Ls=[1, 2, 3, 5, 7, 10], alphas=np.arange(0., 0.99, 0.20)):
        for epsilon in epsilons:
            for L in Ls:
                for alpha in alphas:
                    yield dict(epsilon=epsilon, L=L, alpha=alpha,
                               n_samples=steps, batch_size=batch_size,
                               burn_in=int(steps * 0.10))

    def grid_exec_SGRLD(self, steps, batch_size, epsilons=np.arange(0.0001, 0.03, 0.003)):
        for epsilon in epsilons:
            yield dict(epsilon=epsilon, n_samples=steps, burn_in=int(steps * 0.10), batch_size=batch_size)
