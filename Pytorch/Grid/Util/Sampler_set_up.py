import numpy as np
from copy import deepcopy


class Sampler_set_up:
    def set_up_sampler(self, sampler_name, sampler_param):
        from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
        from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD

        # (SAMPLER) Select sampler, set up  and sample -------------------------
        # random init state
        self.model.reset_parameters()

        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('Tkagg')
        # plt.ion()

        init = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.model.true_model)
        self.model.plot(*self.data_plot, chain=[init], path=self.basename + '_initmodel')
        self.model.load_state_dict(init)

        # searching for a stable init for sgnht and all RM sampler
        # if sampler_name in ['SGNHT', 'RHMC', 'SGRLD', 'SGRHMC']:
        #     self.sampler_init_run(trainloader)

        if sampler_name in ['SGNHT', 'SGLD', 'MALA']:  # geoopt based models
            Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                       'MALA': MALA,  # step_size
                       'SGLD': SGLD  # step_size
                       }[sampler_name]
            if 'n_samples' in sampler_param.keys():
                sampler_param['num_steps'] = sampler_param['n_samples']
                sampler_param.pop('n_samples')
            self.sampler = Sampler(self.model, self.trainloader, **sampler_param)
            self.sampler.sample()


        elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:
            try:
                n_samples = sampler_param.pop('n_samples')
                burn_in = sampler_param.pop('burn_in')
            except Exception as error:
                print(error)

            Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                       'SGRLD': myRSGLD,  # epsilon
                       'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                       }[sampler_name]
            self.sampler = Sampler(self.model, **sampler_param)
            self.sampler.sample(self.trainloader, burn_in, n_samples)

        else:
            raise ValueError('sampler_name was not correctly specified')

        # save the sampler instance (containing the model instance as attribute)
        # NOTICE: to recover the info, both the model class and the
        # sampler class must be imported, to make them accessible
        self.sampler.save(self.basename)

        self.sampler.traceplots(path=self.basename + '_traces_baseline.pdf')
        self.sampler.traceplots(path=self.basename + '_traces.pdf', baseline=False)
        self.sampler.acf_plots(nlags=500, path=self.basename + '_acf.pdf')

        self.sampler.ess(nlags=200)
        print(self.sampler.ess_min)

    # generator functions ------------------------------------------------------
    def grid_exec_MALA(self, steps, epsilons=np.arange(0.0001, 0.03, 0.003)):
        for epsilon in epsilons:
            yield dict(epsilon=epsilon, num_steps=steps, pretrain=False,
                       tune=False, burn_in=int(steps * 0.10), num_chains=1)

    def grid_exec_SGLD(self, steps, epsilons=np.arange(0.0001, 0.03, 0.003)):
        return self.grid_exec_MALA(steps, epsilons)

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
                           burn_in=int(steps * 0.10))

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
