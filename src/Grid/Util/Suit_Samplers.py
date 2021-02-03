import os
from subprocess import check_output
from copy import deepcopy


def samplers(cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons, Ls, seperated=True, repeated=1, name=''):
    git = check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    samp_name = cls.__name__

    base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
    rooting = lambda run: base + '/Experiment/Result_{}/{}/'.format(git, run)

    path = base + '/Experiment/'
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except FileExistsError:
            print('file result existed, still continuing')

    path = base + '/Experiment/Result_{}/'.format(git)
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except FileExistsError:
            print('file result existed, still continuing')
    #
    # (3) RHMC -------------------------------------------------

    run = samp_name + '_RHMC' + '_' + name
    root = rooting(run)
    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_RHMC(steps=steps, epsilons=epsilons, Ls=Ls)

    for config in prelim_configs:
        for i in range(repeated):
            config_copy = {key: deepcopy(value) for key, value in config.items()}
            bnn_unittest.main(
                seperated=seperated,
                n=n, n_val=n_val,
                model_class=cls, model_param=model_param,
                sampler_name='RHMC', sampler_param=config_copy)

    # (4) SGRLD -------------------------------------------------
    run = samp_name + '_SGRLD' + '_' + name
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGRLD(steps=steps, batch_size=batch, epsilons=epsilons)

    for config in prelim_configs:
        for i in range(repeated):
            config_copy = {key: deepcopy(value) for key, value in config.items()}
            bnn_unittest.main(
                seperated=seperated,
                n=n, n_val=n_val,
                model_class=cls, model_param=model_param,
                sampler_name='SGRLD', sampler_param=config_copy)

    # (0) SGNHT --------------------------------------------------------------------

    run = samp_name + '_SGNHT' + '_' + name
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGNHT(epsilons=epsilons, Ls=Ls, steps=steps, batch_size=batch)

    for config in prelim_configs:
        for i in range(repeated):
            config_copy = {key: deepcopy(value) for key, value in config.items()}
            bnn_unittest.main(
                seperated=seperated,
                n=n, n_val=n_val,
                model_class=cls, model_param=model_param,
                sampler_name='SGNHT', sampler_param=config_copy)

    # (2) SGLD -------------------------------------------------

    # run = samp_name + '_SGLD' + '_' + name
    # root = rooting(run)
    #
    # bnn_unittest = cls_Grid(root)
    # prelim_configs = bnn_unittest.grid_exec_SGLD(steps=steps, epsilons=epsilons)
    #
    # for config in prelim_configs:
    #     for i in range(repeated):
    #         config_copy = {key: deepcopy(value) for key, value in config.items()}
    #         bnn_unittest.main(
    #             seperated=seperated,
    #             n=n, n_val=n_val,
    #             model_class=cls, model_param=model_param,
    #             sampler_name='SGLD', sampler_param=config_copy)

    # (5) SGRHMC -------------------------------------------------

    run = samp_name + '_SGRHMC' + '_' + name
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGRHMC(steps=steps, batch_size=batch, epsilons=epsilons, Ls=Ls)

    for config in prelim_configs:
        for i in range(repeated):
            config_copy = {key: deepcopy(value) for key, value in config.items()}
            bnn_unittest.main(
                seperated=seperated,
                n=n, n_val=n_val,
                model_class=cls, model_param=model_param,
                sampler_name='SGRHMC', sampler_param=config_copy)

    # (1) MALA -------------------------------------------------
    #
    # run = samp_name + '_MALA' + '_' + name
    # root = rooting(run)
    #
    # bnn_unittest = cls_Grid(root)
    # prelim_configs = bnn_unittest.grid_exec_MALA(steps=steps, epsilons=epsilons)
    #
    # for config in prelim_configs:
    #     for i in range(repeated):
    #         config_copy = {key: deepcopy(value) for key, value in config.items()}
    #         bnn_unittest.main(
    #             seperated=seperated,
    #             n=n, n_val=n_val,
    #             model_class=cls, model_param=model_param,
    #             sampler_name='MALA', sampler_param=config_copy)
