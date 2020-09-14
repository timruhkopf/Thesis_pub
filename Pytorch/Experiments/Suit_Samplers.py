import os


def samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch=None):
    # (0) SGNHT --------------------------------------------------------------------

    rooting = lambda run: os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
        os.getcwd() + '/Results/{}/'.format(run)

    run = name + '_SGNHT'
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGNHT(steps=steps, batch_size=batch)

    for config in prelim_configs:
        bnn_unittest.main(
            seperated=False,
            n=n, n_val=n_val,
            model_class=cls, model_param=model_param,
            sampler_name='SGNHT', sampler_param=config)

    # (1) MALA -------------------------------------------------

    run = name + '_MALA'
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_MALA(steps=steps)

    for config in prelim_configs:
        bnn_unittest.main(
            seperated=False,
            n=n, n_val=n_val,
            model_class=cls, model_param=model_param,
            sampler_name='MALA', sampler_param=config)

    # (2) SGLD -------------------------------------------------

    run = name + '_SGLD'
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGLD(steps=steps)

    for config in prelim_configs:
        bnn_unittest.main(
            seperated=False,
            n=n, n_val=n_val,
            model_class=cls, model_param=model_param,
            sampler_name='SGLD', sampler_param=config)

    # (3) RHMC -------------------------------------------------

    run = name + '_RHMC'
    root = rooting(run)
    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_RHMC(steps=steps)

    for config in prelim_configs:
        bnn_unittest.main(
            seperated=False,
            n=n, n_val=n_val,
            model_class=cls, model_param=model_param,
            sampler_name='RHMC', sampler_param=config)

    # (4) SGRLD -------------------------------------------------
    run = name + '_SGRLD'
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGRLD(steps=steps, batch_size=batch)

    for config in prelim_configs:
        bnn_unittest.main(
            seperated=False,
            n=n, n_val=n_val,
            model_class=cls, model_param=model_param,
            sampler_name='SGRLD', sampler_param=config)

    # (5) SGRHMC -------------------------------------------------

    run = name + '_SGRHMC'
    root = rooting(run)

    bnn_unittest = cls_Grid(root)
    prelim_configs = bnn_unittest.grid_exec_SGRHMC(steps=steps, batch_size=batch)

    for config in prelim_configs:
        bnn_unittest.main(
            seperated=False,
            n=n, n_val=n_val,
            model_class=cls, model_param=model_param,
            sampler_name='SGRHMC', sampler_param=config)
