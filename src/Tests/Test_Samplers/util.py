import torch


def chain_mat(chain):
    vecs = [torch.cat([p.reshape(p.nelement()) for p in state.values()], axis=0) for state in chain]
    return torch.stack(vecs, axis=0)


def posterior_mean(chain):
    chain_matrix = chain_mat(chain)
    return chain_matrix.mean(dim=0)


def plot_sampler_path(sampler, model, steps, loss=None, skip=10, error_margin=0.09):
    """
    debugtool: plot the trajectory taken of a sampler in a 2d parameter case
    cleans the chain of any nan or infinit values!
    :param sampler: sampler contains chain attribute
    :param model: contains init & true model.
    :param steps: int. number of steps of the sampler that are to be plotted
    :param loss: ndarray 1d - used to annotate
    :param skip: every skip'th loss is annotated.
    :param error_margin =
    """
    import matplotlib.pyplot as plt
    import numpy as np
    chain = chain_mat(sampler.chain).numpy()
    chain = chain[~(np.isnan(chain).any(axis=1) + np.isinf(chain).any(axis=1))][:steps]
    last = torch.tensor(chain)[-1].numpy()
    true = chain_mat([model.true_model]).numpy()
    init = chain_mat([model.init_model]).numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax.set(  # xlim=(-100, 100), ylim=(-100, 100),
        xlabel='ß1', ylabel='ß0',
        title='Regession; chain of parameters')

    x, y = chain[:, 0], chain[:, 1]

    # draw the steps as vector, also plot true, init and last value of the chain
    ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1)
    ax.plot(true[:, 0], true[:, 1], marker='x')
    ax.plot(init[:, 0], init[:, 1], marker='o')
    ax.plot(last[0], last[1], marker='v')

    # Consider: gradient quiver like : https://stackoverflow.com/questions/12849350/quiver-vectors-superimposed-on-a-plot

    # annotate the plot with the loss
    if loss is not None:
        texts = []
        for i, (xi, yi, s) in enumerate(zip(x[::skip], y[::skip], loss[:steps:skip])):
            texts.append(plt.text(xi, yi, '{}:{:10.3f}'.format(i * skip, s), color='red'))

    # draw a "convergence circle" ; if final value lies in this circle, the result is accepted
    if error_margin is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.sqrt(error_margin)

        x1 = r * np.cos(theta) + true[:, 0]
        x2 = r * np.sin(theta) + true[:, 1]
        ax.plot(x1, x2)

    plt.show()
