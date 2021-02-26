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
    pmean = posterior_mean(sampler.chain[-200:]).numpy()
    chain = chain_mat(sampler.chain).numpy()
    chain = chain[~(np.isnan(chain).any(axis=1) + np.isinf(chain).any(axis=1))][:steps]
    last = torch.tensor(chain)[-1].numpy()
    true = chain_mat([model.true_model]).numpy()
    init = chain_mat([model.init_model]).numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    title = 'Regession; chain of parameters.{}'.format(' encountered nan after {}'.format(len(chain)) \
                                                           if len(chain) != len(sampler.chain) else '')
    ax.set(  # xlim=(-100, 100), ylim=(-100, 100),
        xlabel='ß1', ylabel='ß0',
        title=title)

    x, y = chain[:, 0], chain[:, 1]

    # draw the steps as vector, also plot true, init and last value of the chain & posterior mean on last 200 steps
    # of chain
    ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1)
    ax.plot(true[:, 0], true[:, 1], marker='x')
    ax.plot(init[:, 0], init[:, 1], marker='o')
    ax.plot(last[0], last[1], marker='v')
    ax.plot(pmean[0], pmean[1], marker='.')
    ax.plot(model.LS[1], model.LS[0], marker='.')

    # Consider: gradient quiver like : https://stackoverflow.com/questions/12849350/quiver-vectors-superimposed-on-a-plot

    # annotate the plot with the loss
    if loss is not None:
        texts = []
        for i, (xi, yi, s) in enumerate(zip(x[::skip], y[::skip], loss[:steps:skip])):
            texts.append(plt.text(xi, yi, '{}:{:10.3f}'.format(i * skip, s), color='red'))
    # annotate Least squares solution
    plt.text(model.LS[1], model.LS[0], 'LS', color='red')

    # draw a "convergence circle" ; if final value lies in this circle, the result is accepted
    if error_margin is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.sqrt(error_margin)

        x1 = r * np.cos(theta) + true[:, 0]
        x2 = r * np.sin(theta) + true[:, 1]
        ax.plot(x1, x2)

    # contour lines

    plt.show()
#
#
# def plot_log_prob(self, model):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from collections import OrderedDict
#
#     chain = chain_mat(self.sampler.chain)
#     minim = torch.min(chain, dim=0).values.detach().numpy()
#     maxim = torch.max(chain, dim=0).values.detach().numpy()
#     x_min = minim[0]
#     x_max = maxim[0]
#     y_min = minim[1]
#     y_max = maxim[1]
#
#     # def z_func(x, y):
#     #     return (1 - (x ** 2 + y ** 3)) * np.exp(-(x ** 2 + y ** 2) / 2)
#
#     def z_func(b0, b1):
#         state_dict = OrderedDict([('W', torch.tensor([[b1]])), ('b', torch.tensor([b0]))])
#         model.load_state_dict(state_dict)
#         return model.log_prob(self.X, self.y)
#
#     x = np.arange(x_min, x_max, -(x_min - x_max) / 10)
#     y = np.arange(y_min, y_max, -(y_min - y_max) / 10)
#     X, Y = np.meshgrid(x, y)
#     Z = torch.zeros_like(torch.tensor(X))
#     for r, (xs, ys) in  enumerate(zip(X, Y)):
#         for c, (x, y) in enumerate(zip(xs, ys)):
#             Z[r][c] = z_func(x, y)
#
#
#     im = plt.imshow(Z.detach().numpy(), cmap=plt.cm.RdBu, extent=(x_min, x_max, y_min, y_max))
#     cset = plt.contour(Z.detach().numpy(), # np.arange(-1, 1.5, 0.2), linewidths=2,
#                        cmap=plt.cm.Set2,
#                        extent=(x_min, x_max, y_min, y_max))
#     plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
#     plt.colorbar(im)
#
# plt.show()
