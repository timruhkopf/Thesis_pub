import numpy as np
import torch
import tidynamics



class Sampler_Experimental:

    def chain_log_prob(self, X, y):
        """calculate the log_prob value for each state in the chain cond. on X & y"""
        log_prob = torch.Tensor(len(self.chain))
        for i, state in enumerate(self.chain):
            self.model.vec_to_attrs(state)
            log_prob[i] = (self.model.log_prob(X, y))

        return log_prob

    # def autocorr(self, column):
    #     x = (self.chain_mat[:, column]).numpy()
    #     result = np.correlate(x, x, mode='full')
    #     self.acf = result[result.size // 2:]
    #
    # @staticmethod
    # def autocorr1(x, t=1):
    #     """autocrrelation of a 1d array for a lag t"""
    #     return np.corrcoef(np.array([x[:-t], x[t:]]))
    #
    # def fast_autocorr(self, column):
    #     """http://lab.pdebuyl.be/tidynamics/auto_examples/plot_acf_1.html#sphx-glr-auto-examples-plot-acf-1-py"""
    #     self.acf = tidynamics.acf(self.chain_mat[:, column].numpy())
    #     # self.acf = tidynamics.acf((self.chain_mat - self.chain_mat.mean(axis=0)).numpy())
    #
    # def plot_acf(self, path=None):
    #     # lag = np.arange(len(self.chain.) // 64) * dt
    #     lag = np.arange(len(self.acf))
    #     if path is None:
    #         plt.legend()
    #         plt.title('Chain\'s ACF')
    #         plt.xlabel(r'$\tau$')
    #         plt.ylabel(r'$v(t) v(t+\tau)$')
    #         plt.plot(lag, self.acf)
    #         plt.show()
    #
    #     else:
    #         plt.savefig('{}_acf.png'.format(path), bbox_inches='tight')
    #
    #
    # def acf(self, lag=100):
    #     self.chain.numpy()
    #
    #
    #
    # def auto_corr(self, X, y, kappa=100):
    #     '''
    #     This is the intuitive and naive way to calculate autocorrelations. See
    #     auto_corr_fast(...) instead.
    #     '''
    #     M = self.np_chain
    #     #   The autocorrelation has to be truncated at some point so there are enough
    #     #   data points constructing each lag. Let kappa be the cutoff
    #     auto_corr = np.zeros((kappa - 1))
    #
    #     # FIXME: Expected value of a variable is not necessarily its mean: \sum_i x_i *P(x = x_i)!
    #     # consider the log probability!
    #     mu = self.expected_value(X, y).numpy()
    #     # mu = np.mean(M)
    #     for s in range(1, kappa - 1):
    #         auto_corr[s] = np.mean((M[:-s] - mu) * (M[s:] - mu)) / np.var(M)
    #     return auto_corr
    #
    # def auto_corr_fast(self, X, y, kappa=100):
    #     '''
    #     AUTOCORRELATION: FAST FOURTIER TRANSFORM
    #     The bruteforce way to calculate autocorrelation curves is defined in
    #     auto_corr(M). The correlation is computed for an array against itself, and
    #     then the indices of one copy of the array are shifted by one and the
    #     procedure is repeated. This is a typical "convolution-style" approach.
    #     An incredibly faster method is to Fourier transform the array first, since
    #     convolutions in Fourier space is simple multiplications. This is the approach
    #     in auto_corr_thefast(...)
    #     '''
    #     M = self.np_chain
    #     #   The autocorrelation has to be truncated at some point so there are enough
    #     #   data points constructing each lag. Let kappa be the cutoff
    #
    #     # FIXME: Expected value of a variable is not necessarily its mean: \sum_i x_i *P(x = x_i)!
    #     # consider the log probability!
    #     M = M - self.expected_value(X, y).numpy()
    #     # M = M - np.mean(M)
    #
    #     N = len(M)
    #     fvi = np.fft.fft(M, n=2 * N)
    #     #   G is the autocorrelation curve
    #     G = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    #     G /= N - np.arange(N);
    #     G /= G[0]
    #     G = G[:kappa]
    #
    #     # TODO plot the autocorrelation curve G
    #     return G

    # def aggregate_priors(self, N, seperated=False):
    #     """
    #     sampling prior models
    #
    #     allows e.g.
    #     prior_models = self.aggregate_priors(N=100)
    #     X = ...
    #     mus = list()
    #     for state in prior_models:
    #         # TODO: parallelize the model_priors sampling
    #         self.model.
    #         self.mus.append(self.model.forward(X))
    #     """
    #     prior_models = list()
    #     for i in range(N):
    #         # TODO: parallelize the model_priors sampling
    #         self.model.reset_parameters()
    #         prior_models.append(self.model.vec)
    #     return prior_models

    # def plot_traces(self, X, y, stride):
    #     """
    #
    #     :param stride: for large chains, choose a stride of the chain to display
    #     :return: plot
    #     """
    #
    #     raise NotImplementedError()
    #     self.model(X)
    #     import matplotlib.pyplot as plt
    #
    #     # PSEUDO CODE ------------------------------
    #     # plt.figure()
    #     # if X.shape[1] == 1:
    #     #     # 1d plot
    #     #     plt = self.model.plot1d(X, y, param=self.model.true_model, plt=plt)
    #     #     plt.scatter(X, y)
    #     #     for c in self.chain[::stride]:
    #     #         plt = self.model.plot1d(X, y, param=c, plt=plt)
    #     #
    #     # elif X.shape[1] == 2:
    #     #     # trisurface plot
    #     #     self.model.plot2d
    #     # else:
    #     #     # joint plot of GAM & BNN in shrinkage
    #     #     self.model.plot1d
    #     #     self.model.plot2d
    #     #
    #     # plt.show()
    #
    #     pass

    # def parallel_coord_chain(self, df):
    #     """plot the chain in a parallel coordinates plot to find"""
    #
    #     # taken from: https://stackoverflow.com/a/60401570/10226710
    #     import matplotlib.pyplot as plt
    #     from matplotlib.path import Path
    #     import matplotlib.patches as patches
    #     import numpy as np
    #
    #     # create some dummy data
    #     ynames = list(df.columns[:-1])  # ['P1', 'P2', 'P3', 'P4', 'P5']
    #     N = len(df)
    #     category = list(df.labels)  # np.concatenate([np.full(N1, 1), np.full(N2, 2), np.full(N3, 3)])
    #     # organize the data
    #     ys = np.array(df.loc[:, df.columns != 'labels'])
    #     ymins = ys.min(axis=0)
    #     ymaxs = ys.max(axis=0)
    #     dys = ymaxs - ymins
    #     ymins -= dys * 0.05  # add 5% padding below and above
    #     ymaxs += dys * 0.05
    #     dys = ymaxs - ymins
    #
    #     # transform all data to be compatible with the main axis
    #     zs = np.zeros_like(ys)
    #     zs[:, 0] = ys[:, 0]
    #     zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    #
    #     fig, host = plt.subplots()
    #     axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    #     for i, ax in enumerate(axes):
    #         ax.set_ylim(ymins[i], ymaxs[i])
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['bottom'].set_visible(False)
    #         if ax != host:
    #             ax.spines['left'].set_visible(False)
    #             ax.yaxis.set_ticks_position('right')
    #             ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
    #
    #     host.set_xlim(0, ys.shape[1] - 1)
    #     host.set_xticks(range(ys.shape[1]))
    #     host.set_xticklabels(ynames, fontsize=14)
    #     host.tick_params(axis='x', which='major', pad=7)
    #     host.spines['right'].set_visible(False)
    #     host.xaxis.tick_top()
    #     host.set_title('Parallel Coordinates Plot', fontsize=18)
    #
    #     colors = plt.cm.Set2.colors
    #     legend_handles = [None for _ in category]
    #     for j in range(N):
    #         # to just draw straight lines between the axes:
    #         # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])
    #
    #         # create bezier curves
    #         # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
    #         #   at one third towards the next axis; the first and last axis have one less control vertex
    #         # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
    #         # y-coordinate: repeat every point three times, except the first and last only twice
    #         verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
    #                          np.repeat(zs[j, :], 3)[1:-1]))
    #         codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    #         path = Path(verts, codes)
    #         patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.7, edgecolor=colors[np.array(df.labels)[j]])
    #         legend_handles[np.array(df.labels)[j]] = patch
    #         host.add_patch(patch)
    #         host.legend(legend_handles, category,
    #                     loc='lower center', bbox_to_anchor=(0.5, -0.18),
    #                     ncol=len(category), fancybox=True, shadow=True)
    #     plt.tight_layout()
    #     plt.show()
    #
    # def cluster_chain(self, min_cluster_size=15, **kwargs):
    #     """
    #     clustering the chain to figure out potential multimodality.
    #     Using HDBSCAN as a clusterer, dense regions are summarized
    #     :param min_cluster_size:  HDBSCAN param
    #     :param kwargs: HDBSCAN params
    #     :return: pd.DataFrame
    #     """
    #
    #     import hdbscan
    #     import pandas as pd
    #
    #     chainmat = pd.DataFrame(self.chain_mat.numpy())
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size).fit(chainmat)
    #     chainmat['labels'] = clusterer.labels_
    #
    #     return chainmat
    #
    # def stratified_clusters(self, n=2, **kwargs):
    #     """
    #     parallel-coordinates plot on stratified sample of the chain clustered by
    #     hdbscan.
    #
    #     :param n: max number of samples per cluster
    #     :param kwargs: hdbscan params
    #     :return:
    #     """
    #     chainmat = self.cluster_chain(**kwargs)
    #     chainmat = chainmat[chainmat.labels != -1]  # remove "noise" label of hdbscan
    #
    #     # stratified sampling
    #     sample = chainmat.groupby('labels', group_keys=False).apply(lambda x: x.sample(min(len(x), n)))
    #
    #     self.parallel_coord_chain(sample)
