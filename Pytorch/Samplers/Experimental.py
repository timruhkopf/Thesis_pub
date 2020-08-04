class Experimental:
    def aggregate_priors(self, N, seperated=False):
        """
        sampling prior models

        allows e.g.
        prior_models = self.aggregate_priors(N=100)
        X = ...
        mus = list()
        for state in prior_models:
            # TODO: parallelize the model_priors sampling
            self.model.
            self.mus.append(self.model.forward(X))
        """
        prior_models = list()
        for i in range(N):
            # TODO: parallelize the model_priors sampling
            self.model.reset_parameters()
            prior_models.append(self.model.vec)
        return prior_models

    def autocorrelation(self):
        raise NotImplementedError()

    def plot_chain(self, X, y, stride):
        """

        :param stride: for large chains, choose a stride of the chain to display
        :return: plot
        """

        raise NotImplementedError()
        self.model(X)
        import matplotlib.pyplot as plt

        # PSEUDO CODE ------------------------------
        # plt.figure()
        # if X.shape[1] == 1:
        #     # 1d plot
        #     plt = self.model.plot1d(X, y, param=self.model.true_model, plt=plt)
        #     plt.scatter(X, y)
        #     for c in self.chain[::stride]:
        #         plt = self.model.plot1d(X, y, param=c, plt=plt)
        #
        # elif X.shape[1] == 2:
        #     # trisurface plot
        #     self.model.plot2d
        # else:
        #     # joint plot of GAM & BNN in shrinkage
        #     self.model.plot1d
        #     self.model.plot2d
        #
        # plt.show()

        pass

    def parallel_coord_chain(self, df):

        # taken from: https://stackoverflow.com/a/60401570/10226710
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
        import numpy as np

        # create some dummy data
        ynames = list(df.columns[:-1])  # ['P1', 'P2', 'P3', 'P4', 'P5']
        N = len(df)
        category = list(df.labels)  # np.concatenate([np.full(N1, 1), np.full(N2, 2), np.full(N3, 3)])
        # organize the data
        ys = np.array(df.loc[:, df.columns != 'labels'])
        ymins = ys.min(axis=0)
        ymaxs = ys.max(axis=0)
        dys = ymaxs - ymins
        ymins -= dys * 0.05  # add 5% padding below and above
        ymaxs += dys * 0.05
        dys = ymaxs - ymins

        # transform all data to be compatible with the main axis
        zs = np.zeros_like(ys)
        zs[:, 0] = ys[:, 0]
        zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

        fig, host = plt.subplots()
        axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
        for i, ax in enumerate(axes):
            ax.set_ylim(ymins[i], ymaxs[i])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if ax != host:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
                ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

        host.set_xlim(0, ys.shape[1] - 1)
        host.set_xticks(range(ys.shape[1]))
        host.set_xticklabels(ynames, fontsize=14)
        host.tick_params(axis='x', which='major', pad=7)
        host.spines['right'].set_visible(False)
        host.xaxis.tick_top()
        host.set_title('Parallel Coordinates Plot', fontsize=18)

        colors = plt.cm.Set2.colors
        legend_handles = [None for _ in category]
        for j in range(N):
            # to just draw straight lines between the axes:
            # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                             np.repeat(zs[j, :], 3)[1:-1]))
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.7, edgecolor=colors[np.array(df.labels)[j]])
            legend_handles[np.array(df.labels)[j]] = patch
            host.add_patch(patch)
            host.legend(legend_handles, category,
                        loc='lower center', bbox_to_anchor=(0.5, -0.18),
                        ncol=len(category), fancybox=True, shadow=True)
        plt.tight_layout()
        plt.show()

    def cluster_chain(self, min_cluster_size=15, **kwargs):
        """
        clustering the chain to figure out potential multimodality.
        Using HDBSCAN as a clusterer, dense regions are summarized
        :param min_cluster_size:  HDBSCAN param
        :param kwargs: HDBSCAN params
        :return: pd.DataFrame
        """

        import hdbscan
        import pandas as pd

        chainmat = pd.DataFrame(self.chain_mat.numpy())
        clusterer = hdbscan.HDBSCAN(min_cluster_size).fit(chainmat)
        chainmat['labels'] = clusterer.labels_

        return chainmat

    def stratified_clusters(self, n=2, **kwargs):
        """
        parallel-coordinates plot on stratified sample of the chain clustered by
        hdbscan.

        :param n: max number of samples per cluster
        :param kwargs: hdbscan params
        :return:
        """
        chainmat = self.cluster_chain(**kwargs)
        chainmat = chainmat[chainmat.labels != -1]  # remove "noise" label of hdbscan

        # stratified sampling
        sample = chainmat.groupby('labels', group_keys=False).apply(lambda x: x.sample(min(len(x), n)))

        self.parallel_coord_chain(sample)
