import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import math
from itertools import chain


class Util_Sampler:
    def __init__(self, model):
        """
        Samplers class, implementing all the functionality shared by the samplers:
        given the MCMC-chain: predictions of (mode, mean --> these two may
        become problematic in the context of multimodality), uncertainty,
        clustering the chain vectors, evaluating single parameters -- unconditional distribution
        plotting of predictions / chains.
        """
        self.chain = list()  # list of 1D Tensors, representing the param state
        self.model = model

    def save(self, path):
        import pickle

        torch.save(self.model.state_dict(), path + '.model')
        # self.__delattr__(self.model)

        with open(path + '.sampler_pkl', "wb") as output_file:
            d = {'chain': self.chain}
            pickle.dump(d, output_file)

    def load(self, path):
        import pickle
        with open(path + '.sampler_pkl', "rb") as input_file:
            self.chain = pickle.load(input_file)['chain']

        self.model.load_state_dict(torch.load(path + '.model'))

    def ess(self, nlags=500):
        """
        Effective Sample Size
        following TACTHMC's formulation:
        n/(1+2\sum_{k=1}^{inf} p(k)) with p(k) the autocorrelation at lag k
        """

        if not hasattr(self, 'acf'):
            self._calc_acf(nlags)

        ess = len(self.chain) / (1 + 2 * np.sum(self.acf, axis=0))
        self.ess_min = int(min(ess))
        return ess

    @property
    def chain_mat(self):
        vecs = [torch.cat([p.view(p.nelement()) for p in chain.values()], axis=0) for chain in self.chain]
        return torch.stack(vecs, axis=0).numpy()

        # return torch.cat(self.chain).reshape(len(self.chain), -1)

    def traceplots(self, path=None):
        df = pd.DataFrame(self.chain_mat)
        s = int(math.ceil(math.sqrt(df.shape[1])))
        axes = df.plot(subplots=True, layout=(s, s), sharex=True, title='Traces', legend=False)

        for ax, v in zip(chain(*axes), self.model.true_vec.detach().numpy()):
            ax.axhline(y=v, color='r')

        plt.show()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight')

    def _calc_acf(self, nlags):
        df = pd.DataFrame(self.chain_mat)
        df_acf = pd.DataFrame(columns=df.columns)
        for i, column in enumerate(list(df)):  # iterate over chain_mat columns
            df_acf[i] = acf(df[column], nlags=nlags, fft=True)

        self.acf = df_acf

    def acf_plots(self, nlags, path=None):
        """
        Autocorrrelation function plot using Fast Fourier Transforms
        connection between acf and fft: Wiener-Khintchine theorem:
        https://www.itp.tu-berlin.de/fileadmin/a3233/grk/pototskyLectures2012/pototsky_lectures_part1.pdf
        :param nlags: int. number of lags, that are to be displayed in"""

        if not hasattr(self, 'acf'):
            self._calc_acf(nlags)

        s = int(math.ceil(math.sqrt(self.acf.shape[1])))
        self.acf.plot(subplots=True, layout=(s, s), sharey=True, sharex=True, title='Autocorrelation', legend=False)

        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight')

    def clean_chain(self):
        """:returns the list of 1d Tensors, that are not consecutively same (i.e.
        the step was rejected)"""
        N = len(self.chain)
        chain = [self.chain[0]]
        chain.extend([s1 for s0, s1 in zip(self.chain, self.chain[1:]) if any(s0 != s1)])
        self.chain = chain

        self.acceptance = len(self.chain) / N
        print(self.acceptance)

    def posterior_mean(self):
        return self.chain_mat.mean(dim=0)

    def posterior_mode(self):
        """
        assumes an attribute self.logs exist, is a 1D tensor, containing
        all log_prob evaluations
        :return: 1D. Tensor: state of the chain, which has the max log_prob
        """
        # flattening the list of indecies
        a = [a[0] for a in torch.nonzero(self.log_probability == max(
            self.log_probability)).numpy().tolist()]
        # return the mode value(s)
        return [c for i, c in enumerate(self.chain) if i in a]


if __name__ == '__main__':
    s = Util_Sampler(model=None)
    chain = torch.distributions.Normal(0., scale=torch.tensor(1.)).sample([1000, 2])
    new = torch.Tensor(1000, )

    alpah = torch.distributions.Normal(0., 0.1).sample([1000, 2])
    alpha = torch.linspace(0.01, 0.05, 10)  # *torch.tensor([-1., 1.]*5)
    new[0:10] = chain[:, 0][0:10]

    for i, c in enumerate(chain[10:-1, 0]):
        i += 10
        new[i + 1] = sum(alpha * new[(i - 10):i]) + c

    s.chain = new

    # Version 1
    x = s.chain.numpy()
    x = (s.chain - s.chain.mean()).numpy()

    result = np.correlate(x, x, mode='full')
    s.acf = result[result.size // 2:][1:]

    # version 2
    s.acf = tidynamics.acf(x)[1: len(s.chain) // 10]

    # version 3
    import scipy.signal as sc

    s.acf = sc.correlate(x, x, mode='full', method='fft')[1:]

    # plot any of the above versions
    lag = np.arange(len(s.acf) - 1)
    plt.plot(lag, s.acf[1:])
    plt.show()

    t = np.arange(len(s.chain))
    plt.plot(t, s.chain)

    s.autocorr1(x=x, t=2)

    s.fast_autocorr(column=0)
