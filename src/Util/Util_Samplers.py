import torch


class Util_Sampler:
    @property
    def chain_mat(self):
        vecs = [torch.cat([p.reshape(p.nelement()) for p in chain.values()], axis=0) for chain in self.chain]
        return torch.stack(vecs, axis=0).numpy()

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
