import torch

class Util_Sampler:

    def save(self, path):
        import pickle
        torch.save(self.model.state_dict(), path + '.model')

        with open(path + '.sampler_pkl', "wb") as output_file:
            d = {'chain': self.chain, 'true_model': self.model.true_model, 'init_model': self.model.init_model}
            pickle.dump(d, output_file)

    def load(self, path):
        import pickle
        with open(path + '.sampler_pkl', "rb") as input_file:
            self.chain = pickle.load(input_file)['chain']

        self.model.load_state_dict(torch.load(path + '.model'))

    @property
    def chain_mat(self):
        vecs = [torch.cat([p.reshape(p.nelement()) for p in chain.values()], axis=0) for chain in self.chain]
        return torch.stack(vecs, axis=0).numpy()

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
