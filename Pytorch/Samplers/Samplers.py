import torch


class Sampler:
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
        self.acceptance = None

    @property
    def chain_mat(self):
        return torch.cat(self.chain).reshape(len(self.chain), -1)

    @torch.no_grad()
    def predict(self, model, X):
        """predict X for each on each accepted state of the chain"""
        y = list()

        for state in self.chain:
            # deserialize it potentially
            # TODO paralleliize the prediction for loop:
            # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop

            model.vec_to_attrs(state)
            y.append(model.forward(X))

    def clean_chain(self):
        """:returns the list of 1d Tensors, that are not consecutively same (i.e.
        the step was rejected)"""
        chain = [self.chain[0]]
        chain.extend([s1 for s0, s1 in zip(self.chain, self.chain[1:]) if any(s0 != s1)])
        self.chain = chain

        self.acceptance = len(self.chain) / N
        print(self.acceptance)

    @property
    def np_chain(self):
        """:return: 2d-ndarray (numpy), each row representing a state"""
        return self.chain_mat.numpy()

    def posterior_mean(self):
        return self.chain_mat.mean(dim=1)

    def posterior_mode(self):
        """
        assumes an attribute self.logs exist, is a 1D tensor, containing
        all log_prob evaluations
        :return: 1D. Tensor: state of the chain, which has the max log_prob
        """
        return self.chain[self.logs.index(max(self.logs))]

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


if __name__ == '__main__':
    sampler = Sampler()
    sampler.chain.extend([torch.ones(10), torch.zeros(10), torch.ones(10)])
    sampler.np_chain
