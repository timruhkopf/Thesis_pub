import torch

from Pytorch.Samplers.Sampler_Experimental import Sampler_Experimental


class Sampler(Sampler_Experimental):
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

    @property
    def np_chain(self):
        """:return: 2d-ndarray (numpy), each row representing a state"""
        return self.chain_mat.numpy()

    def clean_chain(self):
        """:returns the list of 1d Tensors, that are not consecutively same (i.e.
        the step was rejected)"""
        N = len(self.chain)
        chain = [self.chain[0]]
        chain.extend([s1 for s0, s1 in zip(self.chain, self.chain[1:]) if any(s0 != s1)])
        self.chain = chain

        self.acceptance = len(self.chain) / N
        print(self.acceptance)

    def save(self, path):
        # import pickle

        torch.save(self, path)
        #
        # with open(path, 'wb') as p:
        #     pickle.dump(self, p)

    @staticmethod
    def load(path):
        # import pickle

        # with open(path, 'rb') as p:
        #     return pickle.load(self, p)
        return torch.load(path)

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
    from Pytorch.Models.GAM import GAM
    import torch
    import torch.distributions as td
    from Tensorflow.Effects.bspline import get_design

    # saving model:
    no_basis = 20
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([100]))
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis), dtype=torch.float32, requires_grad=False)
    Z.detach()

    gam = GAM(no_basis=no_basis, order=1)
    gam.reset_parameters()
    gam(Z)

    gam.forward(Z)
    y = gam.likelihood(Z).sample()

    theta = gam.flatten()

    from Pytorch.Samplers.TRASH.Hamil import Hamil
    theta
    hamil = Hamil(gam, Z, y, theta)
    hamil.sample_NUTS(1000, 0.3, 5)

    hamil.model.vec
    hamil.save(path='/home/tim/PycharmProjects/Thesis/Pytorch/Chains/hamil2')
    hamil.model.vec_to_attrs(torch.ones_like(theta))
    hamil2 = Hamil.load('/home/tim/PycharmProjects/Thesis/Pytorch/Chains/hamil2')
    hamil2.forward(Z)

    # FIXME X & Z in GAM PLOTTING !!!
    hamil2.model.plot1d(X, Z, y)

    sampler = Sampler()
    sampler.chain.extend([torch.ones(10), torch.zeros(10), torch.ones(10)])
    sampler.np_chain
