import torch
class Samplers:
    def __init__(self):
        """Samplers class, implementing all the functionality shared by the samplers:
        given the MCMC-chain: predictions of (mode, mean --> these two may
        become problematic in the context of multimodality), uncertainty,
        clustering the chain vectors, evaluating single parameters -- unconditional distribution
        plotting of predictions / chains."""
        self.chain = list() # list of 1D Tensors, representing the param state

    def predict(self):
        with torch.no_grad(): # refrain from creating a graph in prediction mode - avoid gradient computation
            for paramset in self.chain:
                # deserialize it potentially
                # paralleliize the for loop:
                # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop

                pass

    @property
    def np_chain(self):
        return torch.cat(self.chain).reshape(len(self.chain), -1).numpy()

if __name__ == '__main__':
    sampler = Samplers()
    sampler.chain.extend([torch.ones(10), torch.zeros(10), torch.ones(10)])
    sampler.np_chain
