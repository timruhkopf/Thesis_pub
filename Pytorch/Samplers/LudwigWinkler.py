import torch
from Pytorch.Samplers.Samplers import Sampler
from thirdparty_repo.ludwigwinkler.src.MCMC_Sampler import SGLD_Sampler, MALA_Sampler, SGNHT_Sampler
from torch.utils.data import TensorDataset, DataLoader

class LudwigWinkler(Sampler):
    def __init__(self, model, X, y, batch_size=None):
        """interface to https://github.com/ludwigwinkler/
        BE AWARE, that the entire repo is build on Pytorch.Optim, which require
        that each parameter used and optimized is a nn.Parameter!"""
        self.model = model
        self.data = X
        self.target = y

        if batch_size is not None:
            batch_size = self.data.shape[0]


        self.model.dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=batch_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.log_prob_ = self.model.log_prob
        self.model.log_prob = lambda X, y : {'log_prob':self.model.log_prob_(X, y)}

        if torch.cuda.is_available():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            FloatTensor = torch.cuda.FloatTensor
            Tensor = torch.cuda.FloatTensorf
        else:
            device = torch.device('cpu')
            FloatTensor = torch.FloatTensor
            Tensor = torch.FloatTensor

    def sample_MALA(self, step_size, num_steps, burn_in, pretrain, tune, num_chains):
        self.sampler = MALA_Sampler(
            probmodel=self.model,
            step_size=step_size,
            num_steps=num_steps,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune,
            num_chains=num_chains)
        self.sampler.sample_chains()

    def sample_SGNHT(self, step_size, num_steps, burn_in, pretrain, tune, hmc_traj_length, num_chains):
        self.sampler = SGNHT_Sampler(
            probmodel=self.model,
            step_size=step_size,
            num_steps=num_steps,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune,
            traj_length=hmc_traj_length,
            num_chains=num_chains)

        self.sampler.sample_chains()

        # self.sampler.posterior_dist()
        # sampler.trace()

        # plt.plot(sampler.chain.accepted_steps)
        # plt.show()

    def sample_SGLD(self, step_size, num_steps, burn_in, pretrain, tune):
        self.sampler = SGLD_Sampler(
            probmodel=self.model,
            step_size=step_size,
            num_steps=num_steps,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune)
        self.sampler.sample_chains()


if __name__ == '__main__':
    pass
