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


    def convert_chain(self):
        vectorize = lambda D: torch.cat([a.view(a.nelement()) for a in D.values()])
        return [vectorize(c) for c in self.sampler.chain.samples]

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
        self.chain = self.convert_chain()


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
        self.chain = self.convert_chain()


        # self.sampler.posterior_dist()
        # sampler.trace()

        # plt.plot(sampler.chain.accepted_steps)
        # plt.show()

    def sample_SGLD(self, step_size, num_steps, burn_in, pretrain, tune, num_chains=7):
        self.sampler = SGLD_Sampler(
            probmodel=self.model,
            step_size=step_size,
            num_steps=num_steps,
            num_chains=num_chains,
            burn_in=burn_in,
            pretrain=pretrain,
            tune=tune)
        self.sampler.sample_chains()
        self.chain = self.convert_chain()


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.distributions as td
    from Pytorch.Layer.Layer_Probmodel.Hidden_Probmodel import Hidden_ProbModel

    no_in = 10
    no_out = 1

    # single Hidden Unit Example
    reg = Hidden_ProbModel(no_in, no_out, bias=True, activation=nn.Identity())
    reg.true_model = reg.vec

    # reg.W = reg.W_.data
    # reg.b = reg.b_.data
    reg.forward(X=torch.ones(100, no_in))
    reg.prior_log_prob()

    # generate data X, y
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample()

    # log_prob sg mode:
    print(reg.log_prob(X, y))

    ludi = LudwigWinkler(reg, X, y, batch_size=X.shape[0])

    num_samples = 100
    sampler = 'sgnht'
    step_size = 0.1
    num_steps = 2000  # <-------------- important
    pretrain = False
    tune = False
    burn_in = 2000
    # num_chains 		type=int, 	default=1
    num_chains = 1  # os.cpu_count() - 1
    batch_size = 50
    hmc_traj_length = 20
    val_split = 0.9  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val
    val_prediction_steps = 50
    val_converge_criterion = 20
    val_per_epoch = 200

    ludi.sample_SGNHT(step_size, num_steps, burn_in, pretrain=False, tune=tune, hmc_traj_length=hmc_traj_length,
                      num_chains=num_chains)

    type(ludi.sampler.chain)
    ludi.sampler.chain.__dict__
