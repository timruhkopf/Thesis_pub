import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from src.Samplers import *
from src.Layer import Hidden



def chain_mat(chain):
    vecs = [torch.cat([p.reshape(p.nelement()) for p in state.values()], axis=0) for state in chain]
    return torch.stack(vecs, axis=0)


def posterior_mean(chain):
    chain_matrix = chain_mat(chain)
    return chain_matrix.mean(dim=0)


class Test_Samplers(unittest.TestCase):
    def setUp(self) -> None:
        # Regression example data (from Hidden layer)
        # TODO make it 5 regressors + bias!
        p = 1  # no. regressors
        self.model = Hidden(p, 1, bias=True, activation=nn.Identity())
        X, y = self.model.sample_model(n=1000)
        self.model.reset_parameters()
        self.X = X
        self.y = y

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        self.X.to(device)
        self.y.to(device)

        X = self.X.clone()
        X = torch.cat([torch.ones(X.shape[0], 1), X], 1)
        self.LS = torch.inverse(X.t() @ X) @ X.t() @ y  # least squares
        self.model_in_LS_format = lambda: torch.cat([self.model.b.reshape((1, 1)), self.model.W.data], 0)

    def tearDown(self) -> None:
        """
        Teardown the setUP's regression example to ensure next sampler has new
        model instance to optimize & check the sampler's progression
        """

        # ensure, the sampler moved at all
        self.assertFalse(torch.all(torch.eq(
            chain_mat([self.model.true_model])[0],
            chain_mat([self.model.init_model]))),
            msg='sampler did not progress the chain: init & true model are same')

        if not torch.allclose(
                chain_mat([self.model.init_model])[0],
                chain_mat([self.model.true_model])[0], atol=0.03):
            # ensure, sampler does not stay in the vicinity of the init model
            self.assertFalse(torch.allclose(
                chain_mat([self.model.init_model])[0],
                chain_mat([self.sampler.chain[-1]]), atol=0.08),
                msg='sampler did not progress far away from init model, even though'
                    'init and true are distinct from another')

        # check the resulting estimates are within a certain range
        self.assertTrue(torch.allclose(
            chain_mat([self.model.true_model])[0],
            posterior_mean(self.sampler.chain[-200:]), atol=0.09),
            msg='True parameters != posterior mean(on last 200 steps of chain)')

        chain_mat([self.model.init_model])[0]
        chain_mat([self.sampler.chain[-1]])[0]

        # todo avg MSE loss check avg MSE LOSS <= 0.99 quantile von standard Normal?

        del self.model
        del self.X
        del self.y


    def test_RHMC(self):
        # sampler config
        eps, L = 0.001, 1
        sampler_param = dict(epsilon=eps, L=L)

        burn_in, n_samples = 100, 1000
        batch_size = self.X.shape[0]

        # dataset setup
        trainset = TensorDataset(self.X, self.y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # check that the model's log_prob is not broken
        # TODO RHMC.step(partial(self.model.log_prob, *data))  # maybe negative?
        # self.model.log_prob = lambda X, y: self.model.prior_log_prob() + self.model.likelihood(X).log_prob(y).mean()

        # init sampler & sample
        Sampler = myRHMC
        self.sampler = Sampler(self.model, **sampler_param)
        self.sampler.sample(trainloader, burn_in, n_samples)

    # def test_MALA(self):
    #     burn_in, n_samples = 100, 1000
    #     batch_size = self.X.shape[0]
    #
    #     sampler_param = {'epsilon': 0.001, 'num_steps': n_samples,
    #                      'burn_in': burn_in, 'pretrain': False, 'tune': False, 'num_chains': 1}
    #
    #     # dataset setup
    #     trainset = TensorDataset(self.X, self.y)
    #     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    #     Sampler = MALA  # step_size
    #     self.sampler = Sampler(self.model, trainloader, **sampler_param)
    #     self.sampler.sample()

    #     self.assertTrue(torch.allclose(chain_mat([self.model.true_model])[0],
    #                                    posterior_mean(sampler.chain[-200:]), atol=0.03),
    #                     msg='True parameters != posterior mean(on last 200 steps of chain)')
    #
    #     self.model.init_model
    #     self.model.true_model
    #     self.sampler.chain[-1]
    #
    # def test_SGRHMC(self):
    #     # instantiate Sampler & sampler_config
    #     # if sampler_name in ['SGNHT', 'RHMC', 'SGRHMC']:
    #     #     sampler_param.update(dict(L=L))
    #     #
    #     # if sampler_name == 'SGRHMC':
    #     #     sampler_param.update(dict(alpha=0.2))
    #     #
    #     # if 'SG' in sampler_name:
    #     #     batch_size = sg_batch
    #     # else:
    #     #     batch_size = X.shape[0]
    #     #
    #     # # sample
    #
    #     # check the resulting estimates are within a certain range
    #
    #     pass
    #
    # def test_SGNHT(self):
    #     pass
    #


if __name__ == '__main__':
    unittest.main(exit=False)
