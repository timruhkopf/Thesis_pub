import torch
import torch.nn as nn

from src.Layer import Hidden


def chain_mat(chain):
    vecs = [torch.cat([p.reshape(p.nelement()) for p in state.values()], axis=0) for state in chain]
    return torch.stack(vecs, axis=0)


def posterior_mean(chain):
    chain_matrix = chain_mat(chain)
    return chain_matrix.mean(dim=0)


class Convergence_Unittest_Setup:
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
