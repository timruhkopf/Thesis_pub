import torch
import torch.nn as nn

from src.Layer.Hidden import Hidden
from src.Tests.Test_Samplers.util import chain_mat, posterior_mean
from ..Test_Samplers.util import plot_sampler_path


class Regression_Convergence_Setup:
    def setUp(self) -> None:
        # Regression example data (from Hidden layer)
        # TODO make it 5 regressors + bias!
        p = 1  # no. regressors
        self.model = Hidden(p, 1, bias=True, activation=nn.Identity())
        X, y = self.model.sample_model(n=100)
        self.model.reset_parameters()
        self.X = X
        self.y = y

        print(self.model.true_model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        self.X.to(device)
        self.y.to(device)

        # steps for (real) Samplers
        self.steps = 1000

        # OLS for comparison
        X = self.X.clone()
        X = torch.cat([torch.ones(X.shape[0], 1), X], 1)
        self.model.LS = torch.inverse(X.t() @ X) @ X.t() @ y  # least squares
        self.model.LS = self.model.LS.numpy()
        self.model_in_LS_format = lambda: torch.cat([self.model.b.reshape((1, 1)), self.model.W.data], 0)

    def tearDown(self) -> None:
        """
        Teardown the setUP's regression example to ensure next sampler has new
        model instance to optimize & check the sampler's progression
        """

        # self.model.load_state_dict(self.model.true_model)
        # print('true_log_prob: {}\n'
        #       'last_log_prob:{}'.format(self.model.log_prob(self.X, self.y), -self.sampler.loss[-1]))
        #
        plot_sampler_path(self.sampler, self.model, steps=self.steps, skip=50,
                          loss=self.sampler.loss)

        # plot_log_prob(self, self.model)

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
        # FIXME: plot circle does not allign with this models's decision!
        self.assertTrue(torch.allclose(
            chain_mat([self.model.true_model])[0],
            posterior_mean(self.sampler.chain[-200:]), atol=0.15),
            msg='True parameters != posterior mean(on last 200 steps of chain)')

        del self.model
        del self.X
        del self.y
