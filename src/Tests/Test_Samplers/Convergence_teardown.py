import torch

from .util import chain_mat, posterior_mean


class Convergence_teardown:
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
