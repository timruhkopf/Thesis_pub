import torch
import torch.nn as nn

from .util import chain_mat, odict_pmean


class Convergence_teardown:
    def tearDown(self) -> None:
        """
        Teardown the setUP's regression example to ensure next sampler has new
        model instance to optimize & check the sampler's progression
        """

        # (0) ensure, the sampler moved at all
        self.assertFalse(torch.all(torch.eq(
            chain_mat([self.sampler.chain[-1]]),
            chain_mat([self.model.init_model]))),
            msg='sampler did not progress the chain: init last state are same')

        if not torch.allclose(
                chain_mat([self.model.init_model])[0],
                chain_mat([self.model.true_model])[0], atol=0.03):

            # (1) ensure, sampler does not stay in the vicinity of the init model
            self.assertFalse(torch.allclose(
                chain_mat([self.model.init_model])[0],
                chain_mat([self.sampler.chain[-1]]), atol=0.08),
                msg='sampler did not progress far away from init model, even though'
                    'init and true are distinct from another')

            # (2) check MSE of true and 'posterior mean'
            self.model.load_state_dict(self.model.true_model)
            y_true = self.model.forward(self.X)
            losstrue = nn.MSELoss()(self.y, y_true)

            self.model.load_state_dict(self.model.init_model)
            y_init = self.model.forward(self.X)

            chain = self.sampler.chain
            self.model.load_state_dict(odict_pmean(chain[:int(len(chain) * 3 / 4): 5]))
            y_pmean = self.model.forward(self.X)

            losspmean = nn.MSELoss()(y_pmean, y_true)
            lossinit = nn.MSELoss()(y_init, y_true)
            self.assertTrue(losspmean < lossinit,
                            msg='In terms of avg MSE, the model was not improved during '
                                'sampling (using posterior mean on last quarter of the '
                                'thinned chain as reference)')

            if lossinit - losstrue > 0:
                self.assertTrue(losspmean < (lossinit - losstrue) / 4,
                                msg='Sampling did not reduce the MSE to a fourth of the init-true distance.')

                # self.assertTrue(torch.allclose(loss, torch.tensor(1.), rtol=0.2), msg= "check MSE of true and 'posterior "
                #                                                                         "mean' failed")

            # check the resulting estimates are within a certain range
            # TODO replace posterior_mean difference to Truemodel test with
            #   avg_MSE difference of those two. BUT this means the post.mean must be determined
            #   and parsed from vec back in the model. instead use last state as proxy
            # self.assertTrue(torch.allclose(
            #     chain_mat([self.model.true_model])[0],
            #     posterior_mean(self.sampler.chain[-200:]), atol=0.09),
            #     msg='True parameters != posterior mean(on last 200 steps of chain)')

            # chain_mat([self.model.init_model])[0]
            # chain_mat([self.sampler.chain[-1]])[0]

            # todo avg MSE loss check avg MSE LOSS <= 0.99 quantile von standard Normal?

        # del self.model
        # del self.X
        # del self.y
