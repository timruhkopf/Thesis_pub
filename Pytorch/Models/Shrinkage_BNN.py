from Pytorch.Models.BNN import BNN
from Pytorch.Models.GAM import GAM
import Pytorch.Layer as layer


class Shrinkage_BNN(BNN):
    # Inheritance allows to reuse layer tailored forward & log_prob & properties of parameters & bijectors
    shrinkage = {
        'glasso': layer.Group_lasso,
        'gspike': layer.Group_SpikeNSlab,
        'ghorse': layer.Group_HorseShoe
    }

    def __init__(self, hunits=[2,10,1], activation='relu', shrinkage='glasso',
                 **gam_param):
        """
        by default, it is to be decided whether the first input is to be estimated by
        bnn or gam!
        :param hunits:
        :param activation:
        :param shrinkage:
        :param gam_param:
        """
        super(Shrinkage_BNN, self).__init__(hunits, activation)
        self.layers[0] = self.shrinkage[shrinkage](no_in=hunits[0]-1, no_units=hunits[1], activation=activation)

        self.gam = GAM(**gam_param)
    def sample(self, seperate=True):
        """
        Sample the entire Prior Model
        :param seperate: True indicates whether or not, the
        :return: list of randomly sampled parameters
        """
        if seperate:
            pass
        else:
            pass

    def forward(self, X, param_bnn, param_gam):
        # param_bnn contains the
        self.gam.dense(X[:, 0], param_gam)
        self.bnn.forward(X[:, 1:], bnn_param)


if __name__ == '__main__':
    from Pytorch.Models.Likelihood import Like_NormalHomo

    Like_NormalHomo(model_mean=Shrinkage_BNN())
