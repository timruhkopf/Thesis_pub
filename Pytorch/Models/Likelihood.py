import torch
import torch.distributions as td
from Pytorch.Models.Shrinkage_BNN import Shrinkage_BNN
from Pytorch.Models.GAM import GAM




class Like_NormalHeter:
    def __init__(self, model_mean, model_variance):
        self.model_mean = model_mean
        self.model_variance = model_variance

    @property
    def parameters(self):
        return [*self.model_mean.parameters, 'sigma']

    @property
    def bijectors(self):
        return None

    def sample_param(self, **kwargs):
        return [self.model_mean.sample(**kwargs), self.model_variance.sample()]

    def sample_y(self, X, n):
        pass

    def log_prob(self, X, y, param):
        # TODO check if closure is necessary for full data models!
        # return self.model_mean.log_prob(param) + td.reduce_mean(self.likelihood(self.true_param).log_prob(param))
        pass

    # EXPERIMENTAL --------------------------------------------------------------------
    def predict_mean(self, X, param_mean):
        """"""
        self.model_mean.forward(X, param_mean)

    def predict_dist(self, X, param_mean, param_var):
        return td.Normal(self.predict_mean(param_mean), self.model_variance(param_var))

class Like_NormalHomo:
    def __init__(self, model_mean):
        "Assuming homoscedastic, uniit variance"
        self.model_mean = model_mean
        # self.likelihood = td.Normal FIXME: check how torch implementation works

    @property
    def parameters(self):
        return self.model_mean.parameters

    @property
    def bijectors(self):
        return None

    def sample_param(self, **kwargs):
        """
        sample the mean_model's priors to get the prior model initialization
        :param kwargs: passing the mean_model's sampling method's arguments"""
        self.init_param = self.model_mean.sample(**kwargs)
        return self.init_param

    def sample_y(self, X, n, **kwargs):
        self.true_param = self.sample_param(**kwargs)
        # return self.likelihood(self.true_param).sample(n)


    def log_prob(self, param):
        pass

if __name__ == '__main__':
    # EXAMPLE OF INTENDED USAGE
    from Pytorch.Models.BNN import BNN
    bnn_prior = BNN()
    bnn_model = Like_NormalHomo(model_mean=bnn_prior)

    from Pytorch.Models.Shrinkage_BNN import Shrinkage_BNN
    shrinkage_bnn_prior = Shrinkage_BNN()
    shrinkage_bnn_model = Like_NormalHomo(model=shrinkage_bnn_prior)
