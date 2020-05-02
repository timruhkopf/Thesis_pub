from itertools import chain
from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Bayesian.layers.Hidden import Hidden
from Python.Bayesian.layers.HiddenFinal import HiddenFinal


class BNN:
    # a tfp implementation with post:
    # https://janosh.io/blog/hmc-bnn

    # on how to write BNN models in Code (later example has even TB)
    # https://github.com/tensorflow/probability/issues/292 # but claimed to not work

    # explicit formulation (formula) of BNN modeling mean of y~N and indep.
    # gaussian priors for for weights and biases.
    # https://www.cs.tufts.edu/comp/150BDL/2018f/assignments/hw2.html

    # A TORCH impementation of BNN & SGMCMC can be found here:
    # https://sgmcmc.readthedocs.io/en/latest/index.html

    def __init__(self, hunits=[2, 10, 9, 8, 1], activation='relu'):
        """
        model specification:
        :param hunits: list: number of hidden units per layer. first element is
        input shape. last element is output_shape.
        :param activation: activation function for
        """
        self.hunits = hunits
        self.layers = [Hidden(i, u, activation)
                       for i, u in zip(self.hunits[:-2], self.hunits[1:-1])]
        self.layers.append(HiddenFinal(input_shape=self.hunits[-2],
                                       no_units=self.hunits[-1],
                                       activation='identity'))

        # to meet sample_chain's flat tensorlist format. (Layers)
        self.layer_index = list(chain(*[[i] * len(h.parameters)
                                        for i, h in enumerate(self.layers)]))
        self.parameters_list = list(chain(*[list(h.parameters) for h in self.layers]))
        self.bijectors = list(chain(*[h.bijectors_list for h in self.layers]))
        # self.parameters = [h.parameters for h in self.layers]

        # to meet sample_chain's flat tensorlist format. (Likelihood)
        self.parameters_list.append('sigma')
        self.bijectors.append(tfb.Exp())

    def _layer_sample(self):
        # draw each layers prior!
        return [h.sample() for h in self.layers]

    @tf.function
    def _layer_log_prob(self, tensorlist):
        return sum([tf.reduce_sum(h.prior_log_prob(p))
                    for h, p in zip(self.layers, tensorlist)])

    @tf.function
    def forward(self, X, param):
        """
        BNN forward path. f(X)
        :param X: input tensor
        :param param: list of dicts, each containing the respective layer's
        parameter tensors. See tfd.JointDistributionNamed(dict()).sample() output
        :return: Tensor; result of last layer
        """
        for h, p in zip(self.layers, param):
            X = h.dense(X, **p)
        return X

    def likelihood_model(self, X, param):
        return tfd.JointDistributionNamed(OrderedDict(
            sigma=tfd.Gamma(0.5, 0.5),
            y=lambda sigma: tfd.Sample(
                tfd.Normal(loc=self.forward(X, param), scale=sigma)))
        )

    def _closure_log_prob(self, X, y):
        """A closure, to preset X, y in this model and
        match HMC's expected model format"""

        # @tf.function  # NOTICE: seems to be ignored by autograph ( Cause: expected exactly one node node, found [<gast.gast.FunctionDef object at 0x7f59e006d940>, <gast.gast.Return object at 0x7f59e006da90>] )
        def BNN_log_prob(*tensorlist):
            paramdicts = self.parse_tensorlist(tensorlist)

            likelihood = self.likelihood_model(X, paramdicts)
            return self._layer_log_prob(param) + \
                   tf.reduce_sum(likelihood.log_prob(y=y, sigma=tensorlist[-1]))

        return BNN_log_prob

    def flat_initialize(self):
        self.layer_init = self._layer_sample()
        paramlist = list(chain(*[list(hparam.values()) for hparam in self.layer_init]))
        paramlist.append(self.likelihood_model(X, self.layer_init).sample()['sigma'])
        return paramlist

    def parse_tensorlist(self, tensorlist):
        paramdicts = [dict() for i in range(len(self.layers))]
        for i, (l, name) in enumerate(zip(self.layer_index, self.parameters_list)):
            paramdicts[l][name] = tensorlist[i]
        return paramdicts


if __name__ == '__main__':
    from Python.Bayesian.plot1d import plot1d_functions
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    # (0) 1D Example
    bnn = BNN(hunits=[1, 5, 10, 1], activation='relu')
    n = 200
    # X = tf.reshape(tf.linspace(-10., 10., n), (n, 1))
    X = tfd.Uniform(-10., 10.).sample((n, 1))

    # check sampling layers & forward work
    param = bnn._layer_sample()
    bnn.forward(X, param)

    # check likelihood works
    likelihood = bnn.likelihood_model(X, param)
    likelihood_true = likelihood.sample()
    y, sigma = likelihood_true['y'], likelihood_true['sigma']

    # check flat_initialize & BNN_log_prob work
    tensorlist = bnn.flat_initialize()
    bnn.unnormalized_log_prob = bnn._closure_log_prob(X, y)
    bnn.unnormalized_log_prob(*tensorlist)

    # check parser:
    print(bnn.layer_init == bnn.parse_tensorlist(tensorlist))

    # plot the funcitions
    true = bnn.forward(X, param)
    init = bnn.forward(X, bnn.layer_init)

    plot1d_functions(X, y, **{'true': true, 'init': init})

    # TODO: FIND A REASONABLE INIT FOR SIGMA! e.g. by local mean & local variance
    #   for a local subsample of the data. (assuming homoscedasticity)
    # sampling
    adHMC = AdaptiveHMC(
        initial=tensorlist,
        bijectors=bnn.bijectors,
        log_prob=bnn.unnormalized_log_prob)

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(10e2),
        num_results=int(5*10e1),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    # mean prediction
    meanPost = adHMC.predict_mean()
    param = bnn.parse_tensorlist(meanPost)
    y_map = bnn.forward(X, param)

    modePost = adHMC.predict_mode(bnn.unnormalized_log_prob)
    param = bnn.parse_tensorlist(modePost)
    y_mode = bnn.forward(X, param)

    plot1d_functions(X, y, **{
        'true': true, 'init': init,
        'posterior mean': y_map, 'posterior mode': y_mode})

    # plot multiple function realizations
    from random import sample


    # TODO FILTER FOR ACCEPTED ONLY!!
    paramsets = [s for s, accepted in zip(
        zip(*samples), traced.inner_results.is_accepted.numpy()) if accepted]
    plot1d_functions(
        X, y, true=true,
        **{str(i): bnn.forward(X, bnn.parse_tensorlist(param)) for i, param in enumerate(
            sample(paramsets, k=20))})  # [-1] is sigma chain!

    # plot confidence intervals
    # paramsets = [s for s in zip(*samples)]
    predictive = [
        tfd.Normal(loc=bnn.forward(X, bnn.parse_tensorlist(paramlist[:-1])),
                   scale=paramlist[-1])
        for i, paramlist in enumerate(sample(paramsets, k=60))]

    predictive_sample = tf.stack([likelihood.sample(100) for likelihood in predictive], axis=-1)
    quantiles = tfp.stats.quantiles(predictive_sample, num_quantiles=10, axis=0)
    ten, ninety = tf.reduce_mean(quantiles[1], axis=1), tf.reduce_mean(quantiles[9], axis=1)

    # CAREFULL: next step is not necessary with GAM_RW!!!
    ten, ninety = tf.reduce_mean(ten, axis = 1), tf.reduce_mean(ninety, axis=1)

    D = tf.reshape(X, (X.shape[0],)).numpy()
    ynumpy = tf.reshape(y, (y.shape[0],)).numpy()
    sortorder = tf.argsort(D).numpy()
    plot1d_functions(
        D, y, true=true, confidence={
            'x': D[sortorder],
            'y1': ten.numpy()[sortorder],
            'y2': ninety.numpy()[sortorder]})

    # (1) check 2d input -----------------------------------------
    bnn2d = BNN(hunits=[2, 10, 9, 8, 1])
    X = tf.constant([[1., 2.], [3., 4.]])
    param2d = bnn2d._layer_sample()
    bnn2d.forward(X=X, param=param2d)
    bnn2d.likelihood_model(X, param=param2d).sample()

print()