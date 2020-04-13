import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from Python.Bayesian.layers.Hidden import Hidden, HiddenFinal


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

    # with @tf.function, forward does not work
    def prior_draw(self):
        """init the layer's parameters."""
        # dist = [h.joint for h in self.layers]
        return [h.joint.sample() for h in self.layers]

    @tf.function
    def prior_log_prob(self, param):
        return sum([tf.reduce_sum(h.joint.log_prob(**p))
                    for h, p in zip(self.layers, param)])

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

    # CAREFULL: cannot be @tf.function as is tfd.Dist not tf.Tensor!
    def likelihood_model(self, X, param):
        return tfd.JointDistributionNamed(dict(
            sigma=tfd.InverseGamma(0.1, 0.1),
            y=lambda sigma: tfd.Sample(tfd.Independent(
                tfd.Normal(loc=self.forward(X, param), scale=sigma)))
        ))

    def _closure_log_prob(self, X, y):
        """A closure, to preset X, y in this model and
        match HMC's expected model format"""

        @tf.function  # NOTICE: seems to be ignored by autograph ( Cause: expected exactly one node node, found [<gast.gast.FunctionDef object at 0x7f59e006d940>, <gast.gast.Return object at 0x7f59e006da90>] )
        def BNN_log_prob(*tensorlist):
            """unnormalized log posterior value: log_priors + log_likelihood"""

            param = self.argparser(tensorlist[:-1])
            likelihood = self.likelihood_model(X, param)
            val = self.prior_log_prob(param) + \
                  tf.reduce_sum(likelihood.log_prob(sigma=tensorlist[-1], y=y))
            print('logposterior: {}'.format(val))
            return val

        return BNN_log_prob

    def argparser(self, tensorlist):
        dist = [h.joint for h in self.layers]
        # nameslist = list(chain(*[d._parameters['model'].keys() for d in dist]))
        nameslist = list([list(d._parameters['model'].keys()) for d in dist])

        from itertools import accumulate
        c = [len(b) for b in nameslist]
        c = list(accumulate(c))
        c.insert(0, 0)

        return [{k: v for k, v in zip(names, tensorlist[i:j + 1])}
                for i, j, names in zip(c[:-1], c[1:], nameslist)]


if __name__ == '__main__':
    # (0) check 2d input -----------------------------------------
    bnn2d = BNN(hunits=[2, 10, 9, 8, 1])
    param2d = bnn2d.prior_draw()
    bnn2d.forward(X=tf.constant([[1., 2.], [3., 4.]]), param=param2d)
    bnn2d.likelihood_model(tf.constant([[1., 2.], [3., 4.]]), param=param2d).sample()

    # (1) (sampling 1d data from prior & fit) --------------------
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import chain
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    bnn = BNN(hunits=[1, 10, 1], activation='sigmoid')
    param_true = bnn.prior_draw()
    X = tf.reshape(tf.range(-10, 10, 0.01), (2000, 1))
    n = 2000
    mu_true = bnn.forward(X=X, param=param_true)
    y_true = bnn.likelihood_model(X, param=param_true).sample()

    # CAREFULL check that sigma is low value!
    y_true['sigma']
    bnn.unnormalized_log_prob = bnn._closure_log_prob(X=X, y=y_true['y'])

    # setting up parameters for estimation
    param_init = bnn.prior_draw()
    flattened = list(chain(*[list(d.values()) for d in param_init]))
    flattened.append(bnn.likelihood_model(X, param_init).sample()['sigma'])
    print('init_param has log_prob: {}'.format(bnn.unnormalized_log_prob(*flattened)))

    nameslist = [list(h.joint._parameters['model'].keys()) for h in bnn.layers]
    bnn.likelihood_model(X, param=param_init).sample()

    # fitting with HMC
    # CAREFULL: PARAMLIST instead of explicit variables!
    adHMC = AdaptiveHMC(
        initial=flattened,  # CAREFULL MUST BE FLOAT!
        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Identity(),  # ['tau', 'W', 'b']
                   # tfb.Exp(), tfb.Identity(), tfb.Identity(),  # ['tau', 'W', 'b']
                   # tfb.Exp(), tfb.Identity(), tfb.Identity(),  # ['tau', 'W', 'b']
                   tfb.Exp(), tfb.Identity(),  # ['tau', 'W']
                   tfb.Exp()],  # ['sigma'],
        log_prob=bnn.unnormalized_log_prob)

    samples, traced = adHMC.sample_chain(num_burnin_steps=int(10e2), num_results=int(10e3),
                                         logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    meanPost = [tf.reduce_mean(chain, axis=0) for chain in samples]
    param = bnn.argparser(meanPost)
    y_map = bnn.forward(X, param)


    y_init = bnn.forward(X, param_init)

    # FIXME: sns plot dies to to shape error
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('init & true function & sampled points')

    sns.lineplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(mu_true, (n,)).numpy(), ax=ax)
    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(flattened).numpy()))

    sns.lineplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_init, (n,)).numpy(), ax=ax)
    sns.lineplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_map, (n,)).numpy(), ax=ax)
    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(meanPost).numpy()))

    sns.scatterplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_true['y'], (n,)).numpy(), ax=ax)

    print('')


    # # (2) sample prior functions -----------------------------------------------
    # # (2.1) (1D Data)
    # from Python.Data.BNN_data import Data_BNN_1D
    #
    # data = Data_BNN_1D(n=100)
    # bnn.unnormalized_log_prob = bnn._closure_log_prob(
    #     X=tf.reshape(data.X, (100, 1)), y=data.y)
    #
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(nrows=4, ncols=4)
    # fig.subplots_adjust(hspace=0.5)
    # fig.suptitle('Sampling BNN Functions from prior')
    #
    # for ax in axes.flatten():
    #     bnn._initialize_from_prior()
    #     y = bnn.forward(X=tf.reshape(tf.range(-1., 10., 0.1), (110, 1)), Ws=bnn.Ws, bs=bnn.bs)
    #
    #     # evaluate bnn functions drawn from prior
    #     sns.lineplot(x=tf.range(-1., 10., 0.1), y=tf.reshape(y, (110,)).numpy(), ax=ax)
    #     ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(bnn.Ws, bnn.bs).numpy()))

    # TODO (2.2) (2D Data)
    from Python.Data.BNN_data_1D import Data_BNN_2D

