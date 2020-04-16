from itertools import accumulate, chain
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
        # FIXME: move sigma prior to prior distribution and make this tfd.Sample
        #  not tfd.JointDist...
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

            param = self.listparser(tensorlist[:-1])
            likelihood = self.likelihood_model(X, param)
            val = self.prior_log_prob(param) + \
                  tf.reduce_sum(likelihood.log_prob(sigma=tensorlist[-1], y=y))
            # print('logposterior: {}'.format(val))
            return val

        return BNN_log_prob

    def listparser(self, tensorlist):
        """
        method to return from sample_chain()'s expected format: list of tensors
        to hierarchical version, where each layer holds its own set of parameters

        :return: list of dicts, each containing the parameters of the
        corresponding layer.
        """
        dist = [h.joint for h in self.layers]
        # nameslist = list(chain(*[d._parameters['model'].keys() for d in dist]))
        nameslist = list([list(d._parameters['model'].keys()) for d in dist])

        c = [len(b) for b in nameslist]
        c = list(accumulate(c))
        c.insert(0, 0)

        return [{k: v for k, v in zip(names, tensorlist[i:j + 1])}
                for i, j, names in zip(c[:-1], c[1:], nameslist)]

    def flatten(self, param):
        """:param param: list of dicts of tensors corresponding to each layer
        :return: tuple: (list of tensors, corresponding names)"""
        flat = list(chain(*[list(d.values()) for d in param]))
        nameslist = list(chain(*[list(h.joint._parameters['model'].keys()) for h in self.layers]))

        return flat, nameslist

    def initialize_full_flat(self, X):
        """initialize a flat list of tensors from the hierachical prior model
        for HMC estimation. Additionally generate the flat list of names
        corresponding to the flat list of tensors. (usable for bijectors)"""
        # Notice: prior_draw does not include likelihood's sigma prior!
        param_init = self.prior_draw()
        flattened, nameslist = self.flatten(param_init)

        # Notice: likelihood sample is dict with y and sigma!
        like = bnn.likelihood_model(X, param_init).sample()
        flattened.append(like['sigma'])
        nameslist.append('sigma')

        print('Prior parameter initialization: {}, sigma: {}'.format(param_init, like['sigma']))

        init_param = param_init
        init_param.append(like['sigma'])

        return flattened, nameslist, init_param


if __name__ == '__main__':
    # (0) check 2d input -----------------------------------------
    bnn2d = BNN(hunits=[2, 10, 9, 8, 1])
    param2d = bnn2d.prior_draw()
    bnn2d.forward(X=tf.constant([[1., 2.], [3., 4.]]), param=param2d)
    bnn2d.likelihood_model(tf.constant([[1., 2.], [3., 4.]]), param=param2d).sample()

    # (1) (sampling 1d data from prior & fit) --------------------
    import seaborn as sns
    import matplotlib.pyplot as plt
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    bnn = BNN(hunits=[1, 10, 1], activation='sigmoid')

    # inialize true function from prior
    n = 2000
    X = tf.reshape(tf.linspace(-10., 10., n), (n, 1))
    param_true = bnn.prior_draw()
    mu_true = bnn.forward(X=X, param=param_true)
    true = bnn.likelihood_model(X, param=param_true).sample()
    y_true = true['y']
    print(true['sigma'])  # CAREFULL check that sigma is low value!

    # setting up parameters for estimation
    flattened, nameslist, param_init = bnn.initialize_full_flat(X)
    bijectors = {'tau': tfb.Exp(), 'W': tfb.Identity(), 'b': tfb.Identity(),
                 'sigma': tfb.Exp()}
    bnn.unnormalized_log_prob = bnn._closure_log_prob(X=X, y=y_true)
    print('init_param has log_prob: {}'.format(bnn.unnormalized_log_prob(*flattened)))

    y_init = bnn.forward(X, param_init)

    # Fitting with HMC
    adHMC = AdaptiveHMC(
        initial=flattened,  # CAREFULL MUST BE FLOAT!
        bijectors=[bijectors[name] for name in nameslist],
        log_prob=bnn.unnormalized_log_prob)

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(10e2),
        num_results=int(10e3),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    # mean prediction
    meanPost = adHMC.predict_mean()
    param = bnn.listparser(meanPost)
    y_map = bnn.forward(X, param)

    # Plot init, true, mean & sampled
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('init-, true-, mean function & sampled points')

    sns.lineplot(x=tf.reshape(X, (n,)).numpy(),
                 y=tf.reshape(mu_true, (n,)).numpy(), ax=ax)
    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(flattened).numpy()))

    sns.lineplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_init, (n,)).numpy(), ax=ax)
    sns.lineplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_map, (n,)).numpy(), ax=ax)
    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(meanPost).numpy()))

    sns.scatterplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_true, (n,)).numpy(), ax=ax)

print('')
