from itertools import accumulate, chain
from inspect import getfullargspec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from Python.Bayesian.layers.Hidden import HiddenFinal


class Regression(HiddenFinal):

    def __init__(self, *args, **kwargs):
        # self.joint holds the prior model (except for sigma)
        super().__init__(activation='identity', *args, **kwargs)
        self.joint = tfd.JointDistributionNamed(dict(
            tau=tfd.InverseGamma(1., 1.),

            W=lambda tau: tfd.Sample(
                distribution=tfd.Normal(0., tau),
                sample_shape=(self.no_units, self.input_shape)),

            sigma=tfd.InverseGamma(0.1, 0.1)
        ))

    def likelihood_model(self, X, W, sigma):
        """distributional assumption"""
        return tfd.Sample(tfd.Independent(
            tfd.Normal(loc=self.dense(X, W), scale=sigma)))

    # @tf.function
    def dense(self, X, W):
        return self.activation(tf.linalg.matvec(X, W))

    def listparser(self, tensorlist):
        nameslist = list(self.joint._parameters['model'].keys())
        return {k: v for k, v in zip(nameslist, tensorlist)}

    def _closure_log_prob(self, X, y):
        """A closure, to preset X, y in this model and
        match HMC's expected model format"""

        @tf.function  # NOTICE: seems to be ignored by autograph ( Cause: expected exactly one node node, found [<gast.gast.FunctionDef object at 0x7f59e006d940>, <gast.gast.Return object at 0x7f59e006da90>] )
        def BNN_log_prob(*tensorlist):
            """unnormalized log posterior value: log_priors + log_likelihood"""

            param = self.listparser(tensorlist)

            like_param = {k: param[k] for k in getfullargspec(self.likelihood_model).args[2:]}
            likelihood = self.likelihood_model(X, **like_param)

            val = self.joint.log_prob(**param) + \
                  tf.reduce_sum(likelihood.log_prob(y))
            print('logposterior: {}'.format(val))
            return val

        return BNN_log_prob


if __name__ == '__main__':
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    tfb = tfp.bijectors
    # (0) generate data ---------------------------------------
    xgrid = (0, 10)
    n = 100
    X = tf.stack(
        values=[tf.ones((n,)),
                tfd.Uniform(xgrid[0], xgrid[1]).sample((n,))],
        axis=1)

    # no units must correspond to X.shape[1]
    p = X.shape[1]
    reg = Regression(input_shape=p, no_units=1)
    param = {'tau': tf.constant(0.5), 'W': tf.constant([-0.5, 2]), 'sigma': tf.constant(0.25)}
    y = reg.likelihood_model(X, W=param['W'], sigma=param['sigma']).sample()



    # fixme: reshape y !
    # y = tf.reshape(y, (100, 1))
    reg.unnormalized_log_prob = reg._closure_log_prob(X, y)

    # check unnormalized log_prob
    XXinv = tf.linalg.inv(tf.linalg.matmul(X, X, transpose_a=True))
    OLS = tf.linalg.matvec(tf.linalg.matmul(XXinv, X, transpose_b=True), y)
    print(reg.unnormalized_log_prob(*[tf.constant(0.5), OLS, tf.constant(0.25)]))
    print(reg._closure_log_prob(X, y)(tf.constant([1.]),  # tau
                                      tf.constant([1., 1.]),  # W
                                      tf.constant([1.])))  # sigma

    reg.unnormalized_log_prob = reg._closure_log_prob(X, y)
    adHMC = AdaptiveHMC(initial=list(param.values()),  # CAREFULL MUST BE FLOAT!
                        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Exp()],  # [tfb.Identity(), tfb.Identity()]
                        log_prob=reg.unnormalized_log_prob)  # unnorm. log_prob

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(10e1),
        num_results=int(10e1),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    reg.listparser(samples)

    # s = [s for s in zip(*samples)]
    # post = tf.stack(list(map(lambda x: reg.unnormalized_log_prob(*x), [s for s in zip(*samples)])), axis=0)
    # maxpost_param = s[tf.argmax(post, axis=0).numpy()]
    # reg.unnormalized_log_prob(*s[0])

    # {getfullargspec(reg.dense).args[2:]}

    n = 100
    Xlin = tf.stack(
        values=[tf.ones((n,)),
                tf.linspace(0., 10., n)], axis=1)

    # MAP & mean aposteriori
    y_hat = reg.dense(Xlin, reg.listparser(adHMC.predict_mode(reg.unnormalized_log_prob))['W'])
    y_hat = reg.dense(Xlin, reg.listparser(adHMC.predict_mean())['W'])

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('init-, true-, mean function & sampled points')

    sns.scatterplot(
        x=X[:, 1].numpy(),
        y=tf.reshape(y, (100,)).numpy(), ax=ax)

    sns.lineplot(Xlin[:, 1], tf.reshape(y_hat, (100,)).numpy(), ax=ax)

    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(flattened).numpy()))
    #
    # sns.lineplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_init, (n,)).numpy(), ax=ax)
    # sns.lineplot(x=tf.reshape(X, (n,)).numpy(), y=tf.reshape(y_map, (n,)).numpy(), ax=ax)
    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(meanPost).numpy()))

    adHMC.predict_mean()

    # DEPREC: plotting the traces - problems:
    #  parameters in matrix form!
    #  possibly thousands of parameters
    # for var, var_samples in reg.listparser(samples).items():  # pooled_samples._asdict().items():
    #     adHMC.plot_traces(var, samples=var_samples, num_chains=4)

    # adHMC.plot_traces('beta', adHMC.chains, 2)
