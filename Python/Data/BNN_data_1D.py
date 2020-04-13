import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class Data_BNN_1D:
    def __init__(self, n, grid=(0, 10, 0.5)):
        """One dimensional effect"""
        self.grid = grid
        self.n = n

        self.X = self.prior_X().sample((self.n,))

        # instantiate the data
        self.prior()  # Consider self.prior().sample()
        likelihood, mu = self.true_likelihood(self.X)
        self.y = likelihood.sample()
        self.mu = mu

    def prior_X(self):
        return tfd.Uniform(self.grid[0], self.grid[1])

    def prior(self):
        """HERE: drawing effect instance from prior"""
        from Python.Effects.Cases1D.Bspline_K import Bspline_K
        self.gmrf = Bspline_K(xgrid=self.grid)

        # FIXME: should return a proper tfd. to allow easy log_prob:
        #  alternative, if improper:
        #  return gmrf
        #  with methods:
        #  self.gmrf.sample() und self.gmrf.log_prob()
        #  BOTH MUST RETURN TF TENSOR

    def true_likelihood(self, X):
        """class of data, allows to generate data instances.
        :return <tuple> tfd for y, mu(X, gamma) """
        mu = tf.constant(self.gmrf.spl(X), dtype=tf.float32)
        likelihood = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=tf.repeat(1., mu.shape[0]))
        return likelihood, mu



if __name__ == '__main__':
    # 1D GAM EXAPMPLE
    from Python.Bayesian.Models.BNN import BNN

    # (0) (generating data) ----------------------------------
    bnn = BNN(hunits=[1, 10, 1], activation='tanh')
    n = 1000
    data = Data_BNN_1D(n=n)
    bnn.unnormalized_log_prob = bnn._closure_log_prob(
        X=tf.reshape(data.X, (n, 1)), y=tf.reshape(data.y, (n, 1)))

    # (1) (setting up posterior model) -----------------------
    # inital state (priors + likelihood's sigma prior)
    from itertools import chain

    param = bnn.prior_draw()
    flattened = list(chain(*[list(d.values()) for d in param]))
    flattened.append(bnn.likelihood_model(tf.reshape(data.X, (n, 1)), param).sample()['sigma'])
    bnn.unnormalized_log_prob(*flattened)

    nameslist = [list(h.joint._parameters['model'].keys()) for h in bnn.layers]

    # (2) (sampling) -----------------------------------------
    from Python.Bayesian.Samplers.AdaptiveHMC import AdaptiveHMC

    adHMC = AdaptiveHMC(  # FIXME:hmc.initial must be list of tensors, but Ws & bs are already list of tensors!!!
        # Notice: bnn.Ws already holds the init state, same for bnn.bs
        initial=flattened,  # CAREFULL MUST BE FLOAT!
        # bijectors=[*[tfb.Identity() for i in range(len(bnn.Ws) + len(bnn.bs))]  # Ws # bs
        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Identity(),  # ['tau', 'W', 'b']
                   # tfb.Exp(), tfb.Identity(), tfb.Identity(),  # ['tau', 'W', 'b']
                   # tfb.Exp(), tfb.Identity(), tfb.Identity(),  # ['tau', 'W', 'b']
                   tfb.Exp(), tfb.Identity(),  # ['tau', 'W']
                   tfb.Exp()],  # ['sigma']
        log_prob=bnn.unnormalized_log_prob)

    samples, traced = adHMC.sample_chain(
        num_burnin_steps=int(10e2),
        num_results=int(10e3),
        logdir='/home/tim/PycharmProjects/Thesis/TFResults')

    import pickle

    with open('bnn_samples.pkl', 'wb') as handle:
        pickle.dump({'samples': samples, 'traced': traced, 'true': param}, handle)

    # (3) (visualize it) ------------------------------------------
    # plot prediciton at MAP
    import seaborn as sns
    import matplotlib.pyplot as plt

    meanPost = [tf.reduce_mean(chain, axis=0) for chain in samples]
    param = bnn.argparser(meanPost)
    y_map = bnn.forward(tf.reshape(data.X, (n, 1)), param)

    param_true = bnn.argparser(flattened)
    y = bnn.forward(tf.reshape(data.X, (n, 1)), param)

    # FIXME: sns plot dies to to shape error
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('init & true function & sampled points')

    sns.lineplot(x=tf.reshape(data.X, (n,)), y=tf.reshape(y, (n,)).numpy(), ax=ax)
    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(flattened).numpy()))

    sns.lineplot(x=tf.reshape(data.X, (n,)), y=tf.reshape(y_map, (n,)).numpy(), ax=ax)
    # ax.set(title='log_prob' + str(bnn.unnormalized_log_prob(meanPost).numpy()))

    sns.scatterplot(x=tf.reshape(data.X, (n,)), y=tf.reshape(data.y, (n,)).numpy(), ax=ax)

print('')
