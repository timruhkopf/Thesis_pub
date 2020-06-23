"""draw one dimensional prior functions from BNN"""

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from Tensorflow.Bayesian.Models.Base.BNN import BNN

fig, axes = plt.subplots(nrows=4, ncols=4)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Sampling BNN Functions from prior')

bnn = BNN(hunits=[1, 10, 1], activation='sigmoid')

for ax in axes.flatten():
    param = bnn.prior_draw()
    flattened, _ = bnn.flatten(param)
    y = bnn.forward(tf.reshape(tf.range(-1., 10., 0.1), (110, 1)), param)

    # evaluate bnn functions drawn from prior
    sns.lineplot(x=tf.range(-1., 10., 0.1), y=tf.reshape(y, (110,)).numpy(), ax=ax)
