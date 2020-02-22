"""
Created on Mon Nov 4 2019
@author: T.Ruhkopf
@email:  tim.ruhkopf@outlook.de
"""

#     # CONSIDER! Estimation: fahrmeir kneib & Lang: p 547: centering &
#     appropriate penalty for joint estimation of both f(x), f(x,y), f(y) in
#     GAM case.

from time import gmtime, strftime

import os

# ensure relative paths on both server & local
pathroot = '$HOME'
# pathdata = pathroot + '/data'
pathresults = pathroot + '/results'
pathlogs = pathresults + '/logs'

print(pathroot)


def get_git_revision_hash():
    from subprocess import check_output
    return check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


githash = '{hash}_{timestamp}'.format(hash=get_git_revision_hash(),
                                      timestamp=strftime("%Y%m%d_%H%M%S", gmtime()))
print(githash)

import tensorflow as tf

x = tf.constant([1, 2, 3, 4, 5])
y = tf.constant([1, 2, 2, 1, 1])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y,
          epochs=5,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir=pathresults)])
model.evaluate(x, y)

model.save(filepath='/usr/users/truhkop/results/{}'.format(githash))

print('evaluated a NN in tf2.0')
